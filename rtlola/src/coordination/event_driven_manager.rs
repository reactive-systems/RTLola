use crate::basics::{EvalConfig, InputReader, OutputHandler};
use crate::coordination::WorkItem;
use crate::storage::Value;

use lola_parser::ir::{FloatTy, LolaIR, StreamReference, Type, UIntTy};
use std::ops::AddAssign;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub(crate) type EventEvaluation = Vec<(StreamReference, Value)>;

/// Represents the current cycle count for event-driven events.
//TODO(marvin): u128? wouldn't u64 suffice?
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
struct EventDrivenCycleCount(u128);

type EDM = EventDrivenManager;

impl From<u128> for EventDrivenCycleCount {
    fn from(i: u128) -> EventDrivenCycleCount {
        EventDrivenCycleCount(i)
    }
}

impl AddAssign<u128> for EventDrivenCycleCount {
    fn add_assign(&mut self, i: u128) {
        *self = EventDrivenCycleCount(self.0 + i)
    }
}

pub struct EventDrivenManager {
    current_cycle: EventDrivenCycleCount,
    out_handler: OutputHandler,
    input_reader: InputReader,
    in_types: Vec<Type>,
}

impl EventDrivenManager {
    /// Creates a new EventDrivenManager managing event-driven output streams.
    pub(crate) fn setup(ir: LolaIR, config: EvalConfig) -> EventDrivenManager {
        let out_handler = OutputHandler::new(&config);
        let stream_names: Vec<&str> = ir.inputs.iter().map(|i| i.name.as_str()).collect();
        let input_reader = InputReader::from(config.source, stream_names.as_slice());
        let input_reader = match input_reader {
            Ok(r) => r,
            Err(e) => panic!("Cannot create input reader: {}", e),
        };

        let in_types = ir.inputs.iter().map(|i| i.ty.clone()).collect();

        EDM { current_cycle: 0.into(), out_handler, input_reader, in_types }
    }

    fn read_event(&mut self) -> bool {
        self.input_reader.read_blocking().unwrap_or_else(|e| panic!("Error reading data. {}", e))
    }

    fn get_event(&self) -> Vec<(StreamReference, Value)> {
        self.in_types
            .iter()
            .enumerate()
            .flat_map(|(ix, t)| {
                let s = self.input_reader.str_ref_for_stream_ix(ix);
                if s == "#" {
                    None
                } else {
                    let v = Value::try_from(s, t)
                        .unwrap_or_else(|| panic!("Failed to parse {} as value of type {:?}.", s, t));
                    Some((StreamReference::InRef(ix), v))
                }
            })
            .collect()
    }

    fn get_time(&self) -> SystemTime {
        let s = self.input_reader.str_ref_for_time();
        //TODO(marvin): fix this typing issue
        let mut v = Value::try_from(s, &Type::Float(FloatTy::F64));
        if v.is_none() {
            v = Value::try_from(s, &Type::UInt(UIntTy::U64));
        }
        let v = v.unwrap_or_else(|| panic!("Failed to parse time string {}.", s));

        //TODO(marvin): simplify time computation
        match v {
            Value::Unsigned(u) => UNIX_EPOCH + Duration::from_secs(u as u64),
            Value::Float(f) => {
                let f: f64 = (f).into();
                let nanos_per_sec: u32 = 1_000_000_000;
                let nanos = f * (f64::from(nanos_per_sec));
                let nanos = nanos as u128;
                let secs = (nanos / (u128::from(nanos_per_sec))) as u64;
                let nanos = (nanos % (u128::from(nanos_per_sec))) as u32;
                UNIX_EPOCH + Duration::new(secs, nanos)
            }
            _ => panic!("Invalid time stamp value type: {:?}", v),
        }
    }

    pub(crate) fn start(
        mut self,
        offline: bool,
        work_queue: Sender<WorkItem>,
        send_time: bool,
        time_chan: Sender<SystemTime>,
        ack_chan: Receiver<()>,
    ) -> ! {
        let mut start_time: Option<SystemTime> = None;
        loop {
            if !self.read_event() {
                let _ = work_queue.send(WorkItem::End); // Whether it fails or not, we really don't care.
                                                        // Sleep until you slowly fade into nothingness...
                loop {
                    std::thread::sleep(std::time::Duration::new(u64::max_value(), 0))
                }
            }
            let event = self.get_event();

            let time = if offline { self.get_time() } else { SystemTime::now() };

            if offline {
                if start_time.is_none() {
                    start_time = Some(time);
                    let _ = work_queue.send(WorkItem::Start(time));
                }

                if send_time {
                    // Inform the time driven manager first.
                    if let Err(e) = time_chan.send(time) {
                        panic!("Problem with TDM! {:?}", e)
                    }
                    let _ = ack_chan.recv(); // Wait until be get the acknowledgement.
                }
            }

            match work_queue.send(WorkItem::Event(event, time)) {
                Ok(_) => {}
                Err(e) => self.out_handler.runtime_warning(|| format!("Error when sending work item. {}", e)),
            }
            self.current_cycle += 1;
        }
    }
}
