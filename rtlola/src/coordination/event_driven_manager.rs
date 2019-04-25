use crate::basics::{EvalConfig, InputReader, OutputHandler};
use crate::coordination::WorkItem;
use crate::storage::Value;

use lola_parser::ir::{LolaIR, StreamReference, Type};
use std::ops::AddAssign;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Represents the current cycle count for event-driven events.
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
    layers: Vec<Vec<StreamReference>>,
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

        let layers = ir.get_event_driven_layers();
        out_handler.debug(|| format!("Evaluation layers: {:?}", layers));

        EDM { current_cycle: 0.into(), layers, out_handler, input_reader, in_types }
    }

    pub(crate) fn start_online(mut self, work_queue: Sender<WorkItem>) -> ! {
        loop {
            let event = match self.read_event() {
                None => {
                    let _ = work_queue.send(WorkItem::End); // Whether it fails or not, we really don't care.
                                                            // Sleep until you slowly fade into nothingness...
                    loop {
                        std::thread::sleep(std::time::Duration::new(u64::max_value(), 0))
                    }
                }
                Some(e) => e,
            };

            let layers = self.layers.clone(); // TODO: Sending repeatedly is unnecessary.
            let event = event.into_iter().flatten().collect(); // Remove non-existing values.
            let item = EventEvaluation { event, layers };
            match work_queue.send(WorkItem::Event(item, SystemTime::now())) {
                Ok(_) => {}
                Err(e) => self.out_handler.runtime_warning(|| format!("Error when sending work item. {}", e)),
            }
            self.current_cycle += 1;
        }
    }

    fn read_event(&mut self) -> Option<Vec<Option<(StreamReference, Value)>>> {
        let mut buffer = vec![String::new(); self.in_types.len()];
        match self.input_reader.read_blocking(&mut buffer) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(e) => panic!("Error reading data. {}", e),
        }

        Some(
            buffer
                .iter()
                .zip(self.in_types.iter())
                .enumerate()
                .map(|(ix, (s, t))| {
                    if s == "#" {
                        None
                    } else {
                        let v = Value::try_from(s, t)
                            .unwrap_or_else(|| panic!("Failed to parse {} as value of type {:?}.", s, t));
                        Some((ix, v))
                    }
                })
                .map(|opt| opt.map(|(ix, v)| (StreamReference::InRef(ix), v)))
                .collect(),
        )
    }

    pub(crate) fn start_offline(
        mut self,
        work_queue: Sender<WorkItem>,
        time_chan: Sender<SystemTime>,
        ack_chan: Receiver<()>,
    ) -> ! {
        let time_ix = self.input_reader.time_index().unwrap();
        let mut start_time: Option<SystemTime> = None;
        loop {
            let event = match self.read_event() {
                None => {
                    let _ = work_queue.send(WorkItem::End); // Whether it fails or not, we really don't care.
                                                            // Sleep until you slowly fade into nothingness...
                    loop {
                        std::thread::sleep(std::time::Duration::new(u64::max_value(), 0))
                    }
                }
                Some(e) => e,
            };
            let time_value = event[time_ix].as_ref().map(|s| &s.1).expect("Timestamp needs to be present.");
            let now = match time_value {
                Value::Unsigned(u) => UNIX_EPOCH + Duration::from_secs(*u as u64),
                Value::Float(f) => {
                    let f: f64 = (*f).into();
                    let nanos_per_sec: u32 = 1_000_000_000;
                    let nanos = f * (f64::from(nanos_per_sec));
                    let nanos = nanos as u128;
                    let secs = (nanos / (u128::from(nanos_per_sec))) as u64;
                    let nanos = (nanos % (u128::from(nanos_per_sec))) as u32;
                    UNIX_EPOCH + Duration::new(secs, nanos)
                }
                _ => panic!("Time stamps need to be unsigned integers."),
            };

            if start_time.is_none() {
                start_time = Some(now);
                let _ = work_queue.send(WorkItem::Start(now));
            }

            // Inform the time driven manager first.
            if let Err(e) = time_chan.send(now) {
                panic!("Problem with TDM! {:?}", e)
            }
            let _ = ack_chan.recv(); // Wait until be get the acknowledgement.

            let event = event.into_iter().flatten().collect(); // Remove non-existing entries.
            let layers = self.layers.clone(); // TODO: Sending repeatedly is unnecessary.
            let item = EventEvaluation { event, layers };
            match work_queue.send(WorkItem::Event(item, now)) {
                Ok(_) => {}
                Err(e) => self.out_handler.runtime_warning(|| format!("Error when sending work item. {}", e)),
            }
            self.current_cycle += 1;
        }
    }
}

#[derive(Debug)]
pub(crate) struct EventEvaluation {
    pub(crate) event: Vec<(StreamReference, Value)>,
    pub(crate) layers: Vec<Vec<StreamReference>>,
}
