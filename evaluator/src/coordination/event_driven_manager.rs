use crate::basics::{EvalConfig, InputReader, OutputHandler};
use crate::coordination::WorkItem;
use crate::storage::Value;
use crossbeam_channel::Sender;
use std::error::Error;
use std::ops::AddAssign;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use streamlab_frontend::ir::{LolaIR, Type};

pub(crate) type EventEvaluation = Vec<Value>;

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
    out_handler: Arc<OutputHandler>,
    input_reader: InputReader,
    in_types: Vec<Type>,
}

impl EventDrivenManager {
    /// Creates a new EventDrivenManager managing event-driven output streams.
    pub(crate) fn setup(ir: LolaIR, config: EvalConfig, out_handler: Arc<OutputHandler>) -> EventDrivenManager {
        let stream_names: Vec<&str> = ir.inputs.iter().map(|i| i.name.as_str()).collect();
        let input_reader = InputReader::from(config.source, stream_names.as_slice());
        let input_reader = match input_reader {
            Ok(r) => r,
            Err(e) => panic!("Cannot create input reader: {}", e),
        };

        let in_types: Vec<Type> = ir.inputs.iter().map(|i| i.ty.clone()).collect();

        EDM { current_cycle: 0.into(), out_handler, input_reader, in_types }
    }

    fn has_event(&mut self) -> bool {
        self.input_reader.read_blocking().unwrap_or_else(|e| panic!("Error reading data. {}", e))
    }

    fn read_event(&self) -> Vec<Value> {
        let mut buffer = vec![Value::None; self.in_types.len()];
        for (col_ix, s) in self.input_reader.record.iter().enumerate() {
            if let Some(str_ix) = self.input_reader.mapping.col2str[col_ix] {
                if s != "#" {
                    let t = &self.in_types[str_ix];
                    buffer[str_ix] = Value::try_from(s, t)
                        .unwrap_or_else(|| panic!("Failed to parse {} as value of type {:?}.", s, t))
                }
            }
        }
        buffer
    }

    fn get_time(&mut self) -> SystemTime {
        let str_time = self.input_reader.str_for_time();
        let float_secs: f64 = match str_time.parse() {
            Ok(f) => f,
            Err(e) => panic!("Failed to parse time string {}: {}", str_time, e),
        };
        const NANOS_PER_SEC: f64 = 1_000_000_000.0;
        let float_nanos = float_secs * NANOS_PER_SEC;
        let nanos = float_nanos as u64;
        UNIX_EPOCH + Duration::from_nanos(nanos)
    }

    pub(crate) fn start_online(mut self, work_queue: Sender<WorkItem>) -> ! {
        loop {
            if !self.has_event() {
                let _ = work_queue.send(WorkItem::End); // Whether it fails or not, we really don't care.
                                                        // Sleep until you slowly fade into nothingness...
                loop {
                    std::thread::sleep(std::time::Duration::new(u64::max_value(), 0))
                }
            }
            let event = self.read_event();
            let time = SystemTime::now();
            match work_queue.send(WorkItem::Event(event, time)) {
                Ok(_) => {}
                Err(e) => self.out_handler.runtime_warning(|| format!("Error when sending work item. {}", e)),
            }
            self.current_cycle += 1;
        }
    }

    pub(crate) fn start_offline(mut self, work_queue: Sender<WorkItem>) -> Result<(), Box<dyn Error>> {
        let mut start_time: Option<SystemTime> = None;
        loop {
            if !self.has_event() {
                let _ = work_queue.send(WorkItem::End);
                return Ok(());
            }
            let event = self.read_event();
            let time = self.get_time();
            if start_time.is_none() {
                start_time = Some(time);
                let _ = work_queue.send(WorkItem::Start(time));
            }

            match work_queue.send(WorkItem::Event(event, time)) {
                Ok(_) => {}
                Err(e) => self.out_handler.runtime_warning(|| format!("Error when sending work item. {}", e)),
            }
            self.current_cycle += 1;
        }
    }
}
