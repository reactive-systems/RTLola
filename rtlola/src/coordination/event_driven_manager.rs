use crate::basics::{EvalConfig, InputReader, OutputHandler};
use crate::coordination::WorkItem;
use crate::storage::Value;

use lola_parser::ir::{LolaIR, Stream, StreamReference, Type};
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

        if ir.event_driven.is_empty() {
            return EDM { current_cycle: 0.into(), layers: vec![Vec::new()], out_handler, input_reader, in_types };
        }

        // Zip eval layer with stream reference.
        let streams_with_layers: Vec<(usize, StreamReference)> =
            ir.get_event_driven().into_iter().map(|s| (s.eval_layer() as usize, s.as_stream_ref())).collect();

        // Streams are annotated with an evaluation layer. The layer is not minimal, so there might be
        // layers without entries and more layers than streams.
        // Minimization works as follows:
        // a) Find the greatest layer
        // b) For each potential layer...
        // c) Find streams that would be in it.
        // d) If there is none, skip this layer
        // e) If there are some, add them as layer.

        // a) Find the greatest layer. Maximum must exist because vec cannot be empty.
        let max_layer = streams_with_layers.iter().max_by_key(|(layer, _)| layer).unwrap().0;

        let mut layers = Vec::new();
        // b) For each potential layer
        for i in 0..=max_layer {
            // c) Find streams that would be in it.
            let in_layer_i: Vec<StreamReference> =
                streams_with_layers.iter().filter_map(|(l, r)| if *l == i { Some(*r) } else { None }).collect();
            if in_layer_i.is_empty() {
                // d) If there is none, skip this layer
                continue;
            } else {
                // e) If there are some, add them as layer.
                layers.push(in_layer_i);
            }
        }

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
                            .expect(format!("Failed to parse {} as value of type {:?}.", s, t).as_str());
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
                    let nanos = f * (nanos_per_sec as f64);
                    let nanos = nanos as u128;
                    let secs = (nanos / (nanos_per_sec as u128)) as u64;
                    let nanos = (nanos % (nanos_per_sec as u128)) as u32;
                    UNIX_EPOCH + Duration::new(secs, nanos)
                }
                _ => panic!("Time stamps need to be unsigned integers."),
            };

            if start_time.is_none() {
                start_time = Some(now);
                let _ = work_queue.send(WorkItem::Start(now));
            }

            // Inform the time driven manager first.
            match time_chan.send(now) {
                Err(e) => panic!("Problem with TDM! {:?}", e),
                Ok(_) => {}
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
