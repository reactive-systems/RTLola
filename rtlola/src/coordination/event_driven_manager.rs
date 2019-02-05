use crate::basics::{EvalConfig, InputReader, OutputHandler};
use crate::coordination::WorkItem;
use crate::storage::Value;

use lola_parser::{LolaIR, Stream, StreamReference, Type};
use std::ops::AddAssign;
use std::sync::mpsc::Sender;

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

        EDM { current_cycle: 0.into(), layers, out_handler, input_reader, in_types }
    }

    pub(crate) fn start(mut self, work_queue: Sender<WorkItem>) -> ! {
        let mut buffer = vec![String::new(); self.in_types.len()];
        loop {
            // This whole function is awful.
            match self.input_reader.read_blocking(&mut buffer) {
                Ok(true) => {}
                Ok(false) => {
                    let _ = work_queue.send(WorkItem::End); // Whether it fails or not, we really don't care.
                                                            // Sleep until you slowly fade into non-existence...
                    loop {
                        std::thread::sleep(std::time::Duration::new(u64::max_value(), 0))
                    }
                }
                Err(e) => panic!("Error reading data. {}", e),
            }
            let layers = self.layers.clone(); // TODO: Sending repeatedly is unnecessary.
            let event = buffer.iter()
                .zip(self.in_types.iter())
                .map(|(s, t)| Value::try_from(s, t).unwrap()) // TODO: Handle parse error.
                .enumerate()
                .map(|(ix, v)| (StreamReference::InRef(ix), v))
                .collect();
            let item = EventEvaluation { event, layers };
            match work_queue.send(WorkItem::Event(item)) {
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