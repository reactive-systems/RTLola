use crate::basics::{EvalConfig, InputReader, OutputHandler};
use crate::coordination::WorkItem;
use crate::storage::Value;

use lola_parser::{LolaIR, Stream, StreamReference, Type};
use std::ops::AddAssign;
use std::sync::mpsc::Sender;

/// Represents the current cycle count for event-driven events.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
struct EventDrivenCycleCount(u128);

type EDCC = EventDrivenCycleCount;

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
        let layered: Vec<(usize, StreamReference)> =
            ir.get_event_driven().into_iter().map(|s| (s.eval_layer() as usize, s.as_stream_ref())).collect();
        let max_layer = layered.iter().map(|(lay, _)| lay).max();

        if cfg!(debug_assertions) {
            Self::check_layers(&layered.iter().map(|(ix, _)| *ix).collect());
        }

        // Create vec where each element represents one layer.
        // `check_layers` guarantees that max_layer is defined.
        let mut layers = vec![Vec::new(); *max_layer.unwrap()];
        for (ix, stream) in layered {
            layers[ix].push(stream)
        }

        EDM { current_cycle: 0.into(), layers, out_handler, input_reader, in_types }
    }

    pub(crate) fn start(mut self, work_queue: Sender<WorkItem>) -> ! {
        let mut buffer = vec![String::new(); self.in_types.len()];
        loop {
            // This whole function is awful.
            let _ = self.input_reader.read_blocking(&mut buffer); // TODO: Handle error.
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
                Err(e) => self.out_handler.runtime_warning(|| "Error when sending work item."),
            }
            self.current_cycle += 1;
        }
    }

    fn check_layers(vec: &Vec<usize>) {
        let mut indices = vec.clone();
        indices.sort();
        let successive = indices.iter().enumerate().all(|(ix, key)| ix == *key);
        debug_assert!(successive, "Evaluation order not minimal: Some layers do not have entries.");
        let starts_at_0 = *indices.first().unwrap() == 0 as usize; // Fail for empty.
        debug_assert!(starts_at_0, "Evaluation order not minimal: There are no streams in layer 0.");
    }
}

pub(crate) struct EventEvaluation {
    pub(crate) event: Vec<(StreamReference, Value)>,
    pub(crate) layers: Vec<Vec<StreamReference>>,
}
