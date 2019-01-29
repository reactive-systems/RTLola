use crate::basics::{ OutputHandler, EvalConfig };
use crate::coordination::WorkItem;

use lola_parser::{LolaIR, Stream, StreamReference};
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
    layer_counter: usize,
    layers: Vec<Vec<StreamReference>>,
    handler: OutputHandler,
}

impl EventDrivenManager {
    /// Creates a new EventDrivenManager managing event-driven output streams.
    pub(crate) fn setup(ir: LolaIR, config: EvalConfig) -> EventDrivenManager {

        let handler = OutputHandler::new(&config);

        if ir.event_driven.is_empty() {
            return EDM { current_cycle: 0.into(), layer_counter: 0, layers: vec![Vec::new()], handler };
        }

        // Zip eval layer with stream reference.
        let layered = ir
            .get_event_driven()
            .into_iter()
            .map(|o| o as &Stream)
            .chain(ir.inputs.iter().map(|i| i as &Stream))
            .map(|s| (s.eval_layer() as usize, s.as_stream_ref()));
        let max_layer = layered.clone().map(|(lay, _)| lay).max();
        let layered: Vec<(usize, StreamReference)> = layered.collect();

        if cfg!(debug_assertions) {
            Self::check_layers(&layered.iter().map(|(ix, _)| *ix).collect());
        }

        // Create vec where each element represents one layer.
        // `check_layers` guarantees that max_layer is defined.
        let mut layers = vec![Vec::new(); max_layer.unwrap()];
        for (ix, stream) in layered {
            layers[ix].push(stream)
        }

        EDM { current_cycle: 0.into(), layer_counter: 0, layers, handler }
    }

    pub(crate) fn start(self, work_queue: Sender<WorkItem>) -> ! {
        unimplemented!()
    }

    fn check_layers(vec: &Vec<usize>) {
        let mut indices = vec.clone();
        indices.sort();
        let successive = indices.iter().enumerate().all(|(ix, key)| ix == *key);
        debug_assert!(successive, "Evaluation order not minimal: Some layers do not have entries.");
        let starts_at_0 = *indices.first().unwrap() == 0 as usize; // Fail for empty.
        debug_assert!(starts_at_0, "Evaluation order not minimal: There are no streams in layer 0.");
    }

    /// Returns a collection of all event-driven streams that need to be extended next according to
    /// the evaluation order for evaluation cycle `for_cycle`. Returns `None` if the evaluation
    /// cycle is over.
    /// `panic`s if the old cycle is not completed before the next one is requested.
    fn next_evaluation_layer(&mut self, for_cycle: EventDrivenCycleCount) -> Option<&[StreamReference]> {
        debug_assert_eq!(
            for_cycle, self.current_cycle,
            "Requested new evaluation cycle before completing the last one."
        );
        if self.layer_counter as usize == self.layers.len() {
            self.progress_cycle();
            None
        } else {
            self.layer_counter += 1;
            Some(&self.layers[self.layer_counter - 1])
        }
    }

    fn progress_cycle(&mut self) {
        self.layer_counter = 0;
        self.current_cycle += 1;
    }
}

pub(crate) struct EventEvaluation {
    pub(crate) event: Vec<(StreamReference, String)>,
    pub(crate) layers: Vec<Vec<StreamReference>>,
}