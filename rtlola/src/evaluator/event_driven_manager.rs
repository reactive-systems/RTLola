use lola_parser::{LolaIR, Stream, StreamReference};
use std::ops::AddAssign;

/// Represents the current cycle count for event-driven events.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct EventDrivenCycleCount(u128);

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
}

impl EventDrivenManager {
    /// Creates a new EventDrivenManager managing event-driven output streams.
    pub fn new(ir: &LolaIR) -> EventDrivenManager {
        // Zip eval layer with stream reference.
        let inputs = ir
            .inputs
            .iter()
            .map(|i| (i.eval_layer() as usize, i.as_stream_ref()));
        let ir_outputs = ir.outputs(); // Extend lifetime.
        let outputs = ir_outputs
            .iter()
            .map(|r| (ir.get_out(*r).eval_layer() as usize, *r));
        let layered: Vec<(usize, StreamReference)> = inputs.chain(outputs).collect();
        let max_layer = layered.iter().map(|(lay, r)| lay).max();

        assert!(Self::check_layers(
            &layered.iter().map(|(ix, r)| *ix).collect()
        ));

        // Create vec where each element represents one layer.
        // `check_layers` guarantees that max_layer is defined.
        let mut layers = vec![Vec::new(); *max_layer.unwrap()];
        for (ix, stream) in layered {
            layers[ix].push(stream)
        }

        EventDrivenManager {
            current_cycle: 0.into(),
            layer_counter: 0,
            layers,
        }
    }

    fn check_layers(vec: &Vec<usize>) -> bool {
        if vec.is_empty() {
            return false;
        }
        let mut indices = vec.clone();
        indices.sort();
        let successive = indices.iter().enumerate().all(|(ix, key)| ix == *key);
        let starts_at_0 = *indices.first().unwrap() == 0 as usize; // Fail for empty.
        successive && starts_at_0
    }

    /// Returns a collection of all event-driven streams that need to be extended next according to
    /// the evaluation order for evaluation cycle `for_cycle`. Returns `None` if the evaluation
    /// cycle is over.
    /// `panics` if the old cycle is not completed before the next one is requested.
    pub fn next_evaluation_layer(
        &mut self,
        for_cycle: EventDrivenCycleCount,
    ) -> Option<&[StreamReference]> {
        assert_eq!(
            for_cycle, self.current_cycle,
            "Requested new eval cycle before completing the last one."
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
