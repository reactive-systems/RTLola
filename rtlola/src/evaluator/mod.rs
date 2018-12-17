pub mod config;
mod event_driven_manager;
mod io_handler;
mod time_driven_manager;

use crate::evaluator::{
    config::*, event_driven_manager::EventDrivenManager, io_handler::*, time_driven_manager::TimeDrivenManager,
};
use lola_parser::LolaIR;

pub struct Evaluator {
    /// Handles all kind of output behavior according to config.
    output_handler: OutputHandler,
    /// Handles input events.
    input_handler: InputHandler,

    /// Manages correct handling of time-driven output streams.
    time_manager: TimeDrivenManager,
    /// Manages correct handling of event-driven output streams.
    event_manager: EventDrivenManager,

    /// Intermediate representation of input Lola specification.
    spec: LolaIR,
}

impl lola_parser::LolaBackend for Evaluator {
    fn supported_feature_flags() -> Vec<lola_parser::FeatureFlag> {
        unimplemented!()
    }
}

impl Evaluator {
    /// Create a new `Evaluator` for RTLola specifications. Respects settings passed in `config`.
    pub fn new(ir: LolaIR, config: EvalConfig) -> Evaluator {
        // TODO: Return Result<Evaluator>, check streams empty etc.
        let output_handler = OutputHandler::new(&config);
        let input_handler = InputHandler::new(&config);
        let time_manager = TimeDrivenManager::new(&ir);
        let event_manager = EventDrivenManager::new(&ir);


        Evaluator { output_handler, input_handler, time_manager, event_manager, spec: ir }
    }

    /// Starts the evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    pub fn start(&mut self) {
        unimplemented!()
    }

}

