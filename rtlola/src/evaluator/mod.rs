pub mod config;
mod event_driven_manager;
mod io_handler;
mod time_driven_manager;

use crate::evaluator::{
    config::*, event_driven_manager::EventDrivenManager, io_handler::*,
    time_driven_manager::TimeDrivenManager,
};
use crate::util;
use lola_parser::{Duration, LolaIR, Stream, StreamReference};

pub struct Evaluator {
    /// Handles all kind of output behavior according to config.
    output_handler: OutputHandler,
    /// Handles input events.
    input_handler: InputHandler,

    /// Manages correct handling of time-driven output streams.
    time_manager: TimeDrivenManager,
    /// Manages correct handling of event-driven output streams.
    event_manager: EventDrivenManager,

    /// States greatest common denominator of extend periods.
    extend_period: Duration,

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
        // TODO: Return Result<Evaluator>
        let output_handler = OutputHandler::new(&config);
        let input_handler = InputHandler::new(&config);
        let time_manager = TimeDrivenManager::new(&ir);
        let event_manager = EventDrivenManager::new(&ir);
        let extend_period = Evaluator::find_extend_period(&ir);

        Evaluator {
            output_handler,
            input_handler,
            extend_period,
            time_manager,
            event_manager,
            spec: ir,
        }
    }

    /// Starts the evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    pub fn start(&mut self) {
        unimplemented!()
    }

    //////////// Private Functions ////////////

    fn find_extend_period(ir: &LolaIR) -> Duration {
        assert!(!ir.time_outputs.is_empty());
        let rates: Vec<u128> = ir
            .time_outputs
            .iter()
            .flat_map(|s| s.extend_rate.map(|r| r.0))
            .collect();
        let res = rates.iter().fold(rates[0], |a, b| util::gcd(a, *b));
        Duration(res)
    }
}

mod tests {
    use crate::evaluator::Evaluator;
    use lola_parser::{Duration, LolaIR};

    fn to_ir(spec: &str) -> LolaIR {
        unimplemented!("We need some interface from the parser.")
    }

    #[test]
    #[ignore]
    fn test_extension_rate_extraction() {
        let input = "input a: UInt8\n";
        let hz50 = "output b: UInt8 {extend @50Hz} := a";
        let hz40 = "output b: UInt8 {extend @40Hz} := a";
        let ms20 = "output b: UInt8 {extend @20ms} := a"; // 5Hz
        let ms1 = "output b: UInt8 {extend @1ms} := a"; // 100Hz

        let case1 = (format!("{}{}", input, hz50), 2_000);
        let case2 = (format!("{}{}{}", input, hz50, hz50), 20_000);
        let case3 = (format!("{}{}{}", input, hz50, hz40), 5_000);
        let case4 = (format!("{}{}{}", input, hz50, ms1), 1_000);
        let case5 = (format!("{}{}{}{}", input, hz50, ms20, ms1), 1_000);

        let cases = [case1, case2, case3, case4, case5];
        for (spec, expected) in cases.iter() {
            let was = Evaluator::find_extend_period(&to_ir(spec));
            assert_eq!(*expected, was.0);
        }
    }
}
