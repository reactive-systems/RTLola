
pub mod config;
mod io_handler;

use lola_parser::{LolaIR, StreamReference, Duration, Stream};
use crate::evaluator::config::*;
use crate::evaluator::io_handler::*;
use std::collections::HashMap;
use crate::util;

struct TimeDrivenManager {

}

/// Represents the current cycle count for time-driven events. `u128` is sufficient to represent
/// 10^22 years of runtime for evaluation cycles that are 1ns apart.
struct TimeDrivenCycleCount(u128);

impl TimeDrivenManager {
    /// Creates a new TimeDrivenManager managing time-driven output streams.
    fn new(ir: &LolaIR) -> TimeDrivenManager {
        unimplemented!()
    }

    /// Determines how long the current thread can wait until the next time-based evaluation cycle
    /// needs to be started. Calls are time-sensitive, i.e. successive calls do not necessarily
    /// yield identical results.
    ///
    /// *Returns:* `WaitingTime` _t_ and `TimeDrivenCycleCount` _i_ where _i_ nanoseconds can pass
    /// until the _i_th time-driven evaluation cycle needs to be started.
    fn wait_for(&self) -> (Duration, TimeDrivenCycleCount) {
        unimplemented!()
    }

    /// Returns all time-driven streams that are due to be extended in time-driven evaluation
    /// cycle `c`. The returned collection is ordered according to the evaluation order.
    fn due_streams(&mut self, c: TimeDrivenCycleCount) -> &[StreamReference] {
        unimplemented!()
    }
}

struct EventDrivenManager {

}

impl EventDrivenManager {
    /// Creates a new EventDrivenManager managing event-driven output streams.
    fn new(ir: &LolaIR) -> EventDrivenManager {
        unimplemented!()
    }

    /// Returns all event-driven streams that are due to be extended in time-driven evaluation
    /// cycle `c`. The returned collection is ordered according to the evaluation order.
    fn next_evaluation_layer(&mut self) -> &[StreamReference] {
        unimplemented!()
    }
}

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
    spec: LolaIR
}

impl lola_parser::LolaBackend for Evaluator {
    fn supported_feature_flags() -> Vec<lola_parser::FeatureFlag> {
        unimplemented!()
    }
}

impl Evaluator {

    /// Create a new `Evaluator` for RTLola specifications. Respects settings passed in `config`.
    pub fn new(ir: LolaIR, config: EvalConfig) -> Evaluator {
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
    pub fn start() {
        unimplemented!()
    }


    //////////// Private Functions ////////////

    fn find_extend_period(ir: &LolaIR) -> Duration {
        assert!(!ir.time_outputs.is_empty());
        let mut rates: Vec<u128> = ir.time_outputs
            .iter()
            .flat_map(|s| s.extend_rate.map(|r| r.0))
            .collect();
        let res = rates.iter().fold(rates[0], |a,b| Evaluator::gcd(a, *b));
        Duration(res)
    }

    fn gcd(mut a: u128, mut b: u128) -> u128 {
        // Courtesy of wikipedia.
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

}


mod tests {
    use evaluator::Evaluator;
    use lola_parser::{LolaIR, Duration};

    fn to_ir(spec: &str) -> LolaIR {
        unimplemented!("We need some interface from the parser.")
    }

    #[test]
    fn test_gcd() {
        assert_eq!(Evaluator::gcd(3, 18), 3);
        assert_eq!(Evaluator::gcd(18, 3), 3);
        assert_eq!(Evaluator::gcd(1, 25), 1);
        assert_eq!(Evaluator::gcd(5, 13), 1);
        assert_eq!(Evaluator::gcd(25, 40), 5);
        assert_eq!(Evaluator::gcd(7, 7), 7);
    }

    #[test]
    fn test_extension_rate_extraction() {
        let input = "input a: UInt8\n";
        let hz50 = "output b: UInt8 {extend @50Hz} := a";
        let hz40 = "output b: UInt8 {extend @40Hz} := a";
        let hz100 = "output b: UInt8 {extend @100Hz} := a";
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
