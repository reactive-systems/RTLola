use lola_parser::{Duration, LolaIR, StreamReference};

pub struct TimeDrivenManager {}

/// Represents the current cycle count for time-driven events. `u128` is sufficient to represent
/// 10^22 years of runtime for evaluation cycles that are 1ns apart.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct TimeDrivenCycleCount(u128);

impl TimeDrivenManager {
    /// Creates a new TimeDrivenManager managing time-driven output streams.
    pub fn new(ir: &LolaIR) -> TimeDrivenManager {
        unimplemented!()
    }

    /// Determines how long the current thread can wait until the next time-based evaluation cycle
    /// needs to be started. Calls are time-sensitive, i.e. successive calls do not necessarily
    /// yield identical results.
    ///
    /// *Returns:* `WaitingTime` _t_ and `TimeDrivenCycleCount` _i_ where _i_ nanoseconds can pass
    /// until the _i_th time-driven evaluation cycle needs to be started.
    pub fn wait_for(&self) -> (Duration, TimeDrivenCycleCount) {
        unimplemented!()
    }

    /// Returns all time-driven streams that are due to be extended in time-driven evaluation
    /// cycle `c`. The returned collection is ordered according to the evaluation order.
    pub fn due_streams(&mut self, c: TimeDrivenCycleCount) -> &[StreamReference] {
        unimplemented!()
    }
}
