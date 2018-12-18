use crate::evaluator::io_handler::{OutputHandler, OutputKind};
use crate::util;
use lola_parser::{LolaIR, OutputStream, Stream, StreamReference};
use std::cmp::Ordering;
use std::rc::Rc;
use std::time::{Duration, SystemTime};

const NANOS_PER_SEC: u128 = 1_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TDMState {
    cycle: TimeDrivenCycleCount,
    deadline: usize,
    time: SystemTime, // Debug/statistics information.
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StateCompare {
    Equal,
    TimeTravel,
    Next,
    Skipped { cycles: u64, deadlines: u64 },
}

/// Represents the current cycle count for time-driven events. `u128` is sufficient to represent
/// 10^22 years of runtime for evaluation cycles that are 1ns apart.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct TimeDrivenCycleCount(u128);

impl From<u128> for TimeDrivenCycleCount {
    fn from(i: u128) -> TimeDrivenCycleCount {
        TimeDrivenCycleCount(i)
    }
}

#[derive(Debug, Clone)]
struct Deadline {
    pause: Duration,
    due: Vec<StreamReference>,
}

pub struct TimeDrivenManager {
    last_state: Option<TDMState>,
    deadlines: Vec<Deadline>,
    hyper_period: Duration,
    start_time: Option<SystemTime>,
    handler: Rc<OutputHandler>,
}

impl TimeDrivenManager {
    /// Creates a new TimeDrivenManager managing time-driven output streams.
    pub fn new(ir: &LolaIR, handler: Rc<OutputHandler>) -> TimeDrivenManager {
        if ir.time_outputs.is_empty() {
            return TimeDrivenManager {
                last_state: None,
                deadlines: vec![Deadline { pause: Duration::from_secs(0), due: Vec::new() }],
                hyper_period: Duration::from_secs(100000), // Actual value does not matter.
                start_time: None,
                handler,
            };
        }

        fn extr_dur(s: &OutputStream) -> Duration {
            s.extend_rate.unwrap()
        }

        let streams = &ir.time_outputs;
        let rates: Vec<Duration> = streams.iter().map(extr_dur).collect();
        let max = rates.iter().max().unwrap();
        let gcd = Self::find_extend_period(&rates);
        let hyper_period = Self::find_hyper_period(&rates);

        let num_steps = TimeDrivenManager::divide_durations(hyper_period, gcd, true);

        let mut extend_steps = vec![Vec::new(); num_steps];
        for s in streams {
            let ix = Self::divide_durations(extr_dur(s), gcd, false) + 1;
            extend_steps[ix].push(s.as_stream_ref());
        }

        let mut init: (u32, Vec<Deadline>) = (0, Vec::new());
        let (remaining, mut deadlines) = extend_steps.iter().fold(init, |(empty_counter, mut deadlines), step| {
            if step.is_empty() {
                (empty_counter + 1, deadlines)
            } else {
                let pause = (empty_counter + 1) * gcd;
                let deadline = Deadline { pause, due: step.clone() };
                deadlines.push(deadline);
                (0, deadlines)
            }
        });
        if remaining != 0 {
            // There is some gcd periods left at the end of the hyper period.
            // We cannot add them to the first because this would off-set the very first iteration.
            deadlines.push(Deadline { pause: remaining * gcd, due: Vec::new() });
        }

        // TODO: Sort by evaluation order!

        TimeDrivenManager { last_state: None, deadlines, hyper_period, start_time: None, handler }
    }

    pub fn start(&mut self, time: Option<SystemTime>) {
        self.start_time = time.or(Some(SystemTime::now()))
    }

    /// Determines how long the current thread can wait until the next time-based evaluation cycle
    /// needs to be started. Calls are time-sensitive, i.e. successive calls do not necessarily
    /// yield identical results.
    ///
    /// *Returns:* `WaitingTime` _t_ and `TimeDrivenCycleCount` _i_ where _i_ nanoseconds can pass
    /// until the _i_th time-driven evaluation cycle needs to be started.
    pub fn wait_for(&self, time: Option<SystemTime>) -> (Duration, TimeDrivenCycleCount) {
        let time = time.unwrap_or_else(SystemTime::now);
        let current_state = self.current_state(time);
        if let Some(last_state) = self.last_state {
            match self.compare_states(last_state, current_state) {
                StateCompare::TimeTravel => panic!("Bug: Traveled back in time!"),
                StateCompare::Skipped { cycles, deadlines } => {
                    self.skipped_deadline(cycles, deadlines);
                    // Carry on.
                }
                StateCompare::Equal => {
                    self.query_too_soon();
                    // Carry on.
                }
                StateCompare::Next => {} // Nice! Do nothing.
            }
        }

        let deadline = &self.deadlines[current_state.deadline];
        let offset = self.time_since_last_deadline(time);
        assert!(offset < deadline.pause);
        (deadline.pause - offset, current_state.cycle)
    }

    /// Returns all time-driven streams that are due to be extended in time-driven evaluation
    /// cycle `c`. The returned collection is ordered according to the evaluation order.
    pub fn due_streams(&mut self, time: Option<SystemTime>) -> Option<&Vec<StreamReference>> {
        let time = time.unwrap_or_else(SystemTime::now);
        let state = self.current_state(time);
        if let Some(old_state) = self.last_state {
            match self.compare_states(old_state, state) {
                StateCompare::Next => {} // Perfect, skip!
                StateCompare::TimeTravel => panic!("Bug: Traveled back in time!"),
                StateCompare::Equal => {
                    self.query_too_soon();
                    return None;
                }
                StateCompare::Skipped { cycles, deadlines } => self.skipped_deadline(cycles, deadlines),
            }
        }
        self.last_state = Some(state);
        Some(&self.deadlines[state.deadline].due)
    }

    fn skipped_deadline(&self, cycles: u64, deadlines: u64) {
        if cfg!(debug_assertion) {
            // Only panic in non-release config.
            panic!("Missed {} cycles and {} deadlines.", cycles, deadlines);
        } else {
            // Otherwise, inform the output handler and carry on.
            self.handler.emit(OutputKind::RuntimeWarning, || {
                format!("Warning: Pressure exceeds capacity! missed {} cycles and {} deadlines", cycles, deadlines)
            })
        }
    }

    fn query_too_soon(&self) {
        if cfg!(debug_assertion) {
            // Only panic in non-release config.
            panic!("Called `TimeDrivenManager::wait_for` too early; no deadline has passed.");
        } else {
            // Otherwise, inform the output handler and carry on.
            self.handler.emit(OutputKind::Debug, || {
                String::from("Warning: Called `TimeDrivenManager::wait_for` twice for the same deadline.")
            })
        }
    }

    fn time_since_last_deadline(&self, time: SystemTime) -> Duration {
        let last_deadline = self.last_deadline_state(time);
        time.duration_since(last_deadline.time).expect("Last deadline cannot lie in the past.")
    }

    /// Compares two given states in terms of their temporal relation.
    /// Detects equality, successivity, contradiction, i.e. the `new` state is older than the `old`
    /// one, and the amount of missed deadlines in-between `old` and `new`.
    fn compare_states(&self, old: TDMState, new: TDMState) -> StateCompare {
        let c1 = old.cycle.0;
        let c2 = new.cycle.0;
        let d1 = old.deadline;
        let d2 = new.deadline;
        match c1.cmp(&c2) {
            Ordering::Greater => StateCompare::TimeTravel,
            Ordering::Equal => {
                match d1.cmp(&d2) {
                    Ordering::Greater => StateCompare::TimeTravel,
                    Ordering::Equal => StateCompare::Equal,
                    Ordering::Less => {
                        let diff = (d2 - d1) as u64; // Safe: d2 must be greater than d1.
                        if diff == 1 {
                            StateCompare::Next
                        } else {
                            // diff >= 2
                            StateCompare::Skipped { cycles: 0, deadlines: diff - 1 }
                        }
                    }
                }
            }
            Ordering::Less => {
                let diff = (c2 - c1) as u64; // Safe: c2 must to be greater than c1.
                if diff == 1 && d1 == self.deadlines.len() - 1 && d2 == 0 {
                    StateCompare::Next
                } else {
                    let d_diff = (d2 as i128) - (d1 as i128); // Widen to assure safe arithmetic.
                    if d_diff > 0 {
                        StateCompare::Skipped { cycles: diff, deadlines: (d_diff - 1) as u64 }
                    } else if d_diff == 0 {
                        // d_diff <= 0
                        StateCompare::Skipped { cycles: diff - 1, deadlines: self.deadlines.len() as u64 }
                    } else {
                        // d_dif < 0
                        let abs = -d_diff as u64;
                        let actual_d_diff = self.deadlines.len() as u64 - abs;
                        StateCompare::Skipped { cycles: diff - 1, deadlines: actual_d_diff - 1 }
                    }
                }
            }
        }
    }

    /// Computes the last deadline that should have been evaluated according to the given `time`.
    /// The state's time is the earliest point in time where this state could have been the current
    /// one.
    /// Requires start_time to be non-none.
    fn last_deadline_state(&self, time: SystemTime) -> TDMState {
        let start_time = self.start_time.unwrap();
        assert!(start_time < time);
        let hyper_nanos = Self::dur_as_nanos(self.hyper_period);
        let running = Self::dur_as_nanos(time.duration_since(start_time).unwrap());
        let cycle = running / hyper_nanos;
        let in_cycle = running % hyper_nanos;
        let mut sum = 0u128;
        for (ix, dl) in self.deadlines.iter().enumerate() {
            let pause = Self::dur_as_nanos(dl.pause);
            if sum + pause < in_cycle {
                sum += pause
            } else {
                let offset_duration = Self::dur_from_nanos(hyper_nanos * cycle + sum);
                let dl_time = start_time + offset_duration;
                return TDMState { cycle: cycle.into(), deadline: ix, time: dl_time };
            }
        }
        unreachable!()
    }

    /// Computes the TDMState in which we should be right now given the supplied `time`.
    fn current_state(&self, time: SystemTime) -> TDMState {
        let state = self.last_deadline_state(time);
        TDMState { time, ..state }
    }

    /// Divides two durations. If `rhs` is not a divider of `lhs`, a warning is emitted and the
    /// rounding strategy `round_up` is applied.
    fn divide_durations(lhs: Duration, rhs: Duration, round_up: bool) -> usize {
        // The division of durations is currently unstable (feature duration_float) because
        // it falls back to using floats which cannot necessarily represent the durations
        // accurately. We, however, fall back to nanoseconds as u128. Regardless, some inaccuracies
        // might occur, rendering this code TODO *not stable for real-time devices!*
        let lhs = Self::dur_as_nanos(lhs);
        let rhs = Self::dur_as_nanos(rhs);
        let representable = lhs % rhs == 0;
        let mut div = lhs / rhs;
        if !representable {
            println!("Warning: Spec unstable: Cannot accurately represent extend periods.");
            // TODO: Introduce better mechanism for emitting such warnings.
            if round_up {
                div += 1;
            }
        }
        div as usize
    }

    /// Determines the hyper period of the given `rates`.
    fn find_hyper_period(rates: &[Duration]) -> Duration {
        assert!(!rates.is_empty());
        let rates: Vec<u128> = rates.iter().map(|r| Self::dur_as_nanos(*r)).collect();
        let lcm = util::lcm_all(&rates);
        Self::dur_from_nanos(lcm)
    }

    /// Determines the max amount of time the process can wait between successive checks for
    /// due deadlines without missing one.
    fn find_extend_period(rates: &[Duration]) -> Duration {
        assert!(!rates.is_empty());
        let rates: Vec<u128> = rates.iter().map(|r| Self::dur_as_nanos(*r)).collect();
        let gcd = util::gcd_all(&rates);
        Self::dur_from_nanos(gcd)
    }

    fn dur_as_nanos(dur: Duration) -> u128 {
        dur.as_secs() as u128 * NANOS_PER_SEC as u128 + dur.subsec_nanos() as u128
    }

    fn dur_from_nanos(dur: u128) -> Duration {
        // TODO: Introduce sanity checks for `dur` s.t. cast is safe.
        let secs = (dur / NANOS_PER_SEC) as u64; // safe cast for realistic values of `dur`.
        let nanos = (dur % NANOS_PER_SEC) as u32; // safe case
        Duration::new(secs, nanos)
    }
}

mod tests {
    use crate::evaluator::{io_handler::OutputHandler, time_driven_manager::*};
    use lola_parser::LolaIR;
    use std::rc::Rc;
    use std::time::{Duration, SystemTime};

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
            let rates: Vec<std::time::Duration> =
                to_ir(spec).time_outputs.iter().map(|s| s.extend_rate.unwrap()).collect();
            let was = TimeDrivenManager::find_extend_period(&rates);
            let was = TimeDrivenManager::dur_as_nanos(was);
            assert_eq!(*expected, was);
        }
    }

    #[test]
    fn test_divide_durations_round_down() {
        type TestDurations = ((u64, u32), (u64, u32), usize);
        let case1: TestDurations = ((1, 0), (1, 0), 1);
        let case2: TestDurations = ((1, 0), (0, 100_000_000), 10);
        let case2: TestDurations = ((1, 0), (0, 100_000), 10_000);
        let case3: TestDurations = ((1, 0), (0, 20_000), 50_000);
        let case4: TestDurations = ((0, 40_000), (0, 30_000), 1);
        let case4: TestDurations = ((3, 1_000), (3, 5_000), 0);

        let cases = [case1, case2, case3, case4];
        for (a, b, expected) in &cases {
            let to_dur = |(s, n)| Duration::new(s, n);
            let was = TimeDrivenManager::divide_durations(to_dur(*a), to_dur(*b), false);
            assert_eq!(was, *expected, "Expected {}, but was {}.", expected, was);
        }
    }
    #[test]
    fn test_compare_states() {
        let handler = Rc::new(OutputHandler::default());
        fn dl(s: u64, n: u32) -> Deadline {
            Deadline { pause: Duration::new(s, n), due: Vec::new() }
        }
        fn dur(s: u64, n: u32) -> Duration {
            Duration::new(s, n)
        }
        // 1s, 0.1s, 2.3s, 1.6s
        let deadlines = vec![dl(1, 0), dl(0, 100_000_000), dl(2, 300_000_000), dl(1, 600_000_000)];
        let tdm = TimeDrivenManager { last_state: None, deadlines, hyper_period: dur(1, 0), start_time: None, handler };
        fn state(cy: u128, dl: usize) -> TDMState {
            let time = SystemTime::now();
            TDMState { cycle: cy.into(), deadline: dl, time }
        }
        type Case = (TDMState, TDMState, StateCompare);
        let cases: Vec<Case> = vec![
            (state(0, 0), state(0, 0), StateCompare::Equal),
            (state(1, 0), state(0, 0), StateCompare::TimeTravel),
            (state(1, 2), state(2, 1), StateCompare::Skipped { cycles: 0, deadlines: 2 }),
            (state(1, 2), state(3, 1), StateCompare::Skipped { cycles: 1, deadlines: 2 }),
            (state(1, 2), state(3, 0), StateCompare::Skipped { cycles: 1, deadlines: 1 }),
            (state(0, 0), state(0, 1), StateCompare::Next),
            (state(4, 3), state(5, 0), StateCompare::Next),
        ];
        for (old, new, expected) in &cases {
            let was = tdm.compare_states(*old, *new);
            assert_eq!(
                was, *expected,
                "Expected {:?}, but was {:?} for old: {:?} and new: {:?}.",
                expected, was, old, new
            )
        }
    }

    #[test]
    fn test_divide_durations_round_up() {
        type TestDurations = ((u64, u32), (u64, u32), usize);
        let case1: TestDurations = ((1, 0), (1, 0), 1);
        let case2: TestDurations = ((1, 0), (0, 100_000_000), 10);
        let case2: TestDurations = ((1, 0), (0, 100_000), 10_000);
        let case3: TestDurations = ((1, 0), (0, 20_000), 50_000);
        let case4: TestDurations = ((0, 40_000), (0, 30_000), 2);
        let case4: TestDurations = ((3, 1_000), (3, 5_000), 1);

        let cases = [case1, case2, case3, case4];
        for (a, b, expected) in &cases {
            let to_dur = |(s, n)| Duration::new(s, n);
            let was = TimeDrivenManager::divide_durations(to_dur(*a), to_dur(*b), true);
            assert_eq!(was, *expected, "Expected {}, but was {}.", expected, was);
        }
    }

}
