use super::WorkItem;
use crate::basics::{OutputHandler, Time};

use crossbeam_channel::Sender;
use rtlola_frontend::ir::{OutputReference, RTLolaIR};
use spin_sleep::SpinSleeper;
use std::sync::Arc;
use std::time::Instant;

use rtlola_frontend::ir::Deadline;

pub(crate) type TimeEvaluation = Vec<OutputReference>;

pub(crate) struct TimeDrivenManager {
    deadlines: Vec<Deadline>,
    handler: Arc<OutputHandler>,
}

impl TimeDrivenManager {
    /// Creates a new TimeDrivenManager managing time-driven output streams.
    pub(crate) fn setup(ir: RTLolaIR, handler: Arc<OutputHandler>) -> Result<TimeDrivenManager, String> {
        if ir.time_driven.is_empty() {
            // return dummy
            return Ok(TimeDrivenManager { deadlines: vec![], handler });
        }

        let schedule = ir.compute_schedule()?;
        // TODO: Sort by evaluation order!

        Ok(TimeDrivenManager { deadlines: schedule.deadlines, handler })
    }

    pub(crate) fn get_last_due(&self) -> &Vec<OutputReference> {
        assert!(!self.deadlines.is_empty());
        &self.deadlines.last().unwrap().due
    }

    pub(crate) fn get_deadline_cycle(&self) -> impl Iterator<Item = &Deadline> {
        self.deadlines.iter().cycle()
    }

    pub(crate) fn start_online(self, start_time: Instant, work_chan: Sender<WorkItem>) -> ! {
        assert!(!self.deadlines.is_empty());
        // timed streams at time 0
        let item = WorkItem::Time(self.get_last_due().clone(), Time::default());
        if work_chan.send(item).is_err() {
            self.handler.runtime_warning(|| "TDM: Sending failed; evaluation cycle lost.");
        }
        let deadline_cycle = self.get_deadline_cycle();
        let mut due_time = Time::default();
        for deadline in deadline_cycle {
            due_time += deadline.pause;

            let now = Instant::now();
            assert!(now >= start_time, "Time does not behave monotonically!");
            let time = now - start_time;

            if time < due_time {
                let wait_time = due_time - time;
                SpinSleeper::new(1_000_000).sleep(wait_time);
            }

            let item = WorkItem::Time(deadline.due.clone(), due_time);
            if work_chan.send(item).is_err() {
                self.handler.runtime_warning(|| "TDM: Sending failed; evaluation cycle lost.");
            }
        }
        unreachable!("should loop indefinitely")
    }

    //The following code is useful and could partly be used again for robustness.

    /*

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TDMState {
        cycle: TimeDrivenCycleCount,
        deadline: usize,
        time: Time,
        // Debug/statistics information.
    }

    #[allow(dead_code)]
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
    pub(crate) struct TimeDrivenCycleCount(u128);

    impl From<u128> for TimeDrivenCycleCount {
        fn from(i: u128) -> TimeDrivenCycleCount {
            TimeDrivenCycleCount(i)
        }
    }


        pub(crate) fn get_current_deadline(&self, time: Time) -> (Duration, Vec<OutputReference>) {
            let earliest_deadline_state = self.earliest_deadline_state(time);
            let current_deadline = &self.deadlines[earliest_deadline_state.deadline];
            let time_of_deadline = earliest_deadline_state.time + current_deadline.pause;
            assert!(time_of_deadline >= time, "Time does not behave monotonically!");
            let pause = time_of_deadline - time;
            (pause, current_deadline.due.clone())
        }

        /// Compute the state for the deadline earliest deadline that is not missed yet.
        /// Example: If this function is called immediately after setup, it returns
        /// a state containing the start time and deadline 0.
        fn earliest_deadline_state(&self, time: Time) -> TDMState {
            let start_time = self.start_time;
            let hyper_nanos = dur_as_nanos(self.hyper_period);
            assert!(time >= start_time, "Time does not behave monotonically!");
            let time_since_start = (time - start_time).as_nanos();

            let hyper = time_since_start / hyper_nanos;
            let time_within_hyper = time_since_start % hyper_nanos;

            // Determine the index of the current deadline.
            let mut sum = 0u128;
            for (ix, dl) in self.deadlines.iter().enumerate() {
                let pause = dur_as_nanos(dl.pause);
                if sum + pause < time_within_hyper {
                    sum += pause
                } else {
                    let offset_from_start = dur_from_nanos(hyper_nanos * hyper + sum);
                    let dl_time = start_time + offset_from_start;
                    return TDMState { cycle: hyper.into(), deadline: ix, time: dl_time };
                }
            }
            unreachable!()
        }


        /// Determines how long the current thread can wait until the next time-based evaluation cycle
        /// needs to be started. Calls are time-sensitive, i.e. successive calls do not necessarily
        /// yield identical results.
        ///
        /// *Returns:* `WaitingTime` _t_ and `TimeDrivenCycleCount` _i_ where _t_ nanoseconds can pass
        /// until the _i_th time-driven evaluation cycle needs to be started.
        fn wait_for(&self, time: Option<SystemTime>) -> (Duration, TimeDrivenCycleCount) {
            let time = time.unwrap_or_else(SystemTime::now);
            let current_state = self.current_state(time);
            if let Some(last_state) = self.last_state {
                match self.compare_states(last_state, current_state) {
                    StateCompare::TimeTravel => panic!("Bug: Traveled back in time!"),
                    StateCompare::Skipped { cycles, deadlines } => self.skipped_deadline(cycles, deadlines), // Carry on.
                    StateCompare::Next => self.skipped_deadline(0, 1), // Carry one.
                    StateCompare::Equal => {
                        // Nice, we did not miss a deadline!
                    }
                }
            }
            let deadline = &self.deadlines[current_state.deadline];
            let offset = self.time_since_last_deadline(time);
            assert!(offset < deadline.pause);
            (deadline.pause - offset, current_state.cycle)
        }

        /// Returns all time-driven streams that are due to be extended in time-driven evaluation
        /// cycle `c`. The returned collection is ordered according to the evaluation order.
        fn due_streams(&mut self, time: Option<SystemTime>) -> Option<&Vec<StreamReference>> {
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
                self.handler.runtime_warning(|| {
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
                self.handler
                    .debug(|| String::from("Warning: Called `TimeDrivenManager::wait_for` twice for the same deadline."))
            }
        }

        fn time_since_last_deadline(&self, time: SystemTime) -> Duration {
            let last_deadline = self.last_deadline_state(time);
            time.duration_since(last_deadline.time)
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
            let running = Self::dur_as_nanos(time.duration_since(start_time));
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

        */
}
