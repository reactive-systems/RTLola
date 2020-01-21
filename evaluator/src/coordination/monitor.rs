use crate::basics::{EvalConfig, OutputHandler, Time};
use crate::coordination::Event;
use crate::evaluator::{Evaluator, EvaluatorData};
use crate::storage::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use streamlab_common::schedule::{Deadline, Schedule};
use streamlab_frontend::ir::{InputReference, LolaIR, OutputReference};

pub type StateSlice = Vec<(OutputReference, Value)>;

pub struct Update {
    pub timed: Vec<(Time, StateSlice)>,
    pub event: StateSlice,
}

pub struct Monitor {
    ir: LolaIR, // probably not necessary to store here.
    eval: Evaluator,
    pub output_handler: Arc<OutputHandler>,
    deadlines: Vec<Deadline>,
    current_time: Duration,
}

// Crate-public interface
impl Monitor {
    pub(crate) fn setup(ir: LolaIR, output_handler: Arc<OutputHandler>, config: EvalConfig) -> Monitor {
        // Note: start_time only accessed in online mode.
        let eval_data = EvaluatorData::new(ir.clone(), config.clone(), output_handler.clone(), Instant::now());

        let deadlines: Vec<Deadline> = if ir.time_driven.is_empty() {
            vec![]
        } else {
            Schedule::from(&ir).expect("Creation of schedule failed.").deadlines
        };

        Monitor { ir, eval: eval_data.into_evaluator(), output_handler, deadlines, current_time: Time::default() }
    }
}

// Public interface
impl Monitor {
    pub fn accept_event<E: Into<Event>>(&mut self, ev: E, ts: Time) -> Update {
        let ev = ev.into();
        self.output_handler.debug(|| format!("Accepted {:?}.", ev));

        let timed = self.accept_time(ts);

        // Evaluate
        self.output_handler.new_event();
        self.eval.eval_event(ev.as_slice(), ts);
        let event_change = self.eval.peek_fresh();

        self.current_time = ts;

        Update { timed, event: event_change }
    }

    pub fn accept_time(&mut self, ts: Time) -> Vec<(Time, StateSlice)> {
        let mut next_deadline = Duration::default();
        let mut timed_changes: Vec<(Time, StateSlice)> = vec![];

        if !self.deadlines.is_empty() {
            return timed_changes;
        }
        assert!(self.deadlines.len() > 0);
        let mut due_ix = self.deadlines.len() - 1;

        while ts > next_deadline {
            // Go back in time and evaluate,...
            let dl = &self.deadlines[due_ix];
            self.output_handler.debug(|| format!("Schedule Timed-Event {:?}.", (&dl.due, next_deadline)));
            self.output_handler.new_event();
            self.eval.eval_time_driven_outputs(&dl.due, ts);
            due_ix = (due_ix + 1) % self.deadlines.len();
            let dl = &self.deadlines[due_ix];
            timed_changes.push((next_deadline, self.eval.peek_fresh()));
            assert!(dl.pause > Duration::from_secs(0));
            next_deadline += dl.pause;
        }
        timed_changes
    }

    pub fn name_for_input(&self, id: OutputReference) -> &str {
        self.ir.inputs[id].name.as_str()
    }

    pub fn name_for_output(&self, id: InputReference) -> &str {
        self.ir.outputs[id].name.as_str()
    }
}
