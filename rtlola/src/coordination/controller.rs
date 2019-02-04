use lola_parser::*;
use std::fmt;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use crate::basics::EvalConfig;
use crate::basics::OutputHandler;
use crate::evaluator::Evaluator;
use crate::storage::Value;

use super::event_driven_manager::{EventDrivenManager, EventEvaluation};
use super::time_driven_manager::{TimeDrivenManager, TimeEvaluation};

pub struct Controller {
    /// Handles all kind of output behavior according to config.
    output_handler: OutputHandler,

    /// Intermediate representation of input Lola specification.
    spec: LolaIR,

    evaluator: Evaluator,
}

impl lola_parser::LolaBackend for Controller {
    fn supported_feature_flags() -> Vec<lola_parser::FeatureFlag> {
        unimplemented!()
    }
}

impl Controller {
    /// Starts the evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    pub fn evaluate(ir: LolaIR, config: EvalConfig, ts: Option<Instant>) -> ! {
        let (work_tx, work_rx) = mpsc::channel();
        let (eof_tx, eof_rx) = mpsc::channel();

        let start_time = std::time::SystemTime::now();

        let ir_clone_1 = ir.clone();
        let ir_clone_2 = ir.clone();
        let cfg_clone_1 = config.clone();
        let cfg_clone_2 = config.clone();
        let work_tx_clone = work_tx.clone();

        let event = thread::spawn(move || {
            let event_manager = EventDrivenManager::setup(ir_clone_1, cfg_clone_1);
            event_manager.start(work_tx_clone)
        });
        let time = thread::spawn(move || {
            let time_manager = TimeDrivenManager::setup(ir_clone_2, cfg_clone_2);
            time_manager.start(Some(start_time), work_tx, eof_rx)
        });

        let e = Evaluator::new(&ir, ts.unwrap_or_else(Instant::now));
        let mut ctrl = Controller { output_handler: OutputHandler::new(&config), spec: ir, evaluator: e };

        let _ = work_rx.iter().map(|wi| ctrl.eval_workitem(wi));

        panic!("Both producers hung up!");
    }

    fn eval_workitem(&mut self, wi: WorkItem) {
        match wi {
            WorkItem::Event(e) => self.evaluate_event_item(e),
            WorkItem::Time(t) => self.evaluate_timed_item(t),
        }
    }

    fn evaluate_timed_item(&mut self, t: TimeEvaluation) {
        t.into_iter().for_each(|s| self.evaluate_single(s));
    }

    fn evaluate_event_item(&mut self, ee: EventEvaluation) {
        self.evaluate_event(ee.event);
        ee.layers.into_iter().for_each(|layer| self.evaluate_all(layer));
    }

    fn evaluate_event(&mut self, event: Vec<(StreamReference, Value)>) {
        unimplemented!()
    }

    fn evaluate_all(&mut self, streams: Vec<StreamReference>) {
        streams.into_iter().for_each(|s| self.evaluate_single(s))
    }

    fn evaluate_single(&mut self, stream: StreamReference) {
        unimplemented!()
    }
}

#[derive(Debug)]
pub(crate) enum EvaluationError {
    UnknownError,
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "")
    }
}

impl std::error::Error for EvaluationError {}

pub(crate) enum WorkItem {
    Event(EventEvaluation),
    Time(TimeEvaluation),
}
