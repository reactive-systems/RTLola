pub(crate) mod config;
mod event_driven_manager;
mod io_handler;
mod time_driven_manager;
mod evaluation;
mod storage;

use std::thread;
use std::sync::mpsc;
use crate::evaluator::{
    config::*, event_driven_manager::EventDrivenManager, io_handler::*, time_driven_manager::TimeDrivenManager,
};
use lola_parser::*;
use std::rc::Rc;
use std::fmt;
use self::event_driven_manager::EventEvaluation;
use self::time_driven_manager::TimeEvaluation;

pub struct Evaluator {
    /// Handles all kind of output behavior according to config.
    output_handler: OutputHandler,

    /// Intermediate representation of input Lola specification.
    spec: LolaIR,
}

impl lola_parser::LolaBackend for Evaluator {
    fn supported_feature_flags() -> Vec<lola_parser::FeatureFlag> {
        unimplemented!()
    }
}

impl Evaluator {

    /// Starts the evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    pub fn evaluate(ir: LolaIR, config: EvalConfig) -> ! {

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
        // Either it will never terminate, or it will run out of data.
        event.join().expect("Couldn't join event based thread.");
        // Terminate time as well.
        eof_tx.send(true).expect("Cannot stop time based thread.");

        std::process::exit(1)
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

    fn evaluate_event(&mut self, event: Vec<(StreamReference, String)>) {
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
    Time(TimeEvaluation)
}
