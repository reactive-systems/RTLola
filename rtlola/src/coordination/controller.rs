use std::sync::mpsc;
use std::thread;
use std::time::SystemTime;
use streamlab_frontend::ir::{LolaIR, StreamReference};

use crate::basics::EvalConfig;
use crate::basics::OutputHandler;
use crate::evaluator::{Evaluator, EvaluatorData};

use super::event_driven_manager::{EventDrivenManager, EventEvaluation};
use super::time_driven_manager::{TimeDrivenManager, TimeEvaluation};

pub struct Controller<'a> {
    /// Handles all kind of output behavior according to config.
    output_handler: OutputHandler,

    /// Handles evaluating stream expressions and storage of values.
    evaluator: Evaluator<'a>,
}

impl<'a> streamlab_frontend::LolaBackend for Controller<'a> {
    fn supported_feature_flags() -> Vec<streamlab_frontend::ir::FeatureFlag> {
        unimplemented!()
    }
}

impl<'a> Controller<'a> {
    /// Starts the evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    pub fn evaluate(ir: LolaIR, config: EvalConfig, offline: bool) -> ! {
        let (work_tx, work_rx) = mpsc::channel();
        let (time_tx, time_rx) = mpsc::channel();
        let (ack_tx, ack_rx) = mpsc::channel();

        let has_time_driven = !ir.time_driven.is_empty();
        if has_time_driven {
            let work_tx_clone = work_tx.clone();
            let ir_clone = ir.clone();
            let cfg_clone = config.clone();
            let _ = thread::Builder::new().name("TimeDrivenManager".into()).spawn(move || {
                let time_manager = TimeDrivenManager::setup(ir_clone, cfg_clone);
                if offline {
                    time_manager.start_offline(work_tx_clone, time_rx, ack_tx);
                } else {
                    time_manager.start_online(Some(SystemTime::now()), work_tx_clone);
                }
            });
        };

        let ir_clone = ir.clone();
        let cfg_clone = config.clone();
        // TODO: Wait until all events have been read.
        let _event = thread::Builder::new().name("EventDrivenManager".into()).spawn(move || {
            let event_manager = EventDrivenManager::setup(ir_clone, cfg_clone);
            event_manager.start(offline, work_tx, has_time_driven, time_tx, ack_rx);
        });

        let mut evaluatordata = if offline {
            match work_rx.recv() {
                Err(e) => panic!("Both producers hung up! {}", e),
                Ok(wi) => match wi {
                    WorkItem::Start(ts) => EvaluatorData::new(ir, ts, config.clone()),
                    _ => panic!("Did not receive a start event in offline mode!"),
                },
            }
        } else {
            EvaluatorData::new(ir, SystemTime::now(), config.clone())
        };

        let output_handler = OutputHandler::new(&config);
        let evaluator = evaluatordata.as_Evaluator();
        let mut ctrl = Controller { output_handler, evaluator };

        loop {
            match work_rx.recv() {
                Ok(item) => ctrl.eval_workitem(item),
                Err(e) => panic!("Both producers hung up! {}", e),
            }
        }
    }

    fn eval_workitem(&mut self, wi: WorkItem) {
        self.output_handler.debug(|| format!("Received {:?}.", wi));
        match wi {
            WorkItem::Event(e, ts) => self.evaluate_event_item(e, ts),
            WorkItem::Time(t, ts) => self.evaluate_timed_item(t, ts),
            WorkItem::Start(_) => panic!("Received spurious start command."),
            WorkItem::End => {
                self.output_handler.output(|| "Finished entire input. Terminating.");
                std::process::exit(0);
            }
        }
    }

    fn evaluate_timed_item(&mut self, t: TimeEvaluation, ts: SystemTime) {
        self.evaluator.eval_outputs(&t, ts);
    }

    fn evaluate_event_item(&mut self, e: EventEvaluation, ts: SystemTime) {
        self.evaluator.accept_inputs(&e, ts);
        self.evaluator.eval_all_outputs(ts);
    }
}

#[derive(Debug)]
pub(crate) enum WorkItem {
    Event(EventEvaluation, SystemTime),
    Time(TimeEvaluation, SystemTime),
    Start(SystemTime),
    End,
}
