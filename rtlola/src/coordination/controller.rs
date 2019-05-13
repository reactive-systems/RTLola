use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use std::time::SystemTime;
use streamlab_frontend::ir::LolaIR;

use crate::basics::EvalConfig;
use crate::basics::OutputHandler;
use crate::evaluator::{Evaluator, EvaluatorData};

use super::event_driven_manager::{EventDrivenManager, EventEvaluation};
use super::time_driven_manager::{TimeDrivenManager, TimeEvaluation};

pub struct Controller<'e, 'c> {
    /// Handles all kind of output behavior according to config.
    output_handler: OutputHandler,

    /// Handles evaluating stream expressions and storage of values.
    evaluator: Evaluator<'e, 'c>,
}

impl<'e, 'c> streamlab_frontend::LolaBackend for Controller<'e, 'c> {
    fn supported_feature_flags() -> Vec<streamlab_frontend::ir::FeatureFlag> {
        unimplemented!()
    }
}

impl<'e, 'c> Controller<'e, 'c> {
    /// Starts the online evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    pub fn evaluate_online(ir: LolaIR, config: EvalConfig) -> ! {
        let (work_tx, work_rx) = mpsc::channel();

        let has_time_driven = !ir.time_driven.is_empty();
        if has_time_driven {
            let work_tx_clone = work_tx.clone();
            let ir_clone = ir.clone();
            let cfg_clone = config.clone();
            let _ = thread::Builder::new().name("TimeDrivenManager".into()).spawn(move || {
                let time_manager = TimeDrivenManager::setup(ir_clone, cfg_clone, SystemTime::now());
                time_manager.start_online(work_tx_clone);
            });
        };

        let ir_clone = ir.clone();
        let cfg_clone = config.clone();
        // TODO: Wait until all events have been read.
        let _event = thread::Builder::new().name("EventDrivenManager".into()).spawn(move || {
            let event_manager = EventDrivenManager::setup(ir_clone, cfg_clone);
            event_manager.start_online(work_tx);
        });

        let mut evaluatordata = EvaluatorData::new(ir, SystemTime::now(), config.clone());

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

    /// Starts the offline evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    pub fn evaluate_offline(ir: LolaIR, config: EvalConfig) -> ! {
        // Use a bounded channel for offline mode, as we "control" time.
        let (work_tx, work_rx) = mpsc::sync_channel(1024);

        let ir_clone = ir.clone();
        let cfg_clone = config.clone();
        let _event = thread::Builder::new().name("EventDrivenManager".into()).spawn(move || {
            let event_manager = EventDrivenManager::setup(ir_clone, cfg_clone);
            event_manager.start_offline(work_tx);
        });

        let start_time = match work_rx.recv() {
            Err(e) => panic!("Both producers hung up! {}", e),
            Ok(item) => match item {
                WorkItem::Start(ts) => ts,
                _ => panic!("Did not receive a start event in offline mode!"),
            },
        };

        let has_time_driven = !ir.time_driven.is_empty();
        let ir_clone = ir.clone();
        let cfg_clone = config.clone();
        let time_manager = TimeDrivenManager::setup(ir_clone, cfg_clone, start_time);
        let (wait_time, mut due_streams) =
            if has_time_driven { time_manager.get_current_deadline(start_time) } else { (Duration::default(), vec![]) };
        let mut next_deadline = start_time + wait_time;

        let mut evaluatordata = EvaluatorData::new(ir, start_time, config.clone());

        let output_handler = OutputHandler::new(&config);
        let evaluator = evaluatordata.as_Evaluator();
        let mut ctrl = Controller { output_handler, evaluator };

        loop {
            let item = work_rx.recv().unwrap_or_else(|e| panic!("Both producers hung up! {}", e));
            ctrl.output_handler.debug(|| format!("Received {:?}.", item));
            match item {
                WorkItem::Event(e, ts) => {
                    if has_time_driven {
                        while ts >= next_deadline {
                            // Go back in time, evaluate,...
                            ctrl.evaluate_timed_item(&due_streams, next_deadline);
                            let (wait_time, due) = time_manager.get_current_deadline(next_deadline);
                            next_deadline += wait_time;
                            due_streams = due;
                        }
                    }
                    ctrl.evaluate_event_item(&e, ts)
                }
                WorkItem::Time(_, _) => panic!("Received time command in offline mode."),
                WorkItem::Start(_) => panic!("Received spurious start command."),
                WorkItem::End => {
                    ctrl.output_handler.output(|| "Finished entire input. Terminating.");
                    std::process::exit(0);
                }
            }
        }
    }

    fn eval_workitem(&mut self, wi: WorkItem) {
        self.output_handler.debug(|| format!("Received {:?}.", wi));
        match wi {
            WorkItem::Event(e, ts) => self.evaluate_event_item(&e, ts),
            WorkItem::Time(t, ts) => self.evaluate_timed_item(&t, ts),
            WorkItem::Start(_) => panic!("Received spurious start command."),
            WorkItem::End => {
                self.output_handler.output(|| "Finished entire input. Terminating.");
                std::process::exit(0);
            }
        }
    }

    fn evaluate_timed_item(&mut self, t: &TimeEvaluation, ts: SystemTime) {
        self.evaluator.eval_some_outputs(t, ts);
    }

    fn evaluate_event_item(&mut self, e: &EventEvaluation, ts: SystemTime) {
        self.evaluator.accept_inputs(e, ts);
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
