use super::event_driven_manager::EventDrivenManager;
use super::time_driven_manager::TimeDrivenManager;
use super::WorkItem;
use crate::basics::{EvalConfig, OutputHandler};
use crate::evaluator::EvaluatorData;
use crossbeam_channel::{bounded, unbounded};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime};
use streamlab_frontend::ir::LolaIR;

pub struct Controller {
    ir: LolaIR,

    config: EvalConfig,

    /// Handles all kind of output behavior according to config.
    output_handler: Arc<OutputHandler>,
}

impl streamlab_frontend::LolaBackend for Controller {
    fn supported_feature_flags() -> Vec<streamlab_frontend::ir::FeatureFlag> {
        unimplemented!()
    }
}

impl Controller {
    pub(crate) fn new(ir: LolaIR, config: EvalConfig) -> Self {
        let output_handler = Arc::new(OutputHandler::new(&config));
        Self { ir, config, output_handler }
    }

    pub(crate) fn start(&self) -> ! {
        if self.config.offline {
            self.evaluate_offline();
        } else {
            self.evaluate_online();
        }
    }

    /// Starts the online evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    fn evaluate_online(&self) -> ! {
        let (work_tx, work_rx) = unbounded();

        let copy_output_handler = self.output_handler.clone();

        let has_time_driven = !self.ir.time_driven.is_empty();
        if has_time_driven {
            let work_tx_clone = work_tx.clone();
            let ir_clone = self.ir.clone();
            let cfg_clone = self.config.clone();
            let _ = thread::Builder::new().name("TimeDrivenManager".into()).spawn(move || {
                let time_manager =
                    TimeDrivenManager::setup(ir_clone, cfg_clone, SystemTime::now(), copy_output_handler);
                time_manager.start_online(work_tx_clone);
            });
        };

        let copy_output_handler = self.output_handler.clone();

        let ir_clone = self.ir.clone();
        let cfg_clone = self.config.clone();
        // TODO: Wait until all events have been read.
        let _event = thread::Builder::new().name("EventDrivenManager".into()).spawn(move || {
            let event_manager = EventDrivenManager::setup(ir_clone, cfg_clone, copy_output_handler);
            event_manager.start_online(work_tx);
        });

        let copy_output_handler = self.output_handler.clone();
        let mut evaluatordata =
            EvaluatorData::new(self.ir.clone(), SystemTime::now(), self.config.clone(), copy_output_handler);

        let mut evaluator = evaluatordata.as_Evaluator();

        loop {
            match work_rx.recv() {
                Ok(item) => evaluator.eval_workitem(item),
                Err(e) => panic!("Both producers hung up! {}", e),
            }
        }
    }

    /// Starts the offline evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    fn evaluate_offline(&self) -> ! {
        // Use a bounded channel for offline mode, as we "control" time.
        let (work_tx, work_rx) = bounded(1024);

        let output_copy_handler = self.output_handler.clone();

        let ir_clone = self.ir.clone();
        let cfg_clone = self.config.clone();
        let _event = thread::Builder::new().name("EventDrivenManager".into()).spawn(move || {
            let event_manager = EventDrivenManager::setup(ir_clone, cfg_clone, output_copy_handler);
            event_manager.start_offline(work_tx);
        });

        let start_time = match work_rx.recv() {
            Err(e) => panic!("Both producers hung up! {}", e),
            Ok(item) => match item {
                WorkItem::Start(ts) => ts,
                _ => panic!("Did not receive a start event in offline mode!"),
            },
        };

        let has_time_driven = !self.ir.time_driven.is_empty();
        let ir_clone = self.ir.clone();
        let cfg_clone = self.config.clone();
        let output_copy_handler = self.output_handler.clone();
        let time_manager = TimeDrivenManager::setup(ir_clone, cfg_clone, start_time, output_copy_handler);
        let (wait_time, mut due_streams) =
            if has_time_driven { time_manager.get_current_deadline(start_time) } else { (Duration::default(), vec![]) };
        let mut next_deadline = start_time + wait_time;

        let output_copy_handler = self.output_handler.clone();
        let mut evaluatordata =
            EvaluatorData::new(self.ir.clone(), start_time, self.config.clone(), output_copy_handler);

        let mut evaluator = evaluatordata.as_Evaluator();

        loop {
            let item = work_rx.recv().unwrap_or_else(|e| panic!("Both producers hung up! {}", e));
            self.output_handler.debug(|| format!("Received {:?}.", item));
            match item {
                WorkItem::Event(e, ts) => {
                    if has_time_driven {
                        while ts >= next_deadline {
                            // Go back in time, evaluate,...
                            evaluator.evaluate_timed_item(&due_streams, next_deadline);
                            let (wait_time, due) = time_manager.get_current_deadline(next_deadline);
                            next_deadline += wait_time;
                            due_streams = due;
                        }
                    }
                    evaluator.evaluate_event_item(&e, ts)
                }
                WorkItem::Time(_, _) => panic!("Received time command in offline mode."),
                WorkItem::Start(_) => panic!("Received spurious start command."),
                WorkItem::End => {
                    self.output_handler.output(|| "Finished entire input. Terminating.");
                    self.output_handler.terminate();
                    std::process::exit(0);
                }
            }
        }
    }
}
