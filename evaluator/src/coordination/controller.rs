use super::event_driven_manager::EventDrivenManager;
use super::time_driven_manager::TimeDrivenManager;
use super::{WorkItem, CAP_WORK_QUEUE};
use crate::basics::{EvalConfig, ExecutionMode::*, OutputHandler};
use crate::coordination::{EventEvaluation, TimeEvaluation};
use crate::evaluator::{Evaluator, EvaluatorData};
use crossbeam_channel::{bounded, unbounded};
use std::error::Error;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime};
use streamlab_frontend::ir::LolaIR;

pub struct Controller {
    ir: LolaIR,

    config: EvalConfig,

    /// Handles all kind of output behavior according to config.
    pub(crate) output_handler: Arc<OutputHandler>,
}

impl streamlab_frontend::LolaBackend for Controller {
    fn supported_feature_flags() -> Vec<streamlab_frontend::ir::FeatureFlag> {
        unimplemented!()
    }
}

impl Controller {
    pub(crate) fn new(ir: LolaIR, config: EvalConfig) -> Self {
        let output_handler = Arc::new(OutputHandler::new(&config, ir.triggers.len()));
        Self { ir, config, output_handler }
    }

    pub(crate) fn start(&self) -> Result<(), Box<dyn Error>> {
        match self.config.mode {
            Offline => self.evaluate_offline(),
            Online => self.evaluate_online(),
        }
    }

    /// Starts the online evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    fn evaluate_online(&self) -> Result<(), Box<dyn Error>> {
        let (work_tx, work_rx) = unbounded();

        let copy_output_handler = self.output_handler.clone();

        let has_time_driven = !self.ir.time_driven.is_empty();
        if has_time_driven {
            let work_tx_clone = work_tx.clone();
            let ir_clone = self.ir.clone();
            let _ = thread::Builder::new().name("TimeDrivenManager".into()).spawn(move || {
                let time_manager = TimeDrivenManager::setup(ir_clone, SystemTime::now(), copy_output_handler);
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
            let item = match work_rx.recv() {
                Ok(item) => item,
                Err(e) => panic!("Both producers hung up! {}", e),
            };
            self.output_handler.debug(|| format!("Received {:?}.", item));
            match item {
                WorkItem::Event(e, ts) => self.evaluate_event_item(&mut evaluator, &e, ts),
                WorkItem::Time(t, ts) => self.evaluate_timed_item(&mut evaluator, &t, ts),
                WorkItem::End => {
                    self.output_handler.output(|| "Finished entire input. Terminating.");
                    std::process::exit(0);
                }
            }
        }
    }

    /// Starts the offline evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    fn evaluate_offline(&self) -> Result<(), Box<dyn Error>> {
        // Use a bounded channel for offline mode, as we "control" time.
        let (work_tx, work_rx) = bounded(CAP_WORK_QUEUE);
        let (time_tx, time_rx) = bounded(1);

        let output_copy_handler = self.output_handler.clone();

        let ir_clone = self.ir.clone();
        let cfg_clone = self.config.clone();
        let edm_thread = thread::Builder::new()
            .name("EventDrivenManager".into())
            .spawn(move || {
                let event_manager = EventDrivenManager::setup(ir_clone, cfg_clone, output_copy_handler);
                event_manager
                    .start_offline(work_tx, time_tx)
                    .unwrap_or_else(|e| panic!("EventDrivenManager failed: {}", e));
            })
            .unwrap_or_else(|e| panic!("Failed to start EventDrivenManager thread: {}", e));

        let start_time = match time_rx.recv() {
            Err(e) => panic!("Did not receive a start event in offline mode! {}", e),
            Ok(ts) => ts,
        };

        let has_time_driven = !self.ir.time_driven.is_empty();
        let ir_clone = self.ir.clone();
        let output_copy_handler = self.output_handler.clone();
        let time_manager = TimeDrivenManager::setup(ir_clone, start_time, output_copy_handler);
        let (wait_time, mut due_streams) =
            if has_time_driven { time_manager.get_current_deadline(start_time) } else { (Duration::default(), vec![]) };
        let mut next_deadline = start_time + wait_time;

        let output_copy_handler = self.output_handler.clone();
        let mut evaluatordata =
            EvaluatorData::new(self.ir.clone(), start_time, self.config.clone(), output_copy_handler);

        let mut evaluator = evaluatordata.as_Evaluator();

        'outer: loop {
            let local_queue = work_rx.recv().unwrap_or_else(|e| panic!("EventDrivenManager hung up! {}", e));
            for item in local_queue {
                self.output_handler.debug(|| format!("Received {:?}.", item));
                match item {
                    WorkItem::Event(e, ts) => {
                        if has_time_driven {
                            while ts >= next_deadline {
                                // Go back in time, evaluate,...
                                self.evaluate_timed_item(&mut evaluator, &due_streams, next_deadline);
                                let (wait_time, due) = time_manager.get_current_deadline(next_deadline);
                                assert!(wait_time > Duration::from_secs(0));
                                next_deadline += wait_time;
                                due_streams = due;
                            }
                        }
                        self.evaluate_event_item(&mut evaluator, &e, ts)
                    }
                    WorkItem::Time(_, _) => panic!("Received time command in offline mode."),
                    WorkItem::End => {
                        self.output_handler.output(|| "Finished entire input. Terminating.");
                        self.output_handler.terminate();
                        break 'outer;
                    }
                }
            }
        }

        edm_thread.join().expect("Could not join on EventDrivenManger thread");
        Ok(())
    }

    #[inline]
    pub(crate) fn evaluate_timed_item(&self, evaluator: &mut Evaluator, t: &TimeEvaluation, ts: SystemTime) {
        self.output_handler.new_event();
        evaluator.eval_time_driven_outputs(t.as_slice(), ts);
    }

    #[inline]
    pub(crate) fn evaluate_event_item(&self, evaluator: &mut Evaluator, e: &EventEvaluation, ts: SystemTime) {
        self.output_handler.new_event();
        evaluator.eval_event(e.as_slice(), ts)
    }
}
