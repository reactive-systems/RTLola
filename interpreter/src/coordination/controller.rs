use super::event_driven_manager::EventDrivenManager;
use super::time_driven_manager::TimeDrivenManager;
use super::{WorkItem, CAP_WORK_QUEUE};
use crate::basics::{EvalConfig, ExecutionMode::*, OutputHandler, Time};
use crate::coordination::monitor::Monitor;
use crate::coordination::{EventEvaluation, TimeEvaluation};
use crate::evaluator::{Evaluator, EvaluatorData};
use crossbeam_channel::{bounded, unbounded};
use either::Either;
use rtlola_frontend::ir::Deadline;
use rtlola_frontend::ir::{OutputReference, RTLolaIR};
use std::error::Error;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

pub(crate) struct Controller {
    ir: RTLolaIR,

    config: EvalConfig,

    /// Handles all kind of output behavior according to config.
    pub(crate) output_handler: Arc<OutputHandler>,
}

impl Controller {
    pub(crate) fn new(ir: RTLolaIR, config: EvalConfig) -> Self {
        let output_handler = Arc::new(OutputHandler::new(&config, ir.triggers.len()));
        Self { ir, config, output_handler }
    }

    pub(crate) fn start(self) -> Result<Either<Monitor, Arc<OutputHandler>>, Box<dyn Error>> {
        // TODO: Returning the Arc here makes no sense, fix asap.
        match self.config.mode {
            Offline => self.evaluate_offline().map(|_| Either::Right(self.output_handler)),
            Online => self.evaluate_online().map(|_| Either::Right(self.output_handler)),
            API => Ok(Either::Left(self.setup_api())),
        }
    }

    fn setup_api(self) -> Monitor {
        Monitor::setup(self.ir, self.output_handler, self.config)
    }

    /// Starts the online evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    fn evaluate_online(&self) -> Result<(), Box<dyn Error>> {
        let (work_tx, work_rx) = unbounded();
        let now = Instant::now();

        let copy_output_handler = self.output_handler.clone();

        let has_time_driven = !self.ir.time_driven.is_empty();
        if has_time_driven {
            let work_tx_clone = work_tx.clone();
            let ir_clone = self.ir.clone();
            let _ = thread::Builder::new().name("TimeDrivenManager".into()).spawn(move || {
                let time_manager =
                    TimeDrivenManager::setup(ir_clone, copy_output_handler).unwrap_or_else(|s| panic!(s));
                time_manager.start_online(now, work_tx_clone);
            });
        };

        let copy_output_handler = self.output_handler.clone();

        let ir_clone = self.ir.clone();
        let cfg_clone = self.config.clone();
        // TODO: Wait until all events have been read.
        let _event = thread::Builder::new().name("EventDrivenManager".into()).spawn(move || {
            let event_manager = EventDrivenManager::setup(ir_clone, cfg_clone, copy_output_handler, now);
            event_manager.start_online(work_tx);
        });

        let copy_output_handler = self.output_handler.clone();
        let evaluatordata = EvaluatorData::new(self.ir.clone(), self.config.clone(), copy_output_handler, now);

        let mut evaluator = evaluatordata.into_evaluator();

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
                let event_manager = EventDrivenManager::setup(ir_clone, cfg_clone, output_copy_handler, Instant::now());
                event_manager
                    .start_offline(work_tx, time_tx)
                    .unwrap_or_else(|e| unreachable!("EventDrivenManager failed: {}", e));
            })
            .unwrap_or_else(|e| unreachable!("Failed to start EventDrivenManager thread: {}", e));

        let start_time = match time_rx.recv() {
            Err(e) => unreachable!("Did not receive a start event in offline mode! {}", e),
            Ok(ts) => ts,
        };

        let mut start_time_ref = self.output_handler.start_time.lock().unwrap();
        *start_time_ref = start_time;
        drop(start_time_ref);

        let has_time_driven = !self.ir.time_driven.is_empty();
        let ir_clone = self.ir.clone();
        let output_copy_handler = self.output_handler.clone();
        let time_manager = TimeDrivenManager::setup(ir_clone, output_copy_handler)?;
        let hlp = vec![];
        let mut due_streams = if has_time_driven {
            // timed streams at time 0
            time_manager.get_last_due()
        } else {
            &hlp
        };
        let mut next_deadline = Duration::default();
        let mut deadline_cycle = time_manager.get_deadline_cycle();

        let output_copy_handler = self.output_handler.clone();
        let evaluatordata =
            EvaluatorData::new(self.ir.clone(), self.config.clone(), output_copy_handler, Instant::now());

        let mut evaluator = evaluatordata.into_evaluator();

        let mut current_time = Time::default();
        'outer: loop {
            let local_queue = work_rx.recv().unwrap_or_else(|e| panic!("EventDrivenManager hung up! {}", e));
            for item in local_queue {
                self.output_handler.debug(|| format!("Received {:?}.", item));
                match item {
                    WorkItem::Event(e, ts) => {
                        while has_time_driven && ts > next_deadline {
                            // Go back in time, evaluate,...
                            due_streams = self.schedule_timed(
                                &mut evaluator,
                                &mut deadline_cycle,
                                due_streams,
                                &mut next_deadline,
                            );
                        }
                        self.output_handler.debug(|| format!("Schedule Event {:?}.", (&e, ts)));
                        self.evaluate_event_item(&mut evaluator, &e, ts);
                        current_time = ts;
                    }
                    WorkItem::Time(_, _) => panic!("Received time command in offline mode."),
                    WorkItem::End => {
                        while has_time_driven && current_time == next_deadline {
                            // schedule last timed event before terminating
                            due_streams = self.schedule_timed(
                                &mut evaluator,
                                &mut deadline_cycle,
                                due_streams,
                                &mut next_deadline,
                            );
                        }
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

    fn schedule_timed<'a>(
        &'a self,
        evaluator: &mut Evaluator,
        mut deadline_iter: impl Iterator<Item = &'a Deadline>,
        due_streams: &Vec<OutputReference>,
        next_deadline: &mut Time,
    ) -> &'a Vec<OutputReference> {
        self.output_handler.debug(|| format!("Schedule Timed-Event {:?}.", (due_streams, *next_deadline)));
        self.evaluate_timed_item(evaluator, due_streams, *next_deadline);
        let deadline = deadline_iter.next().unwrap();
        assert!(deadline.pause > Duration::from_secs(0));
        *next_deadline += deadline.pause;
        &deadline.due
    }

    #[inline]
    pub(crate) fn evaluate_timed_item(&self, evaluator: &mut Evaluator, t: &TimeEvaluation, ts: Time) {
        self.output_handler.new_event();
        evaluator.eval_time_driven_outputs(t.as_slice(), ts);
    }

    #[inline]
    pub(crate) fn evaluate_event_item(&self, evaluator: &mut Evaluator, e: &EventEvaluation, ts: Time) {
        self.output_handler.new_event();
        evaluator.eval_event(e.as_slice(), ts)
    }
}
