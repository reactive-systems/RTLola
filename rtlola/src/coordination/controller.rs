use lola_parser::ir::{LolaIR, StreamReference};
use std::sync::mpsc;
use std::thread;
use std::time::SystemTime;

use crate::basics::EvalConfig;
use crate::basics::OutputHandler;
use crate::evaluator::Evaluator;
use crate::storage::Value;

use super::event_driven_manager::{EventDrivenManager, EventEvaluation};
use super::time_driven_manager::{TimeDrivenManager, TimeEvaluation};

pub struct Controller {
    /// Handles all kind of output behavior according to config.
    output_handler: OutputHandler,

    /// Handles evaluating stream expressions and storeage of values.
    evaluator: Evaluator,
}

impl lola_parser::LolaBackend for Controller {
    fn supported_feature_flags() -> Vec<lola_parser::ir::FeatureFlag> {
        unimplemented!()
    }
}

impl Controller {
    /// Starts the evaluation process, i.e. periodically computes outputs for time-driven streams
    /// and fetches/expects events from specified input source.
    pub fn evaluate(ir: LolaIR, config: EvalConfig, online: bool) -> ! {
        let (work_tx, work_rx) = mpsc::channel();
        let (time_tx, time_rx) = mpsc::channel();
        let (ack_tx, ack_rx) = mpsc::channel();

        let ir_clone_1 = ir.clone();
        let ir_clone_2 = ir.clone();
        let cfg_clone_1 = config.clone();
        let cfg_clone_2 = config.clone();
        let work_tx_clone = work_tx.clone();

        // TODO: Wait until all events have been read.
        let _event = thread::Builder::new().name("EventDrivenManager".to_string()).spawn(move || {
            let event_manager = EventDrivenManager::setup(ir_clone_1, cfg_clone_1);
            if online {
                event_manager.start_online(work_tx_clone)
            } else {
                event_manager.start_offline(work_tx_clone, time_tx, ack_rx);
            }
        });
        let _ = thread::Builder::new().name("TimeDrivenManager".to_string()).spawn(move || {
            let time_manager = TimeDrivenManager::setup(ir_clone_2, cfg_clone_2);
            if online {
                time_manager.start_online(Some(SystemTime::now()), work_tx);
            } else {
                time_manager.start_offline(work_tx, time_rx, ack_tx);
            }
        });

        let output_handler = OutputHandler::new(&config);

        // Layers of event based output streams
        let layers = ir.get_event_driven_layers();
        output_handler.debug(|| format!("Evaluation layers: {:?}", layers));

        let e = if online {
            Evaluator::new(ir, SystemTime::now(), config.clone())
        } else {
            match work_rx.recv() {
                Err(e) => panic!("Both producers hung up! {}", e),
                Ok(wi) => match wi {
                    WorkItem::Start(ts) => Evaluator::new(ir, ts, config.clone()),
                    _ => panic!("Did not receive a start event in offline mode!"),
                },
            }
        };

        let mut ctrl = Controller { output_handler, evaluator: e };

        loop {
            match work_rx.recv() {
                Ok(item) => ctrl.eval_workitem(item, &layers),
                Err(e) => panic!("Both producers hung up! {}", e),
            }
        }
    }

    fn eval_workitem(&mut self, wi: WorkItem, layers: &Vec<Vec<StreamReference>>) {
        self.output_handler.debug(|| format!("Received {:?}.", wi));
        match wi {
            WorkItem::Event(e, ts) => self.evaluate_event_item(e, ts, layers),
            WorkItem::Time(t, ts) => self.evaluate_timed_item(t, ts),
            WorkItem::Start(_) => panic!("Received spurious start command."),
            WorkItem::End => {
                self.output_handler.trigger(|| "Finished entire input. Terminating.");
                std::process::exit(0);
            }
        }
    }

    fn evaluate_timed_item(&mut self, t: TimeEvaluation, ts: SystemTime) {
        t.into_iter().for_each(|s| self.evaluate_single_output(s, ts));
    }

    fn evaluate_event_item(&mut self, e: EventEvaluation, ts: SystemTime, layers: &Vec<Vec<StreamReference>>) {
        self.evaluate_event(e, ts);
        layers.iter().for_each(|layer| self.evaluate_all_outputs(layer, ts));
    }

    fn evaluate_event(&mut self, event: Vec<(StreamReference, Value)>, ts: SystemTime) {
        event.into_iter().for_each(|(sr, v)| self.evaluator.accept_input(sr, v, ts));
    }

    fn evaluate_all_outputs(&mut self, streams: &Vec<StreamReference>, ts: SystemTime) {
        streams.iter().for_each(|s| self.evaluate_single_output(*s, ts))
    }

    fn evaluate_single_output(&mut self, stream: StreamReference, ts: SystemTime) {
        let inst = (stream.out_ix(), Vec::new());
        self.evaluator.eval_stream(inst, ts);
    }
}

#[derive(Debug)]
pub(crate) enum WorkItem {
    Event(EventEvaluation, SystemTime),
    Time(TimeEvaluation, SystemTime),
    Start(SystemTime),
    End,
}
