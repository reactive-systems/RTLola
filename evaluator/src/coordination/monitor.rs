use crate::basics::{EvalConfig, OutputHandler};
use crate::evaluator::{Evaluator, EvaluatorData};
use std::sync::Arc;
use std::time::Instant;
use streamlab_frontend::ir::LolaIR;

pub struct Monitor {
    eval: Evaluator<'static, 'static>,
    handler: Arc<OutputHandler>,
}

impl Monitor {
    pub(crate) fn setup(ir: LolaIR, handler: Arc<OutputHandler>, config: EvalConfig) -> Monitor {
        // Note: start_time only accessed in online mode.
        let eval_data = EvaluatorData::new(ir.clone(), config, handler.clone(), Instant::now());

        Monitor { eval: eval_data.into_evaluator(), handler }
    }
}
