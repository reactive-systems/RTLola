#![deny(unsafe_code)] // disallow unsafe code by default

pub mod basics;
mod coordination;
mod evaluator;
mod storage;

use crate::coordination::Controller;
use streamlab_frontend::ir::LolaIR;

pub fn start_evaluation_online(ir: LolaIR, cfg: EvalConfig) -> ! {
    Controller::evaluate(ir, cfg, false);
}

pub fn start_evaluation_offline(ir: LolaIR, cfg: EvalConfig) -> ! {
    Controller::evaluate(ir, cfg, true);
}

// Public export.
pub use basics::{EvalConfig, InputSource, OutputChannel, Verbosity};
