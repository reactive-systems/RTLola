#![deny(unsafe_code)] // disallow unsafe code by default

pub mod basics;
mod closuregen;
mod coordination;
mod evaluator;
mod storage;

use crate::coordination::Controller;
use streamlab_frontend::ir::LolaIR;

pub fn start_evaluation_online(ir: LolaIR, cfg: EvalConfig) -> ! {
    Controller::evaluate_online(ir, cfg);
}

pub fn start_evaluation_offline(ir: LolaIR, cfg: EvalConfig) -> ! {
    Controller::evaluate_offline(ir, cfg);
}

// Public export.
pub use basics::{EvalConfig, InputSource, OutputChannel, Verbosity};
