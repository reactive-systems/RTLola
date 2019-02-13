#![deny(unsafe_code)] // disallow unsafe code by default

pub mod basics;
mod coordination;
mod evaluator;
mod storage;

use crate::coordination::Controller;
use lola_parser::ir::LolaIR;

pub fn start_evaluation_online(ir: LolaIR, cfg: EvalConfig) -> ! {
    Controller::evaluate(ir, cfg, true);
}

pub fn start_evaluation_offline(ir: LolaIR, cfg: EvalConfig) -> ! {
    Controller::evaluate(ir, cfg, false);
}

// Public export.
pub use basics::{EvalConfig, InputSource, OutputChannel, Verbosity};
