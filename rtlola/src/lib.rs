#![deny(unsafe_code)] // disallow unsafe code by default

pub mod basics;
mod coordination;
mod evaluator;
mod storage;

use crate::coordination::Controller;
use lola_parser::LolaIR;
use std::time::SystemTime;

pub fn start_evaluation_online(ir: LolaIR, cfg: EvalConfig, ts: Option<SystemTime>) -> ! {
    Controller::evaluate(ir, cfg, ts, true);
}

pub fn start_evaluation_offline(ir: LolaIR, cfg: EvalConfig, ts: Option<SystemTime>) -> ! {
    Controller::evaluate(ir, cfg, ts, false);
}

// Public export.
pub use basics::{EvalConfig, InputSource, OutputChannel, Verbosity};
