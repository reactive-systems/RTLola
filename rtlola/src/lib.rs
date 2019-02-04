#![deny(unsafe_code)] // disallow unsafe code by default

pub mod basics;
mod coordination;
mod evaluator;
mod storage;

use crate::coordination::Controller;
use lola_parser::LolaIR;
use std::time::Instant;

pub fn start_evaluation(ir: LolaIR, cfg: EvalConfig, ts: Option<Instant>) -> ! {
    Controller::evaluate(ir, cfg, ts);
}

// Public export.
pub use basics::{EvalConfig, InputSource, OutputChannel, Verbosity};
