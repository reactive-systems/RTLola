#![deny(unsafe_code)] // disallow unsafe code by default

mod evaluator;
mod coordination;
pub mod basics;
mod storage;

use std::time::Instant;
use lola_parser::LolaIR;
use crate::coordination::Controller;

pub fn start_evaluation(ir: LolaIR, cfg: EvalConfig, ts: Option<Instant>) -> ! {
    Controller::evaluate(ir, cfg, ts);
}

// Public export.
pub use basics::{EvalConfig, Verbosity, InputSource, OutputChannel};
