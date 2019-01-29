#![deny(unsafe_code)] // disallow unsafe code by default

mod evaluator;
mod coordination;
pub mod basics;
mod storage;

use lola_parser::LolaIR;
use crate::coordination::Controller;

pub fn start_evaluation(ir: LolaIR, cfg: EvalConfig) -> ! {
    Controller::evaluate(ir, cfg);
}

// Public export.
pub use basics::{EvalConfig, Verbosity, InputSource, OutputChannel};
