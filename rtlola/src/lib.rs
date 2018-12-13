
#![deny(unsafe_code)] // disallow unsafe code by default

extern crate lola_parser;
mod evaluator;
mod util;

pub use crate::evaluator::{Evaluator, config::EvalConfig};
