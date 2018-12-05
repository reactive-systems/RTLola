
#![deny(unsafe_code)] // disallow unsafe code by default

extern crate lola_parser;
mod evaluator;

pub use crate::evaluator::{Evaluator, config::EvalConfig};
