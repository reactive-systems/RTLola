#![deny(unsafe_code)] // disallow unsafe code by default

mod evaluator;
mod util;


// Public export.
pub use crate::evaluator::{config::EvalConfig, Evaluator};
