//! Parser for the Lola language.

#![deny(unsafe_code)] // disallow unsafe code by default
#![forbid(unused_must_use)] // disallow discarding errors

extern crate log;

#[macro_use]
extern crate pest_derive;
#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate ast_node_derive;

mod analysis;
pub mod app;
mod ast;
mod ir;
mod parse;
mod print;
mod reporting;
mod util;
mod stdlib;
mod ty;

// Re-export on the root level
pub use crate::ir::LolaIR;
