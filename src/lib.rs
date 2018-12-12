//! Parser for the Lola language.

#![deny(unsafe_code)] // disallow unsafe code by default

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
mod parse;
mod print;
mod reporting;

// Re-export on the root level
pub use crate::ast::{LanguageSpec, LolaSpec};
