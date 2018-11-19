//! Parser for the Lola language.

#![deny(unsafe_code)] // disallow unsafe code by default

#[macro_use]
extern crate log;
extern crate simplelog;
#[macro_use]
extern crate pest;
#[macro_use]
extern crate pest_derive;
#[macro_use]
extern crate lazy_static;
extern crate ast_node;
extern crate clap;
extern crate termcolor;
#[macro_use]
extern crate ast_node_derive;

mod analysis;
pub mod app;
mod ast;
mod parse;
mod print;

// Re-export on the root level
pub use crate::ast::{LanguageSpec, LolaSpec};
