//! Parser for the Lola language.

#[allow(unused_imports)]
#[macro_use]
extern crate pest;
#[macro_use]
extern crate pest_derive;
#[macro_use]
extern crate lazy_static;
extern crate clap;
extern crate strum;
//#[macro_use]
extern crate strum_macros;

extern crate ast_node;
#[macro_use]
extern crate ast_node_derive;

mod analysis;
pub mod app;
mod ast;
mod parse;
mod print;

// Re-export on the root level
pub use crate::ast::{LanguageSpec, LolaSpec};
