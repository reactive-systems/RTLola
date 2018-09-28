//! Parser for the Lola language.

#[macro_use]
extern crate pest;
#[macro_use]
extern crate pest_derive;
#[macro_use]
extern crate lazy_static;

mod ast;
mod parse;

// Re-export on the root level
pub use ast::{LanguageSpec, LolaSpec};
