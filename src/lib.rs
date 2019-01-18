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
mod lowering;
mod parse;
mod print;
mod reporting;
mod stdlib;
mod ty;

pub trait LolaBackend {
    /// Returns collection of feature flags supported by the `LolaBackend`.
    fn supported_feature_flags() -> Vec<FeatureFlag>;
}

// Replace by more elaborate interface.
pub fn parse(spec_str: &str) -> Option<LolaIR> {
    let spec = crate::parse::parse(&spec_str).ok()?;
    let mapper = crate::parse::SourceMapper::new(std::path::PathBuf::new(), spec_str);
    let handler = reporting::Handler::new(mapper);
    let analysis_result = analysis::analyze(&spec, &handler);
    Some(lowering::Lowering::new(&spec, &analysis_result).lower())
}

// Re-export on the root level
pub use crate::ir::*;
