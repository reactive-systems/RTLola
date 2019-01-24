//! Parser for the Lola language.

#![deny(unsafe_code)] // disallow unsafe code by default
#![forbid(unused_must_use)] // disallow discarding errors

extern crate log;

#[macro_use]
extern crate pest_derive;
#[macro_use]
extern crate lazy_static;

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
pub fn parse(spec_str: &str) -> LolaIR {
    let spec = match crate::parse::parse(&spec_str) {
        Result::Ok(spec) => spec,
        Result::Err(e) => panic!("{}", e),
    };
    println!("Parsed the following spec: \n{}", spec);
    let mapper = crate::parse::SourceMapper::new(std::path::PathBuf::new(), spec_str);
    let handler = reporting::Handler::new(mapper);
    let analysis_result = analysis::analyze(&spec, &handler);
    if analysis_result.is_success() {
        lowering::Lowering::new(&spec, &analysis_result).lower()
    } else {
        panic!("Error in analysis.")
    }
}

// Re-export on the root level
pub use crate::ir::*;
pub use crate::ty::{FloatTy, IntTy, UIntTy};
