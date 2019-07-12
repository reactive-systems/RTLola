//! Parser for the Lola language.

#![deny(unsafe_code)] // disallow unsafe code by default
#![forbid(unused_must_use)] // disallow discarding errors

mod analysis;
mod ast;
pub mod export;
pub mod ir;
mod parse;
mod reporting;
mod stdlib;
#[cfg(test)]
mod tests;
mod ty;

// module containing the code for the executables
pub mod app {
    pub mod analyze;
}

use crate::ir::{FeatureFlag, LolaIR};

pub trait LolaBackend {
    /// Returns collection of feature flags supported by the `LolaBackend`.
    fn supported_feature_flags() -> Vec<FeatureFlag>;
}

// Replace by more elaborate interface.
pub fn parse(spec_str: &str) -> Result<LolaIR, String> {
    let mapper = crate::parse::SourceMapper::new(std::path::PathBuf::new(), spec_str);
    let handler = reporting::Handler::new(mapper);

    let spec = match crate::parse::parse(&spec_str, &handler) {
        Result::Ok(spec) => spec,
        Result::Err(e) => {
            return Err(format!("error: invalid syntax:\n{}", e));
        }
    };

    let analysis_result = analysis::analyze(&spec, &handler);
    if analysis_result.is_success() {
        Ok(ir::lowering::Lowering::new(&spec, &analysis_result).lower())
    } else {
        Err("Analysis failed due to errors in the specification".to_string())
    }
}
