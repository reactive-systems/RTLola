//! Parser for the RTLola language.

#![forbid(unused_must_use)] // disallow discarding errors
#![warn(
    missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]

mod analysis;
pub mod ast;
mod export;
pub mod ir;
mod parse;
mod reporting;
mod stdlib;
pub mod ty;

#[cfg(test)]
mod tests;

// Re-export
pub use ast::RTLolaAst;
pub use export::analyze;
pub use ir::RTLolaIR;
pub use ty::TypeConfig;

/**
This module contains a module for each binary that this crate provides.
*/
pub mod app {
    pub mod analyze;
}

/**
Hold the configuration of the frontend
*/
#[derive(Debug, Clone, Copy)]
pub struct FrontendConfig {
    /**
    Several options regarding the type-system. See the `TypeConfig` documentation for more information.
    */
    pub ty: TypeConfig,
    /**
    A flag whether streams can parameterized.
    */
    pub allow_parameters: bool,
}

impl Default for FrontendConfig {
    fn default() -> Self {
        Self { ty: TypeConfig::default(), allow_parameters: true }
    }
}

// Replace by more elaborate interface.
#[rustfmt::skip]
/**
Parses a RTLola specification.

The string passed in as `spec_str` should be the content of the file specified by `filename`.  
The filename is only used for printing locations.  
See the `FrontendConfig` documentation on more information about the parser options.  
*/
pub fn parse(filename: &str, spec_str: &str, config: FrontendConfig) -> Result<RTLolaIR, String> {
    let mapper = crate::parse::SourceMapper::new(std::path::PathBuf::from(filename), spec_str);
    let handler = reporting::Handler::new(mapper);

    let spec = match crate::parse::parse(&spec_str, &handler, config) {
        Ok(spec) => spec,
        Err(e) => {
            return Err(format!("error: invalid syntax:\n{}", e));
        }
    };

    let analysis_result = analysis::analyze(&spec, &handler, config);
    analysis_result
        .map(|report| ir::lowering::Lowering::new(&spec, &report).lower())
        .map_err(|_| "Analysis failed due to errors in the specification".to_string())
}
