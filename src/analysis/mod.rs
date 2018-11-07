//! This module provides analysis steps based on the AST.
//!
//! In detail,
//! * `naming` provides boundedness analysis for identifiers used in the Lola Specification

mod common;
mod id_assignment;
mod naming;

use super::LolaSpec;

pub trait AnalysisError: std::fmt::Debug {}

#[derive(Debug)]
pub struct Report {
    errors: Vec<Box<AnalysisError>>,
}

pub fn analyze(spec: &mut LolaSpec) -> Report {
    id_assignment::assign_ids(spec);
    let mut naming_analyzer = naming::NamingAnalysis::new();
    naming_analyzer.check(spec);
    unimplemented!();
}
