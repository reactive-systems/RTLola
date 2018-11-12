//! This module provides analysis steps based on the AST.
//!
//! In detail,
//! * `naming` provides boundedness analysis for identifiers used in the Lola Specification

mod common;
mod id_assignment;
mod naming;

use super::LolaSpec;
use crate::parse::Ident;

pub trait AnalysisError<'a>: std::fmt::Debug {}

#[derive(Debug)]
pub struct Report<'a> {
    errors: Vec<Box<AnalysisError<'a>>>,
}

pub fn analyze(spec: &mut LolaSpec) -> Report {
    id_assignment::assign_ids(spec);
    let mut naming_analyzer = naming::NamingAnalysis::new();
    naming_analyzer.check(spec);
    unimplemented!();
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum NamingError<'a> {
    // TODO we first need a common trait for the AST nodes
    NameNotFound(&'a Ident),
    NameAlreadyUsed(&'a Ident, &'a Ident), // current, previous use
}

impl<'a> AnalysisError<'a> for NamingError<'a> {}
