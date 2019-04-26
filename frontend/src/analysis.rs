//! This module provides analysis steps based on the AST.
//!
//! In detail,
//! * `naming` provides boundedness analysis for identifiers used in the Lola Specification
//! * `id_assignment` assigns unique ids to all nodes of the AST
//! * `type_checker` checks whether components of the AST have a valid type

pub(crate) mod graph_based_analysis;
pub(crate) mod id_assignment;
mod lola_version;
pub(crate) mod naming;

use self::lola_version::LolaVersionAnalysis;
use self::naming::NamingAnalysis;
use super::ast::LolaSpec;
use crate::ast;
use crate::reporting::Handler;
use crate::ty::check::TypeAnalysis;

pub trait AnalysisError<'a>: std::fmt::Debug {}

// Export output types.
pub(crate) use self::graph_based_analysis::GraphAnalysisResult;
pub(crate) use self::naming::DeclarationTable;
pub(crate) use crate::ast::LanguageSpec;
pub(crate) use crate::ty::check::TypeTable;

pub(crate) struct AnalysisResult<'a> {
    pub(crate) declaration_table: Option<DeclarationTable<'a>>,
    pub(crate) type_table: Option<TypeTable>,
    pub(crate) version: Option<LanguageSpec>,
    pub(crate) graph_analysis_result: Option<GraphAnalysisResult>,
}

impl<'a> AnalysisResult<'a> {
    fn new() -> AnalysisResult<'a> {
        AnalysisResult { declaration_table: None, type_table: None, version: None, graph_analysis_result: None }
    }

    pub(crate) fn is_success(&self) -> bool {
        self.declaration_table.is_some()
            && self.type_table.is_some()
            && self.version.is_some()
            && self.graph_analysis_result.is_some()
    }
}

pub(crate) fn analyze<'a, 'b>(spec: &'a LolaSpec, handler: &'b Handler) -> AnalysisResult<'a> {
    let mut result = AnalysisResult::new();

    ast::verify::Verifier::new(spec, handler).check();

    if handler.contains_error() {
        handler.error("aborting due to previous error");
        return result;
    }

    let mut naming_analyzer = NamingAnalysis::new(&handler);
    let decl_table = naming_analyzer.check(spec);

    if handler.contains_error() {
        handler.error("aborting due to previous error");
        return result;
    } else {
        result.declaration_table = Some(decl_table);
    }

    let mut type_analysis =
        TypeAnalysis::new(&handler, result.declaration_table.as_ref().expect("We already checked for naming errors"));
    let type_table = type_analysis.check(&spec);
    assert_eq!(type_table.is_none(), handler.contains_error());

    if handler.contains_error() {
        handler.error("aborting due to previous error");
        return AnalysisResult::new();
    } else {
        result.type_table = type_table;
    }

    let mut version_analyzer =
        LolaVersionAnalysis::new(&handler, result.type_table.as_ref().expect("We already checked for type errors"));
    let version_result = version_analyzer.analyse(spec);
    if version_result.is_none() {
        print!("error");
        return AnalysisResult::new();
    } else {
        result.version = version_result;
    }

    let graph_result = graph_based_analysis::analyze(
        spec,
        &version_analyzer.result,
        result.declaration_table.as_ref().expect("We already checked for naming errors"),
        result.type_table.as_mut().expect("We already checked for type errors"),
        &handler,
    );

    if handler.contains_error() || graph_result.is_none() {
        handler.error("aborting due to previous error");
        return AnalysisResult::new();
    } else {
        result.graph_analysis_result = graph_result;
    }

    result
}
