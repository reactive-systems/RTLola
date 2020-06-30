//! This module provides analysis steps based on the AST.
//!
//! In detail,
//! * `naming` provides boundedness analysis for identifiers used in the Lola Specification
//! * `id_assignment` assigns unique ids to all nodes of the AST
//! * `type_checker` checks whether components of the AST have a valid type

pub(crate) mod graph_based_analysis;
// pub(crate) mod id_assignment;
pub(crate) mod naming;

use self::naming::NamingAnalysis;
use crate::ast;
use crate::ast::RTLolaAst;
use crate::reporting::Handler;
use crate::ty::check::TypeAnalysis;
use crate::FrontendConfig;

// Export output types.
pub(crate) use self::graph_based_analysis::GraphAnalysisResult;
pub(crate) use self::naming::DeclarationTable;
pub(crate) use crate::ty::check::TypeTable;

pub(crate) struct Report {
    pub(crate) declaration_table: DeclarationTable,
    pub(crate) type_table: TypeTable,
    pub(crate) graph_analysis_result: GraphAnalysisResult,
}

impl Report {
    fn new(
        declaration_table: DeclarationTable,
        type_table: TypeTable,
        graph_analysis_result: GraphAnalysisResult,
    ) -> Report {
        Report { declaration_table, type_table, graph_analysis_result }
    }
}

pub(crate) fn analyze(spec: &RTLolaAst, handler: &Handler, config: FrontendConfig) -> Result<Report, ()> {
    ast::verify::Verifier::new(spec, handler).check();

    if handler.contains_error() {
        handler.error("aborting due to previous error");
        return Err(());
    }

    let mut naming_analyzer = NamingAnalysis::new(&handler, config);
    let mut decl_table = naming_analyzer.check(spec);

    if handler.contains_error() {
        handler.error("aborting due to previous error");
        return Err(());
    }

    let mut type_analysis = TypeAnalysis::new(&handler, &mut decl_table);
    let type_table = type_analysis.check(&spec);
    assert_eq!(type_table.is_none(), handler.contains_error());

    if handler.contains_error() {
        handler.error("aborting due to previous error");
        return Err(());
    }

    let tt = type_table.unwrap();

    let graph_result = graph_based_analysis::analyze(spec, &decl_table, &tt, &handler);

    if handler.contains_error() || graph_result.is_err() {
        handler.error("aborting due to previous error");
        return Err(());
    }

    let graph_res = graph_result.unwrap();

    Ok(Report::new(decl_table, tt, graph_res))
}
