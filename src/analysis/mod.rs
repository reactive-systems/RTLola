//! This module provides analysis steps based on the AST.
//!
//! In detail,
//! * `naming` provides boundedness analysis for identifiers used in the Lola Specification
//! * `id_assignment` assigns unique ids to all nodes of the AST
//! * `type_checker` checks whether components of the AST have a valid type

mod common;
mod id_assignment;
mod lola_version;
pub(crate) mod naming;
pub(crate) mod typing;

use self::lola_version::LolaVersionAnalysis;
use self::naming::NamingAnalysis;
use self::typing::TypeAnalysis;
use super::ast::LolaSpec;
use crate::parse::SourceMapper;
use crate::reporting::Handler;

pub trait AnalysisError<'a>: std::fmt::Debug {}

pub(crate) fn analyze(spec: &mut LolaSpec, mapper: SourceMapper) -> bool {
    let handler = Handler::new(mapper);
    id_assignment::assign_ids(spec);
    let mut naming_analyzer = NamingAnalysis::new(&handler);
    naming_analyzer.check(spec);

    if handler.contains_error() {
        handler.error("aborting due to previous error");
        return false;
    }

    let mut type_analysis = TypeAnalysis::new(&handler, &naming_analyzer.result);
    type_analysis.check(&spec);
    if handler.contains_error() {
        handler.error("aborting due to previous error");
        return false;
    }

    let mut version_analyzer = LolaVersionAnalysis::new();
    let version_result = version_analyzer.analyse(spec);
    unimplemented!();
}
