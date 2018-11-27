//! This module provides analysis steps based on the AST.
//!
//! In detail,
//! * `naming` provides boundedness analysis for identifiers used in the Lola Specification

mod common;
mod id_assignment;
mod lola_version;
mod naming;
mod reporting;

use self::lola_version::LolaVersionAnalysis;
use self::naming::{Declaration, NamingAnalysis};
use self::reporting::Handler;
use super::ast::LolaSpec;
use ast_node::AstNode;
use crate::parse::SourceMapper;

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

    let mut version_analyzer = LolaVersionAnalysis::new();
    let version_result = version_analyzer.analyse(spec);
    unimplemented!();
}

//#[allow(dead_code)]
//#[derive(Debug)]
pub enum NamingError<'a> {
    MalformedName(&'a AstNode<'a>),
    ReservedKeyword(&'a AstNode<'a>),
    NameNotFound(&'a AstNode<'a>),
    TypeNotFound(&'a AstNode<'a>),
    NotAType(&'a AstNode<'a>),
    TypeNotAllowedHere(&'a AstNode<'a>),
    NotAStream(&'a AstNode<'a>),
    UnnamedTypeDeclaration(&'a AstNode<'a>),
    NameAlreadyUsed {
        current: &'a AstNode<'a>,
        previous: Declaration<'a>,
    },
    TriggerWithSameName {
        current: &'a AstNode<'a>,
        previous: &'a AstNode<'a>,
    },
    FieldWithSameName {
        current: &'a AstNode<'a>,
        previous: &'a AstNode<'a>,
    },
}

impl<'a> std::fmt::Debug for NamingError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        match self {
            NamingError::MalformedName(node) => {
                write!(f, "Name is not well-formed: {:?}", node.span())
            }
            NamingError::ReservedKeyword(node) => {
                write!(f, "Name is a reserved keyword: {:?}", node.span())
            }
            NamingError::NameNotFound(node) => write!(f, "Name not found: {:?}", node.span()),
            NamingError::TypeNotFound(node) => write!(f, "Type not found: {:?}", node.span()),
            NamingError::NotAType(node) => write!(f, "This is not a type: {:?}", node.span()),
            NamingError::TypeNotAllowedHere(node) => {
                write!(f, "Type not allowed here: {:?}", node.span())
            }
            NamingError::NotAStream(node) => write!(f, "This is not a stream: {:?}", node.span()),
            NamingError::UnnamedTypeDeclaration(node) => {
                write!(f, "Declared type has no name: {:?}", node.span())
            }
            NamingError::NameAlreadyUsed { current, previous } => write!(
                f,
                "Name {:?} was already used at the same level at {:?}",
                current.span(),
                previous
            ),
            NamingError::TriggerWithSameName { current, previous } => write!(
                f,
                "Name {:?} was already used for another trigger {:?}",
                current.span(),
                previous.span()
            ),
            NamingError::FieldWithSameName { current, previous } => write!(
                f,
                "Name {:?} was already used for another field {:?} in the same type declaration",
                current.span(),
                previous.span()
            ),
        }
    }
}

impl<'a> AnalysisError<'a> for NamingError<'a> {}
