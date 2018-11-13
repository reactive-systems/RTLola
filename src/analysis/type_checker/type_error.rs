extern crate ast_node;
use ast_node::{NodeId, Span, AstNode};
use std::fmt::{Display, Result, Formatter};
use ::ast::Parameter;
use super::super::AnalysisError;

#[derive(Debug)]
pub enum TypeError<'a> {
    UnknownIdentifier(&'a AstNode<'a>),
    IncompatibleTypes(&'a AstNode<'a>, String),
    InvalidArgument(Parameter, &'a AstNode<'a>, String),
    InvalidType(&'a AstNode<'a>, String),
    UnexpectedNumberOfArguments(&'a AstNode<'a>, String),
}

impl<'a> AnalysisError<'a> for TypeError<'a> {}

impl<'a> Display for TypeError<'a> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        unimplemented!();
    }
}