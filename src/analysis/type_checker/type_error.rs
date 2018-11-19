extern crate ast_node;
use super::super::AnalysisError;
use ast_node::AstNode;
use std::fmt::{Display, Formatter, Result};

#[derive(Debug, Clone)]
pub enum TypeError<'a> {
    UnknownIdentifier(&'a AstNode<'a>),
    IncompatibleTypes(&'a AstNode<'a>, String),
    InvalidArgument(&'a AstNode<'a>, &'a AstNode<'a>, String), // TODO: First call, then arg.
    InvalidNumberOfArguments(&'a AstNode<'a>, String), // TODO: In Rust'18  use expected: u8 and was: u8 rather than string.
    InvalidType(&'a AstNode<'a>, String),
    UnexpectedNumberOfArguments(&'a AstNode<'a>, String),
    MissingExpression(&'a AstNode<'a>),
    ConstantValueRequired(&'a AstNode<'a>, &'a AstNode<'a>), // Tuple projections require a constant projection value. TODO: first call, then arg.
}

impl<'a> TypeError<'a> {
    pub fn inv_num_of_args(node: &'a AstNode<'a>, expected: u8, was: u8) -> TypeError<'a> {
        // TODO Remove for Rust'18.
        TypeError::InvalidNumberOfArguments(node, format!("Expected {} but was {}.", expected, was))
    }
    pub fn inv_argument(
        call: &'a AstNode<'a>,
        argument: &'a AstNode<'a>,
        msg: String,
    ) -> TypeError<'a> {
        // TODO Remove for Rust'18.
        TypeError::InvalidArgument(call, argument, msg)
    }
}

impl<'a> AnalysisError<'a> for TypeError<'a> {}

impl<'a> Display for TypeError<'a> {
    fn fmt(&self, _f: &mut Formatter) -> Result {
        unimplemented!();
    }
}
