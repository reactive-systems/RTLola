extern crate ast_node;
use super::super::AnalysisError;
use ast_node::AstNode;
use std::fmt::{Display, Formatter, Result};

#[derive(Debug, Clone)]
pub enum TypeError<'a> {
    UnknownIdentifier(&'a AstNode<'a>),
    IncompatibleTypes(&'a AstNode<'a>, String),
    InvalidArgument {
        call: &'a AstNode<'a>,
        arg: &'a AstNode<'a>,
        msg: String,
    },
    InvalidNumberOfArguments {
        node: &'a AstNode<'a>,
        expected: u8,
        was: u8,
    },
    UnexpectedNumberOfArguments(&'a AstNode<'a>, String),
    MissingExpression(&'a AstNode<'a>),
    // Tuple projections require a constant projection value.
    ConstantValueRequired {
        call: &'a AstNode<'a>,
        args: &'a AstNode<'a>,
    },
    IncompatibleTiming(&'a AstNode<'a>, String),
}

impl<'a> AnalysisError<'a> for TypeError<'a> {}

impl<'a> Display for TypeError<'a> {
    fn fmt(&self, _f: &mut Formatter) -> Result {
        unimplemented!();
    }
}
