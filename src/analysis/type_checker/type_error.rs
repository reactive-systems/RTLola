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
    ConstantValueRequired {
        call: &'a AstNode<'a>,
        args: &'a AstNode<'a>,
    }, // Tuple projections require a constant projection value.
}

impl<'a> TypeError<'a> {
    //    pub fn inv_num_of_args(node: &'a AstNode<'a>, expected: u8, was: u8) -> TypeError<'a> {
    //        // TODO Remove for Rust'18.
    //        TypeError::InvalidNumberOfArguments(node, format!("Expected {} but was {}.", expected, was))
    ////    }
    //    pub fn inv_argument(
    //        call: &'a AstNode<'a>,
    //        argument: &'a AstNode<'a>,
    //        msg: String,
    //    ) -> TypeError<'a> {
    //        // TODO Remove for Rust'18.
    //        TypeError::InvalidArgument(call, argument, msg)
    //    }
}

impl<'a> AnalysisError<'a> for TypeError<'a> {}

impl<'a> Display for TypeError<'a> {
    fn fmt(&self, _f: &mut Formatter) -> Result {
        unimplemented!();
    }
}
