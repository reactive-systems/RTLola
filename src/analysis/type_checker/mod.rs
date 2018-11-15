use ::ast::LolaSpec;
use super::naming::DeclarationTable;

mod checker;
mod candidates;
mod type_error;

use super::type_checker::checker::*;
use super::type_checker::type_error::TypeError;
use ast_node::NodeId;
use super::common::Type;
use super::AnalysisError;

use std::collections::{HashMap};

type TypeTable = HashMap<NodeId, Vec<Type>>;

#[derive(Debug)]
pub(crate) struct TypeCheckResult<'a> {
    type_table: TypeTable,
    errors: Vec<Box<TypeError<'a>>>,
}

pub(crate) fn type_check<'a>(dt: &'a DeclarationTable, spec: &'a LolaSpec) -> TypeCheckResult<'a> {
    let tc = TypeChecker::new(dt, spec);
    let tc_res = tc.check();
    tc_res.errors.iter().for_each(|e| println!("{:?}", e)); // TODO: pretty print.
    tc_res
}