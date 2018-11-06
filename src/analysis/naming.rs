//! This module provides naming analysis for a given Lola AST.

use super::super::ast::*;
use super::super::LolaSpec;
use std::collections::HashMap;

pub(crate) struct NamingAnalysis<'a> {
    declarations: ScopedDecl<'a>,
    result: HashMap<NodeId, Declaration<'a>>,
    // TODO: need some kind of error reporting, probably stored in a vector
}

impl<'a> NamingAnalysis<'a> {
    pub(crate) fn new() -> Self {
        NamingAnalysis {
            declarations: ScopedDecl::new(),
            result: HashMap::new(),
        }
    }

    pub(crate) fn check(&mut self, spec: &'a LolaSpec) {
        // Store global declarations, i.e., constants, inputs, and outputs of the given specification
        for constant in &spec.constants {
            self.declarations
                .add_decl_for(&constant.name.name, constant.into());
        }
        for input in &spec.inputs {
            self.declarations
                .add_decl_for(&input.name.name, input.into());
        }
        for output in &spec.outputs {
            self.declarations
                .add_decl_for(&output.name.name, output.into());
        }

        // recurse into expressions and check them
        for output in &spec.outputs {
            self.declarations.push();
            self.check_expression(&output.expression);
            self.declarations.pop();
        }

        for trigger in &spec.trigger {
            self.declarations.push();
            self.check_expression(&trigger.expression);
            self.declarations.pop();
        }
    }

    fn check_expression(&mut self, expression: &'a Expression) {
        use self::ExpressionKind::*;

        match &expression.kind {
            Ident(ident) => {
                if let Some(decl) = self.declarations.get_decl_for(&ident.name) {
                    assert!(expression.id != NodeId::DUMMY);
                    self.result.insert(expression.id, decl);
                } else {
                    // unbounded variable, store error to display to user
                    unimplemented!();
                }
            }
            _ => unreachable!(),
        }
        unimplemented!();
    }
}

/// Provides a mapping from `String` to `Declaration` and is able to handle different scopes.
struct ScopedDecl<'a> {
    scopes: Vec<HashMap<&'a str, Declaration<'a>>>,
}

impl<'a> ScopedDecl<'a> {
    fn new() -> Self {
        ScopedDecl {
            scopes: vec![HashMap::new()],
        }
    }

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        assert!(self.scopes.len() > 1);
        self.scopes.pop();
    }

    fn get_decl_for(&self, name: &'a str) -> Option<Declaration<'a>> {
        for scope in self.scopes.iter().rev() {
            if let Some(&decl) = scope.get(name) {
                return Some(decl);
            }
        }
        None
    }

    fn add_decl_for(&mut self, name: &'a str, decl: Declaration<'a>) {
        assert!(self.scopes.last().is_some());
        self.scopes.last_mut().unwrap().insert(name, decl);
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum Declaration<'a> {
    Const(&'a Constant),
    In(&'a Input),
    Out(&'a Output),
}

impl<'a> Into<Declaration<'a>> for &'a Constant {
    fn into(self) -> Declaration<'a> {
        Declaration::Const(self)
    }
}

impl<'a> Into<Declaration<'a>> for &'a Input {
    fn into(self) -> Declaration<'a> {
        Declaration::In(self)
    }
}

impl<'a> Into<Declaration<'a>> for &'a Output {
    fn into(self) -> Declaration<'a> {
        Declaration::Out(self)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    // TODO: implement test cases
}
