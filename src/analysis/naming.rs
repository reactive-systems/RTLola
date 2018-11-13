//! This module provides naming analysis for a given Lola AST.

use super::super::ast::*;
use super::AnalysisError;
use ast_node::{NodeId, AstNode};
use super::super::ast::Offset::*;
use std::collections::HashMap;
use strum::AsStaticRef;
use strum::IntoEnumIterator;

pub(crate) type DeclarationTable<'a> = HashMap<NodeId, Declaration<'a>>;

pub(crate) struct NamingAnalysis<'a> {
    declarations: ScopedDecl<'a>,
    result: DeclarationTable<'a>,
    #[allow(dead_code)]
    errors: Vec<Box<AnalysisError<'a>>>,
    // TODO: need some kind of error reporting, probably stored in a vector
}

fn declaration_is_type(decl: &Declaration) -> bool {
    match decl {
        Declaration::UserDefinedType(_) | Declaration::BuiltinType(_) => true,
        Declaration::Const(_)
        | Declaration::In(_)
        | Declaration::Out(_)
        | Declaration::StreamParameter(_) => false,
    }
}

impl<'a> NamingAnalysis<'a> {
    pub(crate) fn new() -> Self {
        let mut scoped_decls = ScopedDecl::new();

        for (name, ty) in super::common::BuiltinType::all() {
            scoped_decls.add_decl_for(
                name,
                Declaration::BuiltinType(ty),
            );
        }

        NamingAnalysis {
            declarations: scoped_decls,
            result: HashMap::new(),
            errors: Vec::new(),
        }
    }

    fn check_type(&mut self, ty: &Type) {
        match ty.kind {
            TypeKind::Malformed(ref _name) => {
                // TODO warn about wrong name
            }
            TypeKind::Simple(ref name) => {
                if let Some(decl) = self.declarations.get_decl_for(&name) {
                    if !declaration_is_type(&decl) {
                        // TODO it is NOT a type
                        unimplemented!();
                    }
                } else {
                    // it does not exist
                    unimplemented!()
                }
            }
            TypeKind::Tuple(ref elements) => elements.iter().for_each(|ty| {
                self.check_type(ty);
            }),
        }
    }

    fn check_param(&mut self, param: &'a Parameter) {
        // check the name
        if let Some(decl) = self.declarations.get_decl_for(&param.name.name) {
            if declaration_is_type(&decl) {
                // TODO it is a type
                unimplemented!();
            }
        } else {
            // it does not exist
            self.declarations
                .add_decl_for(&param.name.name, Declaration::StreamParameter(param));
        }

        // check the type
        if let Some(ty) = &param.ty {
            self.check_type(ty);
        }
    }

    pub(crate) fn check(&mut self, spec: &'a LolaSpec) {
        // Store global declarations, i.e., constants, inputs, and outputs of the given specification
        self.check_type_declarations(&spec);
        self.add_constants(&spec);
        self.add_inputs(&spec);
        self.add_outputs(&spec);
        self.check_outputs(&spec);
        self.check_triggers(&spec)
    }

    fn check_triggers(&mut self, spec: &'a LolaSpec) -> () {
        for trigger in &spec.trigger {
            self.declarations.push();
            // TODO
            self.check_expression(&trigger.expression);
            self.declarations.pop();
        }
    }

    fn check_outputs(&mut self, spec: &'a LolaSpec) {
        // recurse into expressions and check them
        for output in &spec.outputs {
            self.declarations.push();
            self.check_expression(&output.expression);
            output
                .params
                .iter()
                .for_each(|param| self.check_param(&param));
            self.declarations.pop();
        }
    }

    fn add_outputs(&mut self, spec: &'a LolaSpec) {
        for output in &spec.outputs {
            if let Some(_decl) = self.declarations.get_decl_for(&output.name.name) {
                // TODO error based on the existing declaration and the current one
                unimplemented!();
            } else {
                self.declarations
                    .add_decl_for(&output.name.name, output.into());
            }
            if let Some(ty) = &output.ty {
                self.check_type(ty);
            }
        }
    }

    fn add_inputs(&mut self, spec: &'a LolaSpec) {
        for input in &spec.inputs {
            if let Some(_decl) = self.declarations.get_decl_for(&input.name.name) {
                // TODO error based on the existing declaration and the current one
                unimplemented!();
            } else {
                self.declarations
                    .add_decl_for(&input.name.name, input.into());
            }
            // TODO check type
        }
    }

    fn add_constants(&mut self, spec: &'a LolaSpec) {
        for constant in &spec.constants {
            if let Some(_decl) = self.declarations.get_decl_for(&constant.name.name) {
                // TODO error based on the existing declaration and the current one
                unimplemented!();
            } else {
                self.declarations
                    .add_decl_for(&constant.name.name, constant.into());
            }
            // TODO check type
        }
    }

    fn check_type_declarations(&mut self, spec: &'a LolaSpec) {
        for type_decl in &spec.type_declarations {
            self.declarations.push();
            // check children
            type_decl.fields.iter().for_each(|field| {
                self.check_type(&field.ty);
            });

            type_decl.fields.iter().for_each(|_field| {
                // TODO check the names
            });

            self.declarations.pop();

            // only add the new type name after checking all fields
            if let Some(ref name) = &type_decl.name {
                if let Some(_decl) = self.declarations.get_decl_for(&name.name) {
                    // TODO error based on the existing declaration and the current one
                    unimplemented!();
                } else {
                    self.declarations
                        .add_decl_for(&name.name, Declaration::UserDefinedType(type_decl));
                }
            } else {
                // TODO error unnamed type declaration
            }
        }
    }

    fn check_stream_instance(&mut self, instance: &'a StreamInstance) {
        // TODO check that the look up target exists
        if let Some(decl) = self
            .declarations
            .get_decl_for(&instance.stream_identifier.name)
        {
            if declaration_is_type(&decl) {
                // TODO it is a type
                unimplemented!();
            }
        } else {
            // it does not exist
            unimplemented!()
        }
        // check paramterization
        instance.arguments.iter().for_each(|param| {
            self.check_expression(param);
        });
    }

    fn check_offset(&mut self, offset: &'a Offset) {
        match offset {
            DiscreteOffset(off) | RealTimeOffset(off, _) => self.check_expression(off),
        }
    }

    fn check_expression(&mut self, expression: &'a Expression) {
        use self::ExpressionKind::*;

        match &expression.kind {
            Ident(ident) => {
                if let Some(decl) = self.declarations.get_decl_for(&ident.name) {
                    assert!(*expression.id() != NodeId::DUMMY);
                    // TODO check that the declaration is not a type
                    if declaration_is_type(&decl) {
                        unimplemented!();
                    } else {
                        self.result.insert(*expression.id(), decl);
                    }
                } else {
                    // TODO unbounded variable, store error to display to user
                    unimplemented!();
                }
            }
            Binary(_, left, right) => {
                self.check_expression(left);
                self.check_expression(right);
            }
            Lit(_) | MissingExpression() => {}
            Ite(condition, if_case, else_case) => {
                self.check_expression(condition);
                self.check_expression(if_case);
                self.check_expression(else_case);
            }
            ParenthesizedExpression(_, expr, _) | Unary(_, expr) => {
                self.check_expression(expr);
            }
            Tuple(exprs) => {
                exprs.iter().for_each(|expr| self.check_expression(expr));
            }
            Function(_, exprs) => {
                exprs.iter().for_each(|expr| self.check_expression(expr));
            }
            Default(accessed, default) => {
                self.check_expression(accessed);
                self.check_expression(default);
            }
            Lookup(instance, offset, _) => {
                self.check_stream_instance(instance);
                self.check_offset(offset);
            }
        }
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
        // TODO check for double declaration
        // TODO check for keyword
        self.scopes.last_mut().unwrap().insert(name, decl);
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum Declaration<'a> {
    Const(&'a Constant),
    In(&'a Input),
    Out(&'a Output),
    UserDefinedType(&'a TypeDeclaration),
    BuiltinType(super::common::BuiltinType),
    StreamParameter(&'a Parameter),
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
    use crate::analysis::id_assignment;
    use crate::parse::parse;

    // TODO: implement test cases
    #[test]
    #[ignore]
    fn build_parameter_list() {
        let spec = "output s <a: B, c: D>: E := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
    }
}
