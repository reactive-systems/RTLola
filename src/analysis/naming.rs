//! This module provides naming analysis for a given Lola AST.

use super::super::ast::Offset::*;
use super::super::ast::*;
use super::AnalysisError;
use crate::analysis::*;
use ast_node::{AstNode, NodeId};
use std::collections::HashMap;

pub(crate) type DeclarationTable<'a> = HashMap<NodeId, Declaration<'a>>;

pub(crate) struct NamingAnalysis<'a> {
    declarations: ScopedDecl<'a>,
    pub result: DeclarationTable<'a>,
    pub errors: Vec<Box<AnalysisError<'a> + 'a>>,
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

fn declaration_is_lookup_target(decl: &Declaration) -> bool {
    match decl {
        Declaration::In(_) | Declaration::Out(_) => true,
        Declaration::Const(_)
        | Declaration::UserDefinedType(_)
        | Declaration::BuiltinType(_)
        | Declaration::StreamParameter(_) => false,
    }
}

fn is_malformed_name(_name: &str) -> bool {
    // TODO check for malformed
    false
}

impl<'a> NamingAnalysis<'a> {
    pub(crate) fn new() -> Self {
        let mut scoped_decls = ScopedDecl::new();

        for (name, ty) in super::common::BuiltinType::all() {
            scoped_decls.add_decl_for(name, Declaration::BuiltinType(ty));
        }

        NamingAnalysis {
            declarations: scoped_decls,
            result: HashMap::new(),
            errors: Vec::new(),
        }
    }

    fn add_decl_for(&mut self, name: &'a str, decl: Declaration<'a>, node: &'a AstNode<'a>) {
        // check for keyword
        let lower = name.to_lowercase();
        if common::KEYWORDS.contains(&lower.as_str()) {
            self.errors
                .push(Box::new(NamingError::ReservedKeyword(node)));
        }

        if is_malformed_name(name) {
            self.errors.push(Box::new(NamingError::MalformedName(node)));
        }
        self.declarations.add_decl_for(name, decl);
    }

    fn check_type(&mut self, ty: &'a Type) {
        match ty.kind {
            TypeKind::Simple(ref name) | TypeKind::Malformed(ref name) => {
                if let Some(decl) = self.declarations.get_decl_for(&name) {
                    if !declaration_is_type(&decl) {
                        self.errors.push(Box::new(NamingError::NotAType(ty)));
                    } else {
                        self.result.insert(*ty.id(), decl);
                    }
                } else {
                    // it does not exist
                    self.errors.push(Box::new(NamingError::TypeNotFound(ty)));
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
                self.errors
                    .push(Box::new(NamingError::TypeNotAllowedHere(param)));
            } else if let Some(decl) = self
                .declarations
                .get_decl_in_current_scope_for(&param.name.name)
            {
                self.errors.push(Box::new(NamingError::NameAlreadyUsed {
                    current: param,
                    previous: decl,
                }));
            }
        } else {
            // it does not exist
            self.add_decl_for(&param.name.name, Declaration::StreamParameter(param), param);
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
        let mut trigger_names: Vec<(&'a String, &'a Trigger)> = Vec::new();
        for trigger in &spec.trigger {
            self.declarations.push();
            if let Some(ident) = &trigger.name {
                if let Some(decl) = self.declarations.get_decl_for(&ident.name) {
                    self.errors.push(Box::new(NamingError::NameAlreadyUsed {
                        current: trigger,
                        previous: decl,
                    }));
                }
                let mut found = false;
                for previous_entry in trigger_names.iter() {
                    match previous_entry {
                        (ref name, ref previous_trigger) => {
                            if ident.name == **name {
                                found = true;
                                self.errors.push(Box::new(NamingError::TriggerWithSameName {
                                    current: trigger,
                                    previous: *previous_trigger,
                                }));
                                break;
                            }
                        }
                    }
                }
                if !found {
                    trigger_names.push((&ident.name, trigger))
                }
            }
            self.check_expression(&trigger.expression);
            self.declarations.pop();
        }
    }

    fn check_outputs(&mut self, spec: &'a LolaSpec) {
        // recurse into expressions and check them
        for output in &spec.outputs {
            self.declarations.push();
            output
                .params
                .iter()
                .for_each(|param| self.check_param(&param));
            self.check_expression(&output.expression);
            self.declarations.pop();
        }
    }

    fn add_outputs(&mut self, spec: &'a LolaSpec) {
        for output in &spec.outputs {
            if let Some(decl) = self.declarations.get_decl_for(&output.name.name) {
                self.errors.push(Box::new(NamingError::NameAlreadyUsed {
                    current: output,
                    previous: decl,
                }));
            } else {
                self.add_decl_for(&output.name.name, output.into(), output);
            }
            if let Some(ref ty) = &output.ty {
                self.check_type(ty);
            }
        }
    }

    fn add_inputs(&mut self, spec: &'a LolaSpec) {
        for input in &spec.inputs {
            if let Some(decl) = self.declarations.get_decl_for(&input.name.name) {
                self.errors.push(Box::new(NamingError::NameAlreadyUsed {
                    current: input,
                    previous: decl,
                }));
            } else {
                self.add_decl_for(&input.name.name, input.into(), input);
            }
            self.check_type(&input.ty);

            self.declarations.push();
            input
                .params
                .iter()
                .for_each(|param| self.check_param(&param));
            self.declarations.pop();
        }
    }

    fn add_constants(&mut self, spec: &'a LolaSpec) {
        for constant in &spec.constants {
            if let Some(decl) = self.declarations.get_decl_for(&constant.name.name) {
                self.errors.push(Box::new(NamingError::NameAlreadyUsed {
                    current: constant,
                    previous: decl,
                }));
            } else {
                self.add_decl_for(&constant.name.name, constant.into(), constant);
            }
            if let Some(ref ty) = constant.ty {
                self.check_type(&ty);
            }
        }
    }

    fn check_type_declarations(&mut self, spec: &'a LolaSpec) {
        for type_decl in &spec.type_declarations {
            self.declarations.push();
            // check children
            let mut field_names: Vec<(&'a String, &'a TypeDeclField)> = Vec::new();
            type_decl.fields.iter().for_each(|field| {
                self.check_type(&field.ty);
            });

            type_decl.fields.iter().for_each(|field| {
                let mut found = false;
                for previous_entry in field_names.iter() {
                    match previous_entry {
                        (ref name, ref previous_field) => {
                            if field.name == **name {
                                found = true;
                                self.errors.push(Box::new(NamingError::FieldWithSameName {
                                    current: &**field,
                                    previous: *previous_field,
                                }));
                                break;
                            }
                        }
                    }
                }
                if !found {
                    field_names.push((&field.name, &**field));
                    if is_malformed_name(&field.name) {
                        self.errors
                            .push(Box::new(NamingError::MalformedName(&**field)));
                    }
                }
            });

            self.declarations.pop();

            // only add the new type name after checking all fields
            if let Some(ref name) = &type_decl.name {
                if let Some(decl) = self.declarations.get_decl_for(&name.name) {
                    self.errors.push(Box::new(NamingError::NameAlreadyUsed {
                        current: type_decl,
                        previous: decl,
                    }));
                } else {
                    self.add_decl_for(
                        &name.name,
                        Declaration::UserDefinedType(type_decl),
                        type_decl,
                    );
                }
            } else {
                self.errors
                    .push(Box::new(NamingError::UnnamedTypeDeclaration(type_decl)));
            }
        }
    }

    fn check_stream_instance(&mut self, instance: &'a StreamInstance) {
        if let Some(decl) = self
            .declarations
            .get_decl_for(&instance.stream_identifier.name)
        {
            if !declaration_is_lookup_target(&decl) {
                self.errors
                    .push(Box::new(NamingError::NotAStream(instance)));
            } else {
                self.result.insert(*instance.id(), decl);
            }
        } else {
            // it does not exist
            self.errors
                .push(Box::new(NamingError::NameNotFound(instance)));
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
                    if declaration_is_type(&decl) {
                        self.errors
                            .push(Box::new(NamingError::TypeNotAllowedHere(expression)));
                    } else {
                        self.result.insert(*expression.id(), decl);
                    }
                } else {
                    self.errors
                        .push(Box::new(NamingError::NameNotFound(expression)))
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

    fn get_decl_in_current_scope_for(&self, name: &'a str) -> Option<Declaration<'a>> {
        match self.scopes.last().unwrap().get(name) {
            Some(&decl) => Some(decl),
            None => None,
        }
    }

    fn add_decl_for(&mut self, name: &'a str, decl: Declaration<'a>) {
        assert!(self.scopes.last().is_some());
        self.scopes.last_mut().unwrap().insert(name, decl);
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Declaration<'a> {
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
    fn unknown_types_are_reported() {
        let spec = "output test<ab: B, c: D>: E := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(3, naming_analyzer.errors.len());
    }

    #[test]
    fn unknown_identifiers_are_reported() {
        let spec = "output test: Int8 := A";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(1, naming_analyzer.errors.len());
    }

    #[test]
    fn int8_is_a_known_type() {
        let spec = "output test: Int8 := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(0, naming_analyzer.errors.len());
    }

    #[test]
    fn int16_is_a_known_type() {
        let spec = "output test: Int16 := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(0, naming_analyzer.errors.len());
    }

    #[test]
    fn int32_is_a_known_type() {
        let spec = "output test: Int32 := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(0, naming_analyzer.errors.len());
    }

    #[test]
    fn int64_is_a_known_type() {
        let spec = "output test: Int64 := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(0, naming_analyzer.errors.len());
    }

    #[test]
    fn float32_is_a_known_type() {
        let spec = "output test: Float32 := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(0, naming_analyzer.errors.len());
    }

    #[test]
    fn float64_is_a_known_type() {
        let spec = "output test: Float64 := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(0, naming_analyzer.errors.len());
    }

    #[test]
    fn bool_is_a_known_type() {
        let spec = "output test: Bool := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(0, naming_analyzer.errors.len());
    }

    #[test]
    fn string_is_a_known_type() {
        let spec = "output test: String := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(0, naming_analyzer.errors.len());
    }

    #[test]
    fn duplicate_names_at_the_same_level_are_reported() {
        let spec = "output test: String := 3\noutput test: String := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(1, naming_analyzer.errors.len());
    }

    #[test]
    fn streams_declared_type_is_in_the_result() {
        let spec = "output test: Int8 := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(1, naming_analyzer.result.len());
    }

    #[test]
    fn duplicate_parameters_are_not_allowed_for_outputs() {
        let spec = "output test<ab: Int8, ab: Int8> := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(1, naming_analyzer.errors.len());
    }

    #[test]
    fn duplicate_parameters_are_not_allowed_for_inputs() {
        let spec = "input test<ab: Int8, ab: Int8> : Int8";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(1, naming_analyzer.errors.len());
    }

    #[test]
    fn keyword_are_not_valid_names() {
        let spec = "output Int128 := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut naming_analyzer = NamingAnalysis::new();
        naming_analyzer.check(&mut ast);
        assert_eq!(1, naming_analyzer.errors.len());
    }
}
