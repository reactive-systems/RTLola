//! This module provides naming analysis for a given Lola AST.

use crate::ast::*;
use crate::parse::{Ident, NodeId, Span};
use crate::reporting::{Handler, LabeledSpan};
use crate::stdlib;
use crate::stdlib::FuncDecl;
use crate::ty::ValueTy;
use crate::FrontendConfig;
use std::{collections::HashMap, rc::Rc};

// These MUST all be lowercase
// TODO add an static assertion for this
pub(crate) const KEYWORDS: [&str; 26] = [
    "input",
    "output",
    "trigger",
    "import",
    "type",
    "self",
    "include",
    "invoke",
    "inv",
    "extend",
    "ext",
    "terminate",
    "ter",
    "unless",
    "if",
    "then",
    "else",
    "and",
    "or",
    "not",
    "forall",
    "exists",
    "any",
    "true",
    "false",
    "error",
];

pub(crate) type DeclarationTable = HashMap<NodeId, Declaration>;

pub(crate) struct NamingAnalysis<'b> {
    declarations: ScopedDecl,
    type_declarations: ScopedDecl,
    fun_declarations: ScopedDecl,
    result: DeclarationTable,
    handler: &'b Handler,
}

impl<'b> NamingAnalysis<'b> {
    pub(crate) fn new(handler: &'b Handler, config: FrontendConfig) -> Self {
        let mut scoped_decls = ScopedDecl::new();

        for (name, ty) in ValueTy::primitive_types(config.ty) {
            scoped_decls.add_decl_for(name, Declaration::Type(Rc::new(ty.clone())));
        }

        // add a new scope to distinguish between extern/builtin declarations
        scoped_decls.push();

        NamingAnalysis {
            declarations: ScopedDecl::new(),
            type_declarations: scoped_decls,
            fun_declarations: ScopedDecl::new(),
            result: HashMap::new(),
            handler,
        }
    }

    /// Adds a declaration to the declaration store.
    ///
    /// Checks if
    /// * name of declaration is a keyword
    /// * declaration already exists in current scope
    fn add_decl_for(&mut self, decl: Declaration) {
        assert!(!decl.is_type());
        let name = decl.get_name().expect("added declarations are guaranteed to have a name");

        let span = decl.get_span().expect("all user defined declarations have a `Span`");

        // check for keyword
        let lower = name.to_lowercase();
        if KEYWORDS.contains(&lower.as_str()) {
            self.handler.error_with_span(
                &format!("`{}` is a reserved keyword", name),
                LabeledSpan::new(span, "use a different name here", true),
            )
        }

        if let Some(decl) = self.declarations.get_decl_in_current_scope_for(name) {
            let mut builder = self.handler.build_error_with_span(
                &format!("the name `{}` is defined multiple times", name),
                LabeledSpan::new(span, &format!("`{}` redefined here", name), true),
            );
            if let Some(span) = decl.get_span() {
                builder.add_span_with_label(span, &format!("previous definition of the value `{}` here", name), false);
            }
            builder.emit();
        } else {
            self.declarations.add_decl_for(name, decl.clone());
        }
    }

    /// Checks if given type is bound
    fn check_type(&mut self, ty: &Type) {
        match &ty.kind {
            TypeKind::Simple(name) => {
                if let Some(decl) = self.type_declarations.get_decl_for(&name) {
                    assert!(decl.is_type());
                    self.result.insert(ty.id, decl);
                } else {
                    // it does not exist
                    self.handler.error_with_span(
                        &format!("cannot find type `{}` in this scope", name),
                        LabeledSpan::new(ty.span, "not found in this scope", true),
                    );
                }
            }
            TypeKind::Tuple(elements) => elements.iter().for_each(|ty| {
                self.check_type(ty);
            }),
            TypeKind::Optional(ty) => self.check_type(ty),
            TypeKind::Inferred => {}
        }
    }

    /// Checks that the parameter name and type are both valid
    fn check_param(&mut self, param: &Rc<Parameter>) {
        // check the name
        if let Some(decl) = self.declarations.get_decl_for(&param.name.name) {
            assert!(!decl.is_type());

            // check if there is a parameter with the same name
            if let Some(decl) = self.declarations.get_decl_in_current_scope_for(&param.name.name) {
                let mut builder = self.handler.build_error_with_span(
                    &format!("identifier `{}` is use more than once in this paramater list", param.name.name),
                    LabeledSpan::new(
                        param.name.span,
                        &format!("`{}` used as a parameter more than once", param.name.name),
                        true,
                    ),
                );
                builder.add_span_with_label(
                    decl.get_span().expect("as it is in parameter list, it has a span"),
                    &format!("previous use of the parameter `{}` here", param.name.name),
                    false,
                );
                builder.emit();
            }
        } else {
            // it does not exist
            self.add_decl_for(Declaration::Param(param.clone()));
        }

        // check the type
        self.check_type(&param.ty);
    }

    /// Entry method, checks that every identifier in the given spec is bound.
    pub(crate) fn check(&mut self, spec: &RTLolaAst) -> DeclarationTable {
        stdlib::import_implicit_module(&mut self.fun_declarations);
        for import in &spec.imports {
            match import.name.name.as_str() {
                "math" => stdlib::import_math_module(&mut self.fun_declarations),
                "regex" => stdlib::import_regex_module(&mut self.fun_declarations),
                n => self.handler.error_with_span(
                    &format!("unresolved import `{}`", n),
                    LabeledSpan::new(import.name.span, &format!("no `{}` in the root", n), true),
                ),
            }
        }

        // Store global declarations, i.e., constants, inputs, and outputs of the given specification
        for constant in &spec.constants {
            self.add_decl_for(Declaration::Const(constant.clone()));
            if let Some(ty) = constant.ty.as_ref() {
                self.check_type(ty)
            }
        }

        for input in &spec.inputs {
            self.add_decl_for(Declaration::In(input.clone()));
            self.check_type(&input.ty);

            // check types for parametric inputs
            self.declarations.push();
            input.params.iter().for_each(|param| self.check_param(param));
            self.declarations.pop();
        }

        for output in &spec.outputs {
            if output.params.is_empty() {
                self.add_decl_for(Declaration::Out(output.clone()));
            } else {
                self.add_decl_for(Declaration::ParamOut(output.clone()))
            }
            self.check_type(&output.ty);
        }

        self.check_outputs(&spec);
        self.check_triggers(&spec);

        self.result.clone()
    }

    /// Checks that if the trigger has a name, it is unique
    fn check_triggers(&mut self, spec: &RTLolaAst) {
        let mut trigger_names: Vec<(&String, &Trigger)> = Vec::new();
        for trigger in &spec.trigger {
            if let Some(ident) = &trigger.name {
                if let Some(decl) = self.declarations.get_decl_in_current_scope_for(&ident.name) {
                    let mut builder = self.handler.build_error_with_span(
                        &format!("the name `{}` is defined multiple times", ident.name),
                        LabeledSpan::new(ident.span, &format!("`{}` redefined here", ident.name), true),
                    );
                    if let Some(span) = decl.get_span() {
                        builder.add_span_with_label(
                            span,
                            &format!("previous definition of the value `{}` here", ident.name),
                            false,
                        );
                    }
                    builder.emit();
                }
                let mut found = false;
                for previous_entry in &trigger_names {
                    let (name, previous_trigger) = previous_entry;
                    if ident.name == **name {
                        found = true;
                        let mut builder = self.handler.build_error_with_span(
                            &format!("the trigger `{}` is defined multiple times", ident.name),
                            LabeledSpan::new(ident.span, &format!("`{}` redefined here", ident.name), true),
                        );
                        builder.add_span_with_label(
                            previous_trigger.span,
                            &format!("previous trigger definition `{}` here", ident.name),
                            false,
                        );
                        builder.emit();

                        break;
                    }
                }
                if !found {
                    trigger_names.push((&ident.name, trigger))
                }
            }
            self.declarations.push();
            self.check_expression(&trigger.expression);
            self.declarations.pop();
        }
    }

    fn check_outputs(&mut self, spec: &RTLolaAst) {
        // recurse into expressions and check them
        for output in &spec.outputs {
            self.declarations.push();
            output.params.iter().for_each(|param| self.check_param(&param));
            if let Some(ref template_spec) = output.template_spec {
                if let Some(ref invoke) = template_spec.inv {
                    self.check_expression(&invoke.target);
                    if let Some(ref cond) = invoke.condition {
                        self.check_expression(&cond);
                    }
                }
                if let Some(ref extend) = template_spec.ext {
                    self.check_expression(&extend.target);
                }
                if let Some(ref terminate) = template_spec.ter {
                    self.check_expression(&terminate.target);
                }
            }
            if let Some(expr) = output.extend.expr.as_ref() {
                self.check_expression(expr);
            }
            if let Some(expr) = output.termination.as_ref() {
                self.check_expression(expr);
            }
            self.declarations.add_decl_for("self", Declaration::Out(output.clone()));
            self.check_expression(&output.expression);
            self.declarations.pop();
        }
    }

    fn check_ident(&mut self, expression: &Expression, ident: &Ident) {
        if let Some(decl) = self.declarations.get_decl_for(&ident.name) {
            assert!(!decl.is_type());

            self.result.insert(expression.id, decl);
        } else {
            self.handler.error_with_span(
                &format!("name `{}` does not exist in current scope", &ident.name),
                LabeledSpan::new(ident.span, "does not exist", true),
            );
        }
    }

    fn check_function(&mut self, expression: &Expression, name: &FunctionName) {
        let str_repr = name.to_string();
        if let Some(decl) = self.fun_declarations.get_decl_for(str_repr.as_str()) {
            assert!(decl.is_function());

            self.result.insert(expression.id, decl);
        } else if let Some(Declaration::ParamOut(out)) = self.declarations.get_decl_for(&name.name.name) {
            // parametric outputs are represented as functions
            self.result.insert(expression.id, Declaration::ParamOut(out));
        } else {
            self.handler.error_with_span(
                &format!("function name `{}` does not exist in current scope", str_repr),
                LabeledSpan::new(name.name.span, "does not exist", true),
            );
        }
    }

    fn check_expression(&mut self, expression: &Expression) {
        use self::ExpressionKind::*;

        match &expression.kind {
            Ident(ident) => {
                self.check_ident(expression, ident);
            }
            StreamAccess(expr, _) => self.check_expression(expr),
            Offset(expr, _) => {
                self.check_expression(expr);
            }
            SlidingWindowAggregation { expr, duration, .. } => {
                self.check_expression(expr);
                self.check_expression(duration);
            }
            Binary(_, left, right) => {
                self.check_expression(left);
                self.check_expression(right);
            }
            Lit(_) | MissingExpression => {}
            Ite(condition, if_case, else_case) => {
                self.check_expression(condition);
                self.check_expression(if_case);
                self.check_expression(else_case);
            }
            ParenthesizedExpression(_, expr, _) | Unary(_, expr) | Field(expr, _) => {
                self.check_expression(expr);
            }
            Tuple(exprs) => {
                exprs.iter().for_each(|expr| self.check_expression(expr));
            }
            Function(name, types, exprs) => {
                self.check_function(expression, name);
                types.iter().for_each(|ty| self.check_type(ty));
                exprs.iter().for_each(|expr| self.check_expression(expr));
            }
            Default(accessed, default) => {
                self.check_expression(accessed);
                self.check_expression(default);
            }
            Method(expr, _, types, args) => {
                self.check_expression(expr);
                types.iter().for_each(|ty| self.check_type(ty));
                args.iter().for_each(|expr| self.check_expression(expr));
            }
        }
    }
}

/// Provides a mapping from `String` to `Declaration` and is able to handle different scopes.
pub(crate) struct ScopedDecl {
    scopes: Vec<HashMap<String, Declaration>>,
}

impl ScopedDecl {
    fn new() -> Self {
        ScopedDecl { scopes: vec![HashMap::new()] }
    }

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        assert!(self.scopes.len() > 1);
        self.scopes.pop();
    }

    fn get_decl_for(&self, name: &str) -> Option<Declaration> {
        for scope in self.scopes.iter().rev() {
            if let Some(decl) = scope.get(name) {
                return Some(decl.clone());
            }
        }
        None
    }

    fn get_decl_in_current_scope_for(&self, name: &str) -> Option<Declaration> {
        match self.scopes.last().expect("It appears that we popped the global context.").get(name) {
            Some(decl) => Some(decl.clone()),
            None => None,
        }
    }

    fn add_decl_for(&mut self, name: &str, decl: Declaration) {
        assert!(self.scopes.last().is_some());
        self.scopes.last_mut().expect("It appears that we popped the global context.").insert(name.to_string(), decl);
    }

    pub(crate) fn add_fun_decl(&mut self, fun: &FuncDecl) {
        assert!(self.scopes.last().is_some());
        let name = fun.name.to_string();
        self.scopes
            .last_mut()
            .expect("It appears that we popped the global context.")
            .insert(name, Declaration::Func(Rc::new(fun.clone())));
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Declaration {
    Const(Rc<Constant>),
    In(Rc<Input>),
    /// A non-parametric output
    Out(Rc<Output>),
    /// A paramertric output, internally represented as a function application
    ParamOut(Rc<Output>),
    Type(Rc<ValueTy>),
    Param(Rc<Parameter>),
    Func(Rc<FuncDecl>),
}

impl Declaration {
    fn get_span(&self) -> Option<Span> {
        match &self {
            Declaration::Const(constant) => Some(constant.name.span),
            Declaration::In(input) => Some(input.name.span),
            Declaration::Out(output) => Some(output.name.span),
            Declaration::ParamOut(output) => Some(output.name.span),
            Declaration::Param(p) => Some(p.name.span),
            Declaration::Type(_) | Declaration::Func(_) => None,
        }
    }

    fn get_name(&self) -> Option<&str> {
        match self {
            Declaration::Const(constant) => Some(&constant.name.name),
            Declaration::In(input) => Some(&input.name.name),
            Declaration::Out(output) => Some(&output.name.name),
            Declaration::ParamOut(output) => Some(&output.name.name),
            Declaration::Param(p) => Some(&p.name.name),
            Declaration::Type(_) | Declaration::Func(_) => None,
        }
    }

    fn is_type(&self) -> bool {
        match self {
            Declaration::Type(_) => true,
            Declaration::Const(_)
            | Declaration::In(_)
            | Declaration::Out(_)
            | Declaration::ParamOut(_)
            | Declaration::Param(_)
            | Declaration::Func(_) => false,
        }
    }

    fn is_function(&self) -> bool {
        match self {
            Declaration::Func(_) | Declaration::ParamOut(_) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::parse::{parse, SourceMapper};
    use std::path::PathBuf;

    /// Parses the content, runs naming analysis, and returns number of errors
    fn number_of_naming_errors(content: &str) -> usize {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let ast = parse(content, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        let mut naming_analyzer = NamingAnalysis::new(&handler, FrontendConfig::default());
        naming_analyzer.check(&ast);
        handler.emitted_errors()
    }

    #[test]
    fn unknown_types_are_reported() {
        assert_eq!(3, number_of_naming_errors("output test(ab: B, c: D): E := 3"))
    }

    #[test]
    fn unknown_identifiers_are_reported() {
        assert_eq!(1, number_of_naming_errors("output test: Int8 := A"))
    }

    #[test]
    fn primitive_types_are_a_known() {
        for ty in &["Int8", "Int16", "Int32", "Int64", "Float32", "Float64", "Bool", "String"] {
            assert_eq!(0, number_of_naming_errors(&format!("output test: {} := 3", ty)))
        }
    }

    #[test]
    fn duplicate_names_at_the_same_level_are_reported() {
        assert_eq!(1, number_of_naming_errors("output test: String := 3\noutput test: String := 3"))
    }

    #[test]
    fn duplicate_parameters_are_not_allowed_for_outputs() {
        assert_eq!(1, number_of_naming_errors("output test(ab: Int8, ab: Int8) := 3"))
    }

    #[test]
    fn duplicate_parameters_are_not_allowed_for_inputs() {
        assert_eq!(1, number_of_naming_errors("input test(ab: Int8, ab: Int8) : Int8"))
    }

    #[test]
    fn keyword_are_not_valid_names() {
        assert_eq!(1, number_of_naming_errors("output if := 3"))
    }

    #[test]
    fn template_spec_is_also_tested() {
        assert_eq!(1, number_of_naming_errors("output a {invoke b} := 3"))
    }

    #[test]
    fn self_is_allowed_in_output_expression() {
        assert_eq!(0, number_of_naming_errors("output a  := self[-1]"))
    }

    #[test]
    fn self_not_allowed_in_trigger_expression() {
        //assert_eq!(1, number_of_naming_errors("trigger a  := self[-1]"))
        assert_eq!(1, number_of_naming_errors("trigger self[-1]"))
    }

    #[test]
    fn unknown_import() {
        assert_eq!(1, number_of_naming_errors("import xzy"))
    }

    #[test]
    fn known_import() {
        assert_eq!(0, number_of_naming_errors("import math"))
    }

    #[test]
    fn unknown_function() {
        assert_eq!(1, number_of_naming_errors("output x: Float32 := sqrt(2)"))
    }

    #[test]
    fn wrong_arity_function() {
        let spec = "import math\noutput o: Float64 := cos()";
        assert_eq!(1, number_of_naming_errors(spec));
    }

    #[test]
    fn known_function_though_import() {
        assert_eq!(0, number_of_naming_errors("import math\noutput x: Float32 := sqrt(2)"))
    }

    #[test]
    fn missing_expression() {
        // should not produce an error as we want to be able to handle incomplete specs in analysis
        assert_eq!(0, number_of_naming_errors("input x: Bool\noutput y: Bool := \ntrigger (y || x)"))
    }

    #[test]
    fn parametric_output() {
        let spec = "output x (a: UInt8, b: Bool): Int8 := 1 output y := x(1, false)";
        assert_eq!(0, number_of_naming_errors(spec));
    }

    #[test]
    fn test_param_spec_wrong_parameters() {
        let spec = "input in(a: Int8, b: Int8): Int8\noutput x := in(1)";
        assert_eq!(1, number_of_naming_errors(spec));
    }
}
