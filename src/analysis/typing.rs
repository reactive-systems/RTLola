//! Implementation of the Hindley-Milner type inference through unification
//!
//! Relevant references:
//! * Introduction to Hindley-Milner type inference https://eli.thegreenplace.net/2018/type-inference/
//! * Unification algorithm https://eli.thegreenplace.net/2018/unification/
//! * Unification in Rust http://smallcultfollowing.com/babysteps/blog/2017/03/25/unification-in-chalk-part-1/
//! * Ena (union-find package) https://crates.io/crates/ena

use super::naming::{Declaration, DeclarationTable};
use crate::ast::LolaSpec;
use crate::ast::{
    Constant, Expression, ExpressionKind, Input, LitKind, Literal, Offset, Output, Trigger, Type,
    TypeKind,
};
use crate::reporting::Handler;
use crate::reporting::LabeledSpan;
use crate::stdlib::{FuncDecl, MethodLookup};
use crate::ty::{Ty, TypeConstraint};
use ast_node::NodeId;
use ast_node::Span;
use ena::unify::{EqUnifyValue, InPlaceUnificationTable, UnificationTable, UnifyKey, UnifyValue};
use log::{debug, trace};
use std::collections::HashMap;

pub(crate) struct TypeAnalysis<'a> {
    handler: &'a Handler,
    declarations: &'a DeclarationTable<'a>,
    method_lookup: MethodLookup,
    unifier: Unifier,
    /// maps `NodeId`'s to the variables used in `unifier`
    var_lookup: HashMap<NodeId, InferVar>,
}

impl<'a> TypeAnalysis<'a> {
    pub(crate) fn new(
        handler: &'a Handler,
        declarations: &'a DeclarationTable<'a>,
    ) -> TypeAnalysis<'a> {
        TypeAnalysis {
            handler,
            declarations,
            method_lookup: MethodLookup::new(),
            unifier: Unifier::new(),
            var_lookup: HashMap::new(),
        }
    }

    pub(crate) fn check(&mut self, spec: &'a LolaSpec) {
        self.infer_types(spec);
        if self.handler.contains_error() {
            return;
        }

        self.assign_types(spec);
    }

    fn infer_types(&mut self, spec: &'a LolaSpec) {
        trace!("infer types");
        for constant in &spec.constants {
            self.infer_constant(constant).unwrap_or_else(|_| {
                debug!("type inference failed for {}", constant);
            });
        }

        for input in &spec.inputs {
            self.infer_input(input).unwrap_or_else(|_| {
                debug!("type inference failed for {}", input);
            });
        }

        for output in &spec.outputs {
            self.infer_output(output).unwrap_or_else(|_| {
                debug!("type inference failed for {}", output);
            });
        }

        for output in &spec.outputs {
            self.infer_output_expression(output).unwrap_or_else(|_| {
                debug!("type inference failed for {}", output);
            });
        }

        for trigger in &spec.trigger {
            self.infer_trigger_expression(trigger).unwrap_or_else(|_| {
                debug!("type inference failed for {}", trigger);
            });
        }
    }

    fn new_var(&mut self, node: NodeId) -> InferVar {
        assert!(!self.var_lookup.contains_key(&node));
        let var = self.unifier.new_var();
        self.var_lookup.insert(node, var);
        var
    }

    fn infer_constant(&mut self, constant: &'a Constant) -> Result<(), ()> {
        trace!("infer type for {}", constant);
        let var = self.new_var(constant._id);

        if let Some(ast_ty) = &constant.ty {
            self.infer_type(&ast_ty)?;
            let ty_var = self.var_lookup[&ast_ty._id];
            self.unifier
                .unify_var_ty(var, Ty::EventStream(Ty::Infer(ty_var).into(), vec![]))
                .expect("cannot fail as `var` is a fresh var");
        }

        // generate constraint from literal
        self.unifier
            .unify_var_ty(
                var,
                Ty::EventStream(
                    self.get_constraint_for_literal(&constant.literal).into(),
                    vec![],
                ),
            )
            .map_err(|err| self.handle_error(err, constant.literal._span))?;
        Ok(())
    }

    fn infer_input(&mut self, input: &'a Input) -> Result<(), ()> {
        trace!("infer type for {}", input);
        let var = self.new_var(input._id);

        self.infer_type(&input.ty)?;
        let ty_var = self.var_lookup[&input.ty._id];

        let mut param_types = Vec::new();
        for param in &input.params {
            self.infer_type(&param.ty)?;
            let param_ty_var = self.var_lookup[&param.ty._id];
            let param_var = self.new_var(param._id);
            self.unifier
                .unify_var_var(param_var, param_ty_var)
                .expect("cannot fail as `param_var` is fresh");
            param_types.push(Ty::Infer(param_var));
        }

        // ?var = EventStream<?ty_var, <?param_vars>>
        self.unifier
            .unify_var_ty(
                var,
                Ty::EventStream(Box::new(Ty::Infer(ty_var)), param_types),
            )
            .map_err(|err| self.handle_error(err, input.name.span))
    }

    fn infer_output(&mut self, output: &'a Output) -> Result<(), ()> {
        trace!("infer type for {}", output);
        let var = self.new_var(output._id);

        // generate constraint in case a type is annotated
        self.infer_type(&output.ty)?;
        let ty_var = self.var_lookup[&output.ty._id];

        let mut param_types = Vec::new();
        for param in &output.params {
            self.infer_type(&param.ty)?;
            let param_ty_var = self.var_lookup[&param.ty._id];
            let param_var = self.new_var(param._id);
            self.unifier
                .unify_var_var(param_var, param_ty_var)
                .expect("cannot fail as `param_var` is fresh");
            param_types.push(Ty::Infer(param_var));
        }

        // ?var = EventStream<?ty_var, <?param_vars>>
        self.unifier
            .unify_var_ty(
                var,
                Ty::EventStream(Box::new(Ty::Infer(ty_var)), param_types),
            )
            .map_err(|err| self.handle_error(err, output.name.span))
    }

    fn infer_type(&mut self, ast_ty: &'a Type) -> Result<(), ()> {
        trace!("infer type for {}", ast_ty);
        let ty_var = self.new_var(ast_ty._id);
        match &ast_ty.kind {
            TypeKind::Inferred => Ok(()),
            TypeKind::Simple(_) => {
                match self.declarations[&ast_ty._id] {
                    Declaration::Type(ty) => {
                        // ?ty_var = `ty`
                        self.unifier
                            .unify_var_ty(ty_var, (*ty).clone())
                            .map_err(|err| self.handle_error(err, ast_ty._span))
                    }
                    _ => unreachable!(),
                }
            }
            TypeKind::Tuple(tuple) => {
                let ty = self.get_tuple_type(tuple);
                // ?ty_var = `ty`
                self.unifier
                    .unify_var_ty(ty_var, ty)
                    .map_err(|err| self.handle_error(err, ast_ty._span))
            }
            &TypeKind::Duration(val, unit) => {
                let ty = Ty::new_duration(val, unit);
                // ?ty_var = `ty`
                self.unifier
                    .unify_var_ty(ty_var, ty)
                    .map_err(|err| self.handle_error(err, ast_ty._span))
            }
            _ => unreachable!(),
        }
    }

    fn infer_output_expression(&mut self, output: &'a Output) -> Result<(), ()> {
        trace!("infer type for {}", output);
        let out_var = self.var_lookup[&output._id];

        // check template specification
        if let Some(template_spec) = &output.template_spec {
            if let Some(invoke) = &template_spec.inv {
                self.infer_expression(&invoke.target, None)?;
                let inv_var = self.var_lookup[&invoke.target._id];
                if output.params.len() == 1 {
                    // ?param_var = ?inv_bar
                    let param_var = self.var_lookup[&output.params[0]._id];
                    self.unifier
                        .unify_var_ty(
                            inv_var,
                            Ty::EventStream(Ty::Infer(param_var).into(), vec![]),
                        )
                        .map_err(|err| self.handle_error(err, invoke.target._span))?;
                } else {
                    let target_ty = Ty::Tuple(
                        output
                            .params
                            .iter()
                            .map(|p| {
                                let param_var = self.var_lookup[&p._id];
                                Ty::Infer(param_var)
                            })
                            .collect(),
                    );
                    self.unifier
                        .unify_var_ty(inv_var, Ty::EventStream(target_ty.into(), vec![]))
                        .map_err(|err| self.handle_error(err, invoke.target._span))?;
                }

                // check that condition is boolean
                if let Some(cond) = &invoke.condition {
                    self.infer_expression(cond, Some(Ty::EventStream(Ty::Bool.into(), vec![])))?;
                }
            }
            if let Some(extend) = &template_spec.ext {
                // check that condition is boolean
                if let Some(cond) = &extend.target {
                    self.infer_expression(cond, Some(Ty::EventStream(Ty::Bool.into(), vec![])))?;
                }
            }
            if let Some(terminate) = &template_spec.ter {
                // check that condition is boolean
                self.infer_expression(
                    &terminate.target,
                    Some(Ty::EventStream(Ty::Bool.into(), vec![])),
                )?;
            }
        }

        // generate constraint for expression
        self.infer_expression(&output.expression, Some(Ty::Infer(out_var)))
    }

    fn infer_trigger_expression(&mut self, trigger: &'a Trigger) -> Result<(), ()> {
        trace!("infer type for {}", trigger);
        self.infer_expression(&trigger.expression, Some(Ty::Bool))
    }

    fn get_constraint_for_literal(&self, lit: &Literal) -> Ty {
        use crate::ast::LitKind::*;
        match lit.kind {
            Str(_) | RawStr(_) => Ty::String,
            Bool(_) => Ty::Bool,
            Int(i) => Ty::Constr(if i < 0 {
                TypeConstraint::SignedInteger
            } else {
                TypeConstraint::Integer
            }),
            Float(_) => Ty::Constr(TypeConstraint::FloatingPoint),
        }
    }

    fn infer_expression(&mut self, expr: &'a Expression, target: Option<Ty>) -> Result<(), ()> {
        let var = self.new_var(expr._id);
        if let Some(target_ty) = &target {
            self.unifier
                .unify_var_ty(var, target_ty.clone())
                .expect("unification cannot fail as `var` is fresh");
        }

        use crate::ast::ExpressionKind::*;
        match &expr.kind {
            Lit(l) => {
                // generate constraint from literal
                self.unifier
                    .unify_var_ty(
                        var,
                        Ty::EventStream(self.get_constraint_for_literal(&l).into(), vec![]),
                    )
                    .map_err(|err| self.handle_error(err, expr._span))?;
            }
            Ident(_) => {
                let decl = self.declarations[&expr._id];

                match decl {
                    Declaration::Const(constant) => {
                        let const_var = self.var_lookup[&constant._id];
                        self.unifier
                            .unify_var_var(var, const_var)
                            .map_err(|err| self.handle_error(err, expr._span))?;
                    }
                    Declaration::In(input) => {
                        let in_var = self.var_lookup[&input._id];
                        self.unifier
                            .unify_var_var(var, in_var)
                            .map_err(|err| self.handle_error(err, expr._span))?;
                    }
                    Declaration::Out(output) => {
                        let out_var = self.var_lookup[&output._id];
                        self.unifier
                            .unify_var_var(var, out_var)
                            .map_err(|err| self.handle_error(err, expr._span))?;
                    }
                    _ => unreachable!(),
                }
            }
            Ite(cond, left, right) => {
                // constraints
                // - ?cond = EventStream<bool>
                // - ?left = ?expr
                // - ?right = ?expr
                self.infer_expression(cond, Some(Ty::EventStream(Ty::Bool.into(), vec![])))?;
                self.infer_expression(left, Some(Ty::Infer(var)))?;
                self.infer_expression(right, Some(Ty::Infer(var)))?;
            }
            Unary(op, appl) => {
                self.infer_function_application(
                    var,
                    expr._span,
                    &op.get_func_decl(),
                    &[],
                    &[appl],
                )?;
            }
            Binary(op, left, right) => {
                self.infer_function_application(
                    var,
                    expr._span,
                    &op.get_func_decl(),
                    &[],
                    &[left, right],
                )?;
            }
            // discrete offset
            Lookup(stream, Offset::DiscreteOffset(off_expr), None) => {
                // recursion
                self.infer_expression(off_expr, Some(Ty::Constr(TypeConstraint::Integer)))?;

                let negative_offset = match &off_expr.kind {
                    ExpressionKind::Lit(l) => match l.kind {
                        LitKind::Int(i) => i < 0,
                        _ => unreachable!("offset expressions have to be integers"),
                    },
                    _ => unreachable!("offset expressions have to be literal"),
                };

                let inner_var = self.unifier.new_var();

                let mut params = Vec::with_capacity(stream.arguments.len());
                for arg in &stream.arguments {
                    self.infer_expression(arg, None)?;
                    let inner = match self.unifier.get_type(self.var_lookup[&arg._id]) {
                        Some(Ty::EventStream(inner, _)) => inner,
                        _ => unreachable!(),
                    };
                    params.push(*inner);
                }
                let target = Ty::EventStream(Box::new(Ty::Infer(inner_var)), params);

                // constraints
                // off_expr = {integer}
                // stream = EventStream<?1>
                // if negative_offset || parametric { expr = EventStream<Option<?1>> } else { expr = EventStream<?1> }

                // need to derive "inner" type `?1`
                let decl = self.declarations[&stream._id];
                let decl_var: InferVar = match decl {
                    Declaration::In(input) => self.var_lookup[&input._id],
                    Declaration::Out(output) => self.var_lookup[&output._id],
                    //Declaration::Const(constant) => self.var_lookup[&constant.ty._id],
                    _ => unreachable!(),
                };
                self.unifier.unify_var_ty(decl_var, target).map_err(|err| {
                    self.handle_error(err, stream._span);
                })?;
                if negative_offset || !stream.arguments.is_empty() {
                    self.unifier
                        .unify_var_ty(
                            var,
                            Ty::EventStream(Ty::Option(Ty::Infer(inner_var).into()).into(), vec![]),
                        )
                        .map_err(|err| {
                            self.handle_error(err, expr._span);
                        })?;
                } else {
                    self.unifier
                        .unify_var_ty(var, Ty::EventStream(Ty::Infer(inner_var).into(), vec![]))
                        .map_err(|err| {
                            self.handle_error(err, expr._span);
                        })?;
                }
            }
            Default(left, right) => {
                // constraints
                // expr = EventStream<new_var>
                // left = EventStream<Option<new_var>>
                // right = EventStream<new_var>
                let new_var = self.unifier.new_var();
                self.unifier
                    .unify_var_ty(var, Ty::EventStream(Ty::Infer(new_var).into(), vec![]))
                    .map_err(|err| {
                        self.handle_error(err, expr._span);
                    })?;
                self.infer_expression(
                    left,
                    Some(Ty::EventStream(
                        Ty::Option(Box::new(Ty::Infer(new_var))).into(),
                        vec![],
                    )),
                )?;
                self.infer_expression(
                    right,
                    Some(Ty::EventStream(Ty::Infer(new_var).into(), vec![])),
                )?;
            }
            Function(_, types, params) => {
                let decl = self.declarations[&expr._id];
                let fun_decl = match decl {
                    Declaration::Func(fun_decl) => fun_decl,
                    _ => unreachable!("expected function declaration"),
                };
                assert_eq!(params.len(), fun_decl.parameters.len());

                println!("{:?}", fun_decl);

                for ty in types {
                    self.infer_type(ty)?;
                }

                let params: Vec<&Expression> = params.iter().map(|e| e.as_ref()).collect();

                self.infer_function_application(
                    var,
                    expr._span,
                    fun_decl,
                    types.as_slice(),
                    params.as_slice(),
                )?;
            }
            Method(base, name, types, params) => {
                // recursion
                self.infer_expression(base, None)?;

                if let Some(infered) = self.unifier.get_type(self.var_lookup[&base._id]) {
                    debug!("{} {}", base, infered);
                    if let Some(fun_decl) = self.method_lookup.get(&infered, name.name.as_str()) {
                        debug!("{:?}", fun_decl);

                        for ty in types {
                            self.infer_type(ty)?;
                        }

                        let mut parameters = vec![base.as_ref()];
                        parameters.extend(params.iter().map(|e| e.as_ref()));

                        self.infer_function_application(
                            var,
                            expr._span,
                            &fun_decl,
                            types.as_slice(),
                            parameters.as_slice(),
                        )?;
                    } else {
                        panic!("could not find `{}`", name);
                    }
                } else {
                    panic!("could not get type of `{}`", base);
                }
            }
            Tuple(expressions) => {
                // recursion
                let mut tuples: Vec<Ty> = Vec::with_capacity(expressions.len());
                for element in expressions {
                    self.infer_expression(element, None)?;
                    let inner = match self.unifier.get_type(self.var_lookup[&element._id]) {
                        Some(Ty::EventStream(inner, _)) => inner,
                        _ => unreachable!(),
                    };
                    tuples.push(*inner);
                }
                // ?var = Tuple(?expr1, ?expr2, ..)
                self.unifier
                    .unify_var_ty(var, Ty::EventStream(Ty::Tuple(tuples).into(), vec![]))
                    .map_err(|err| self.handle_error(err, expr._span))?;
            }
            Field(base, ident) => {
                // recursion
                self.infer_expression(base, None)?;

                let infered = match self.unifier.get_type(self.var_lookup[&base._id]) {
                    Some(Ty::EventStream(inner, _)) => self.unifier.normalize_ty(*inner),
                    _ => unreachable!(),
                };

                debug!("{} {}", base, infered);
                match infered {
                    Ty::Tuple(inner) => {
                        let num: usize = ident
                            .name
                            .parse::<usize>()
                            .expect("verify that this is checked earlier");
                        if num >= inner.len() {
                            self.handler.error_with_span(
                                &format!("Try to access tuple at position {}", num),
                                LabeledSpan::new(ident.span, "", true),
                            );
                            return Err(());
                        }
                        // ?var = inner[num]
                        self.unifier
                            .unify_var_ty(var, Ty::EventStream(inner[num].clone().into(), vec![]))
                            .map_err(|err| self.handle_error(err, expr._span))?;
                    }
                    _ => {
                        self.handler.error_with_span(
                            &format!("Type `{}` has no field `{}`", infered, ident.name),
                            LabeledSpan::new(ident.span, "unknown field", true),
                        );
                        return Err(());
                    }
                }
            }
            ParenthesizedExpression(_, expr, _) => self.infer_expression(expr, target)?,
            t => unimplemented!("expression `{:?}`", t),
        }
        Ok(())
    }

    fn infer_function_application(
        &mut self,
        var: InferVar,
        span: Span,
        fun_decl: &FuncDecl,
        types: &[Type],
        params: &[&'a Expression],
    ) -> Result<(), ()> {
        assert!(types.len() <= fun_decl.generics.len());
        assert_eq!(params.len(), fun_decl.parameters.len());

        // build symbolic names for generic arguments
        let generics: Vec<InferVar> = fun_decl
            .generics
            .iter()
            .map(|gen| {
                let var = self.unifier.new_var();
                match &gen.constraint {
                    Ty::Constr(_) => {}
                    _ => unreachable!("currently, only constraints are allowed for generic types"),
                }
                self.unifier
                    .unify_var_ty(var, gen.constraint.clone())
                    .expect("cannot fail as var is freshly created");
                var
            })
            .collect();
        for (provided_type, &generic) in types.iter().zip(&generics) {
            println!("{} {}", provided_type, generic);
            self.unifier
                .unify_var_var(generic, self.var_lookup[&provided_type._id])
                .map_err(|err| self.handle_error(err, provided_type._span))?;
        }

        for (type_param, parameter) in fun_decl.parameters.iter().zip(params) {
            let ty = type_param.replace_params(&generics);
            if let Some(&param_var) = self.var_lookup.get(&parameter._id) {
                // for method calls, we have to infer type for first argument
                // ?param_var = `ty`
                self.unifier
                    .unify_var_ty(param_var, ty)
                    .map_err(|err| self.handle_error(err, parameter._span))?;
            } else {
                // otherwise, we have to check it now
                self.infer_expression(parameter, Some(ty))?;
            }
        }
        // return type
        let ty = fun_decl.return_type.replace_params(&generics);
        // ?param_var = `ty`
        self.unifier
            .unify_var_ty(var, ty)
            .map_err(|err| self.handle_error(err, span))?;
        Ok(())
    }

    fn get_tuple_type(&mut self, tuple: &[Type]) -> Ty {
        let mut inner = Vec::new();
        for t in tuple {
            match &t.kind {
                TypeKind::Simple(_) => {
                    let ty = self.declarations[&t._id];
                    match ty {
                        Declaration::Type(ty) => inner.push(ty.clone()),
                        _ => unreachable!(),
                    }
                }
                TypeKind::Tuple(types) => inner.push(self.get_tuple_type(&types)),
                _ => unreachable!(),
            }
        }
        Ty::Tuple(inner)
    }

    /// Assigns types as infered
    fn assign_types(&mut self, spec: &LolaSpec) {
        for constant in &spec.constants {
            debug!(
                "{} has type {}",
                constant,
                self.unifier.get_final_type(self.var_lookup[&constant._id])
            );
        }
        for input in &spec.inputs {
            debug!(
                "{} has type {}",
                input,
                self.unifier.get_final_type(self.var_lookup[&input._id])
            );
        }
        for output in &spec.outputs {
            debug!(
                "{} has type {}",
                output,
                self.unifier.get_final_type(self.var_lookup[&output._id])
            );
        }
    }

    fn handle_error(&mut self, mut err: InferError, span: Span) {
        err.normalize_types(&mut self.unifier);
        match err {
            InferError::TypeMismatch(ty_l, ty_r) => {
                self.handler.error_with_span(
                    &format!("Type mismatch between `{}` and `{}`", ty_l, ty_r),
                    LabeledSpan::new(
                        span,
                        &format!("expected `{}`, found `{}`", ty_l, ty_r),
                        true,
                    ),
                );
            }
            InferError::ConflictingConstraint(left, right) => {
                self.handler.error_with_span(
                    &format!("Conflicting constraints `{}` and `{}`", left, right),
                    LabeledSpan::new(
                        span,
                        &format!("no concrete type satisfies `{}` and `{}`", left, right),
                        true,
                    ),
                );
            }
            InferError::CyclicDependency(_, _) => {
                self.handler.error_with_span(
                    "Cannot infer type",
                    LabeledSpan::new(span, "consider using a type annotation", true),
                );
            }
        }
    }

    pub(crate) fn get_type(&mut self, id: NodeId) -> Ty {
        if let Some(&var) = self.var_lookup.get(&id) {
            self.unifier.get_final_type(var)
        } else {
            Ty::Error
        }
    }
}

/// Main data structure for the type unfication.
/// Implemented using a union-find data structure where the keys (`InferVar`)
/// represent type variables. Those keys have associated values (`InferVarVal`)
/// that represent whether the type variable is unbounded (`InferVarVal::Unkown`)
/// or bounded (`InferVarVal::Know(ty)` where `ty` has type `Ty`).
struct Unifier {
    /// union-find data structure representing the current state of the unification
    table: InPlaceUnificationTable<InferVar>,
}

impl Unifier {
    fn new() -> Unifier {
        Unifier {
            table: UnificationTable::new(),
        }
    }

    fn new_var(&mut self) -> InferVar {
        self.table.new_key(InferVarVal::Unknown)
    }

    /// Unifies two variables.
    /// Cannot fail if one of them is unbounded.
    /// If both are bounded, we try to unify their types (recursively over the `Ty` type).
    /// If this fails as well, we try to coerce them, i.e., transform one type into the other.
    fn unify_var_var(&mut self, left: InferVar, right: InferVar) -> InferResult {
        debug!("unify var var {} {}", left, right);
        if let (InferVarVal::Known(ty_l), InferVarVal::Known(ty_r)) =
            (self.table.probe_value(left), self.table.probe_value(right))
        {
            // if both variables have values, we try to unify them recursively
            if self.types_equal_rec(&ty_l, &ty_r) {
                // proceed with unification
            } else if self.types_coerce(&ty_l, &ty_r) {
                return Ok(());
            } else {
                return Err(InferError::TypeMismatch(ty_l, ty_r));
            }
        }
        self.table.unify_var_var(left, right)
    }

    /// Unifies a variable with a type.
    /// Cannot fail if the variable is unbounded.
    /// Prevents infinite recursion by checking if `var` appears in `ty`.
    /// Uses the same strategy to merge types as `unify_var_var` (in case `var` is bounded).
    fn unify_var_ty(&mut self, var: InferVar, ty: Ty) -> InferResult {
        debug!("unify var ty {} {}", var, ty);
        if let Ty::Infer(other) = ty {
            return self.unify_var_var(var, other);
        }
        if let Ty::Infer(_) = &ty {
            panic!("internal error: entered unreachable code")
        }
        if self.check_occurrence(var, &ty) {
            return Err(InferError::CyclicDependency(var, ty));
        }
        if let InferVarVal::Known(val) = self.table.probe_value(var) {
            if self.types_equal_rec(&val, &ty) {
                //self.table.unify_var_value(var, InferVarVal::Known(ty))
                return Ok(());
            } else if self.types_coerce(&val, &ty) {
                return Ok(());
            } else {
                Err(InferError::TypeMismatch(val, ty))
            }
        } else {
            self.table.unify_var_value(var, InferVarVal::Known(ty))
        }
    }

    /// Checks if `var` occurs in `ty`
    fn check_occurrence(&mut self, var: InferVar, ty: &Ty) -> bool {
        trace!("check occurrence {} {}", var, ty);
        match ty {
            Ty::Infer(t) => self.table.unioned(var, *t),
            Ty::EventStream(ty, params) => {
                self.check_occurrence(var, &ty)
                    || params.iter().any(|el| self.check_occurrence(var, el))
            }
            Ty::TimedStream(_) => unimplemented!(),
            Ty::Tuple(t) => t.iter().any(|e| self.check_occurrence(var, e)),
            _ => false,
        }
    }

    /// Checks recursively if types are equal. Tries to unify type parameters if possible.
    /// Note: If you change this function, you have to change `Ty::unify` as well.
    fn types_equal_rec(&mut self, left: &Ty, right: &Ty) -> bool {
        trace!("comp {} {}", left, right);
        match (left, right) {
            (&Ty::Infer(l), &Ty::Infer(r)) => {
                if self.table.unioned(l, r) {
                    true
                } else {
                    // try to unify values
                    self.unify_var_var(l, r).is_ok()
                }
            }
            (&Ty::Infer(var), ty) => {
                // try to unify
                self.unify_var_ty(var, ty.clone()).is_ok()
            }
            (ty, &Ty::Infer(var)) => {
                // try to unify
                self.unify_var_ty(var, ty.clone()).is_ok()
            }
            (Ty::Constr(constr_l), &Ty::Constr(constr_r)) => {
                constr_l.conjunction(constr_r).is_some()
            }
            (&Ty::Constr(constr), other) => other.satisfies(constr),
            (other, &Ty::Constr(constr)) => other.satisfies(constr),
            (Ty::Option(l), Ty::Option(r)) => self.types_equal_rec(l, r),
            (Ty::EventStream(l, param_l), Ty::EventStream(r, param_r)) => {
                self.types_equal_rec(l, r)
                    && param_l
                        .iter()
                        .zip(param_r)
                        .all(|(el_l, el_r)| self.types_equal_rec(el_l, el_r))
            }
            (Ty::Tuple(l), Ty::Tuple(r)) => {
                if l.len() != r.len() {
                    return false;
                }
                l.iter()
                    .zip(r)
                    .all(|(e_l, e_r)| self.types_equal_rec(e_l, e_r))
            }
            (Ty::TimedStream(_), Ty::TimedStream(_)) => unimplemented!(),
            (Ty::Window(ty_l, d_l), Ty::Window(ty_r, d_r)) => {
                self.types_equal_rec(ty_l, ty_r) && self.types_equal_rec(d_l, d_r)
            }
            (l, r) => l == r,
        }
    }

    /// Checks recursively if the `right` type can be transformed to match `left`.
    fn types_coerce(&mut self, left: &Ty, right: &Ty) -> bool {
        debug!("coerce {} {}", left, right);
        // types_equal_rec has side effects, thus, create snapshot before
        let snapshot = self.table.snapshot();
        let res = match right {
            Ty::EventStream(ty, params) if params.is_empty() => self.types_equal_rec(left, ty),
            Ty::Int(lower) => match left {
                Ty::Int(upper) => lower <= upper,
                _ => false,
            },
            Ty::UInt(lower) => match left {
                Ty::UInt(upper) => lower <= upper,
                _ => false,
            },
            Ty::Float(lower) => match left {
                Ty::Float(upper) => lower <= upper,
                _ => false,
            },
            _ => false,
        };
        if !res {
            self.table.rollback_to(snapshot);
        } else {
            self.table.commit(snapshot);
        }
        res
    }

    /// Returns type where every inference variable is substituted.
    /// If a constraint remains, it gets replaced by the default value (e.g., `Int32` for `SignedInteger`).
    /// If an infer variable remains, it is replaced by `Ty::Error`.
    fn get_final_type(&mut self, var: InferVar) -> Ty {
        use self::InferVarVal::*;
        match self.table.probe_value(var) {
            Unknown => Ty::Error,
            Known(Ty::Infer(other)) => self.get_final_type(other),
            Known(Ty::Constr(constr)) => {
                if let Some(ty) = constr.has_default() {
                    // TODO: emit warning that default type has been used
                    ty
                } else {
                    Ty::Error
                }
            }
            Known(Ty::EventStream(ty, params)) => {
                let inner = match *ty {
                    Ty::Infer(other) => self.get_final_type(other),
                    ref t => t.clone(),
                };
                let params = params
                    .into_iter()
                    .map(|param| match param {
                        Ty::Infer(other) => self.get_final_type(other),
                        t => t,
                    })
                    .collect();
                Ty::EventStream(Box::new(inner), params)
            }
            Known(t) => t.clone(),
        }
    }

    /// Returns current value of inference variable if it exists, `None` otherwise.
    fn get_type(&mut self, var: InferVar) -> Option<Ty> {
        if let InferVarVal::Known(ty) = self.table.probe_value(var) {
            Some(ty)
        } else {
            None
        }
    }

    /// Tries to remove variables used for inference by their values.
    /// Useful for example in error messages.
    fn normalize_ty(&mut self, ty: Ty) -> Ty {
        match ty {
            Ty::Infer(var) => match self.get_type(var) {
                None => ty,
                Some(other_ty) => self.normalize_ty(other_ty),
            },
            Ty::Tuple(t) => Ty::Tuple(t.into_iter().map(|el| self.normalize_ty(el)).collect()),
            Ty::EventStream(ty, params) => Ty::EventStream(
                Box::new(self.normalize_ty(*ty)),
                params.into_iter().map(|e| self.normalize_ty(e)).collect(),
            ),
            Ty::Option(ty) => Ty::Option(Box::new(self.normalize_ty(*ty))),
            Ty::Window(ty, d) => Ty::Window(
                Box::new(self.normalize_ty(*ty)),
                Box::new(self.normalize_ty(*d)),
            ),
            _ if ty.is_primitive() => ty,
            Ty::Constr(_) => ty,
            Ty::Param(_, _) => ty,
            Ty::Duration(_) => ty,
            _ => unreachable!("cannot normalize {}", ty),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct InferVar(u32);

// used in UnifcationTable
impl UnifyKey for InferVar {
    type Value = InferVarVal;
    fn index(&self) -> u32 {
        self.0
    }
    fn from_index(u: u32) -> InferVar {
        InferVar(u)
    }
    fn tag() -> &'static str {
        "InferVar"
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum InferVarVal {
    Known(Ty),
    Unknown,
}

/// Implements how the types are merged during unification
impl UnifyValue for InferVarVal {
    type Error = InferError;

    /// the idea of the unification is to merge two types and always taking the more concrete value
    fn unify_values(left: &Self, right: &Self) -> Result<Self, InferError> {
        use self::InferVarVal::*;
        match (left, right) {
            (Known(ty_l), Known(ty_r)) => Ok(Known(ty_l.unify(ty_r)?)),
            (Known(_), Unknown) => Ok(left.clone()),
            (Unknown, Known(_)) => Ok(right.clone()),
            (Unknown, Unknown) => Ok(Unknown),
        }
    }
}

impl Ty {
    /// Merges two types.
    /// `types_equal_rec` has to be called before.
    fn unify(&self, other: &Ty) -> Result<Ty, InferError> {
        trace!("unify {} {}", self, other);
        match (self, other) {
            (&Ty::Infer(_), &Ty::Infer(_)) => Ok(self.clone()),
            (&Ty::Infer(_), _) => Ok(other.clone()),
            (_, &Ty::Infer(_)) => Ok(self.clone()),
            (&Ty::Constr(constr_l), &Ty::Constr(constr_r)) => {
                if let Some(combined) = constr_l.conjunction(constr_r) {
                    Ok(Ty::Constr(combined))
                } else {
                    Err(InferError::ConflictingConstraint(constr_l, constr_r))
                }
            }
            (&Ty::Constr(constr), other) => {
                if other.satisfies(constr) {
                    Ok(other.clone())
                } else {
                    Err(InferError::TypeMismatch(self.clone(), other.clone()))
                }
            }
            (other, &Ty::Constr(constr)) => {
                if other.satisfies(constr) {
                    Ok(other.clone())
                } else {
                    Err(InferError::TypeMismatch(self.clone(), other.clone()))
                }
            }
            (Ty::Option(l), Ty::Option(r)) => Ok(Ty::Option(Box::new(l.unify(r)?))),
            (Ty::EventStream(l, param_l), Ty::EventStream(r, param_r)) => {
                if param_l.len() != param_r.len() {
                    return Err(InferError::TypeMismatch(self.clone(), other.clone()));
                }
                let mut param = Vec::new();
                for (el_l, el_r) in param_l.iter().zip(param_r) {
                    param.push(el_l.unify(el_r)?);
                }
                Ok(Ty::EventStream(Box::new(l.unify(r)?), param))
            }
            (Ty::Tuple(l), Ty::Tuple(r)) => {
                if l.len() != r.len() {
                    return Err(InferError::TypeMismatch(self.clone(), other.clone()));
                }
                let mut inner = Vec::new();
                for (e_l, e_r) in l.iter().zip(r) {
                    inner.push(e_l.unify(e_r)?);
                }
                Ok(Ty::Tuple(inner))
            }
            (Ty::TimedStream(_), Ty::TimedStream(_)) => unimplemented!(),
            (Ty::Window(ty_l, d_l), Ty::Window(ty_r, d_r)) => Ok(Ty::Window(
                Box::new(ty_l.unify(ty_r)?),
                Box::new(d_l.unify(d_r)?),
            )),
            (l, r) => {
                if l == r {
                    Ok(self.clone())
                } else {
                    Err(InferError::TypeMismatch(self.clone(), other.clone()))
                }
            }
        }
    }
}

impl EqUnifyValue for Ty {}

impl std::fmt::Display for InferVar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

type InferResult = Result<(), InferError>;

#[derive(Debug)]
pub enum InferError {
    TypeMismatch(Ty, Ty),
    ConflictingConstraint(TypeConstraint, TypeConstraint),
    CyclicDependency(InferVar, Ty),
}

impl InferError {
    fn normalize_types(&mut self, unifier: &mut Unifier) {
        if let InferError::TypeMismatch(ref mut expected, ref mut found) = self {
            std::mem::replace(expected, unifier.normalize_ty(expected.clone()));
            std::mem::replace(found, unifier.normalize_ty(found.clone()));
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::analysis::id_assignment::*;
    use crate::analysis::naming::*;
    use crate::parse::*;
    use crate::reporting::Handler;
    use crate::ty::{FloatTy, IntTy, UIntTy};
    use std::path::PathBuf;

    fn num_type_errors(spec: &str) -> usize {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new(&handler);
        na.check(&spec);
        assert!(
            !handler.contains_error(),
            "Spec produces errors in naming analysis."
        );
        let mut type_analysis = TypeAnalysis::new(&handler, &na.result);
        type_analysis.check(&spec);
        handler.emitted_errors()
    }

    /// Returns the type of the last output of the given spec
    fn get_type(spec: &str) -> Ty {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new(&handler);
        na.check(&spec);
        assert!(
            !handler.contains_error(),
            "Spec produces errors in naming analysis."
        );
        let mut type_analysis = TypeAnalysis::new(&handler, &na.result);
        type_analysis.check(&spec);
        type_analysis.get_type(
            spec.outputs
                .last()
                .expect("spec needs at least one output")
                ._id,
        )
    }

    #[test]
    fn simple_input() {
        let spec = "input i: Int8";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn parametric_input() {
        let spec = "input i<a: Int8, b: Bool>: Int8\noutput o := i(1,false)[0] ? 42";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I8).into(), vec![])
        );
    }

    #[test]
    #[cfg(feature = "methods")] // currently disabled, see `MethodLookup` in stdlib.rs
    fn method_call() {
        let spec = "output count := count.offset(-1).default(0) + 1\n";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I32).into(), vec![])
        );
    }

    #[test]
    #[cfg(feature = "methods")] // currently disabled, see `MethodLookup` in stdlib.rs
    fn method_call_with_type_param() {
        // count has value EventStream<Int8> instead of default value EventStream<Int32>
        let spec = "output count := count.offset<Int8>(-1).default(0) + 1\n";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I8).into(), vec![])
        );
    }

    #[test]
    #[cfg(feature = "methods")] // currently disabled, see `MethodLookup` in stdlib.rs
    fn method_call_with_type_param_faulty() {
        let spec = "output count := count.offset<Float32>(-1).default(0) + 1\n";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn simple_const_float() {
        let spec = "constant c: Float32 := 2.0";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn simple_const_int() {
        let spec = "constant c: Int8 := 3";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn simple_const_faulty() {
        let spec = "constant c: Int8 := true";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_signedness() {
        let spec = "constant c: UInt8 := -2";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_incorrect_float() {
        let spec = "constant c: UInt8 := 2.3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn simple_valid_coersion() {
        for spec in &[
            "constant c: Int8 := 1\noutput o: Int32 := c",
            "constant c: UInt16 := 1\noutput o: UInt64 := c",
            "constant c: Float32 := 1.0\noutput o: Float64 := c",
        ] {
            assert_eq!(0, num_type_errors(spec));
        }
    }

    #[test]
    fn simple_invalid_coersion() {
        let spec = "constant c: Int32 := 1\noutput o: Int8 := c";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn simple_output() {
        let spec = "output o: Int8 := 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I8).into(), vec![])
        );
    }

    #[test]
    fn simple_tigger() {
        let spec = "trigger never := false";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn faulty_tigger() {
        let spec = "trigger failed := 1";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn simple_binary() {
        let spec = "output o: Int8 := 3 + 5";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I8).into(), vec![])
        );
    }

    #[test]
    fn simple_unary() {
        let spec = "output o: Bool := !false";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), Ty::EventStream(Ty::Bool.into(), vec![]));
    }

    #[test]
    fn simple_unary_faulty() {
        // The negation should return a bool even if the underlying expression is wrong.
        // Thus, there is only one error here.
        let spec = "output o: Bool := !3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn simple_binary_faulty() {
        let spec = "output o: Float32 := false + 2.5";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn simple_ite() {
        let spec = "output o: Int8 := if false then 1 else 2";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I8).into(), vec![])
        );
    }

    #[test]
    fn simple_ite_compare() {
        let spec = "output e := if 1 == 0 then 0 else -1";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I32).into(), vec![])
        );
    }

    #[test]
    fn underspecified_ite_type() {
        let spec = "output o := if !false then 1.3 else -2.0";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Float(FloatTy::F32).into(), vec![])
        );
    }

    #[test]
    fn test_ite_condition_faulty() {
        let spec = "output o: UInt8 := if 3 then 1 else 1";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_ite_arms_incompatible() {
        let spec = "output o: UInt8 := if true then 1 else false";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_parenthized_expr() {
        let spec = "input s: String\noutput o: Bool := (s[-1] ? \"\") == \"a\"";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), Ty::EventStream(Ty::Bool.into(), vec![]));
    }

    #[test]
    fn test_underspecified_type() {
        let spec = "output o := 2";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I32).into(), vec![])
        );
    }

    #[test]
    fn test_trigonometric() {
        let spec = "import math\noutput o: Float32 := sin(2.0)";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Float(FloatTy::F32).into(), vec![])
        );
    }

    #[test]
    fn test_trigonometric_faulty() {
        let spec = "import math\noutput o: UInt8 := cos(1)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_trigonometric_faulty_2() {
        let spec = "import math\noutput o: Float64 := cos(true)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_regex() {
        let spec =
            "import regex\ninput s: String\noutput o: Bool := matches_regex(s[0], r\"(a+b)\")";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), Ty::EventStream(Ty::Bool.into(), vec![]));
    }

    #[test]
    fn test_input_lookup() {
        let spec = "input a: UInt8\n output b: UInt8 := a";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::UInt(UIntTy::U8).into(), vec![])
        );
    }

    #[test]
    fn test_input_lookup_faulty() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_stream_lookup() {
        let spec = "output a: UInt8 := 3\n output b: UInt8 := a[0]";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::UInt(UIntTy::U8).into(), vec![])
        );
    }

    #[test]
    fn test_stream_lookup_faulty() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_stream_lookup_dft() {
        let spec = "output a: UInt8 := 3\n output b: UInt8 := a[-1] ? 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::UInt(UIntTy::U8).into(), vec![])
        );
    }

    #[test]
    fn test_stream_lookup_dft_fault() {
        let spec = "output a: UInt8 := 3\n output b: Bool := a[-1] ? false";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_invoke_type() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I8).into(), vec![Ty::Int(IntTy::I8)])
        );
    }

    #[test]
    fn test_invoke_type_two_params() {
        let spec = "input in: Int8\n output a<p1: Int8, p2: Int8>: Int8 { invoke (in, in) } := 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(
                Ty::Int(IntTy::I8).into(),
                vec![Ty::Int(IntTy::I8), Ty::Int(IntTy::I8)]
            )
        );
    }

    #[test]
    fn test_invoke_type_faulty() {
        let spec = "input in: Bool\n output a<p1: Int8>: Int8 { invoke in } := 3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_extend_type() {
        let spec = "input in: Bool\n output a: Int8 { extend in } := 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I8).into(), vec![])
        );
    }

    #[test]
    fn test_extend_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 { extend in } := 3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_terminate_type() {
        let spec = "input in: Bool\n output a: Int8 { terminate in } := 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I8).into(), vec![])
        );
    }

    #[test]
    fn test_terminate_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 { terminate in } := 3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_param_spec() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: Int8 := a(3)[-2] ? 1";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I8).into(), vec![])
        );
    }

    #[test]
    fn test_param_spec_faulty() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: Int8 := a(true)[-2] ? 1";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_lookup_incomp() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: UInt8 := a(3)[2] ? 1";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_tuple() {
        let spec = "output out: (Int8, Bool) := (14, false)";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Tuple(vec![Ty::Int(IntTy::I8), Ty::Bool]).into(), vec![])
        );
    }

    #[test]
    fn test_tuple_faulty() {
        let spec = "output out: (Int8, Bool) := (14, 3)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_tuple_access() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in[0].1";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), Ty::EventStream(Ty::Bool.into(), vec![]));
    }

    #[test]
    fn test_tuple_access_faulty_type() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in[0].0";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_tuple_access_faulty_len() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in[0].2";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_input_offset() {
        let spec = "input a: UInt8\n output b: UInt8 := a[3]";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::UInt(UIntTy::U8).into(), vec![])
        );
    }

    #[test]
    fn test_tuple_of_tuples() {
        let spec = "input in: (Int8, (UInt8, Bool))\noutput out: Int16 := in[0].0";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(
            get_type(spec),
            Ty::EventStream(Ty::Int(IntTy::I16).into(), vec![])
        );
    }

    #[test]
    fn test_tuple_of_tuples2() {
        let spec = "input in: (Int8, (UInt8, Bool))\noutput out: Bool := in[0].1.1";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), Ty::EventStream(Ty::Bool.into(), vec![]));
    }

    #[test]
    #[ignore] // type system for timed streams is currently not implemented
    fn test_window_widening() {
        let spec = "input in: Int8\n output out: Int64 {extend @5Hz}:= in[3s, Σ] ? 0";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    #[ignore] // type system for timed streams is currently not implemented
    fn test_window() {
        let spec = "input in: Int8\n output out: Int8 {extend @5Hz} := in[3s, Σ] ? 0";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    #[ignore] // type system for timed streams is currently not implemented
    fn test_window_untimed() {
        let spec = "input in: Int8\n output out: Int16 := in[3s, Σ] ? 5";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    #[ignore] // type system for timed streams is currently not implemented
    fn test_window_faulty() {
        let spec = "input in: Int8\n output out: Bool {extend @5Hz} := in[3s, Σ] ? 5";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    #[ignore] // type system for timed streams is currently not implemented
    fn test_involved() {
        let spec =
            "input velo: Float32\n output avg: Float64 {extend @5Hz} := velo[1h, avg] ? 10000.0";
        assert_eq!(0, num_type_errors(spec));
    }

}
