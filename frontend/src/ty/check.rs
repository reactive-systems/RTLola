//! Implementation of the Hindley-Milner type inference through unification
//!
//! Relevant references:
//! * [Introduction to Hindley-Milner type inference](https://eli.thegreenplace.net/2018/type-inference/)
//! * [Unification algorithm](https://eli.thegreenplace.net/2018/unification/)
//! * [Unification in Rust](http://smallcultfollowing.com/babysteps/blog/2017/03/25/unification-in-chalk-part-1/)
//! * [Ena (union-find package)](https://crates.io/crates/ena)

use super::unifier::{InferError, StreamVar, StreamVarOrTy, UnifiableTy, Unifier, ValueUnifier, ValueVar};
use super::{Activation, Freq, StreamTy, TypeConstraint, ValueTy};
use crate::analysis::naming::{Declaration, DeclarationTable};
use crate::ast::{
    BinOp, Constant, Expression, ExpressionKind, Input, Literal, LolaSpec, Offset, Output,
    StreamAccessKind, Trigger, Type, TypeKind, WindowOperation,
};
use crate::parse::{NodeId, Span};
use crate::reporting::{Handler, LabeledSpan};
use crate::stdlib;
use crate::stdlib::{FuncDecl, MethodLookup};
use log::{debug, trace};
use num::traits::ops::inv::Inv;
use num::Signed;
use std::collections::HashMap;
use uom::si::frequency::hertz;
use uom::si::rational64::Frequency as UOM_Frequency;
use uom::si::time::second;

pub(crate) struct TypeAnalysis<'a, 'b, 'c> {
    handler: &'b Handler,
    declarations: &'c mut DeclarationTable<'a>,
    method_lookup: MethodLookup<'a>,
    unifier: ValueUnifier<ValueTy>,
    stream_unifier: ValueUnifier<StreamTy>,
    /// maps `NodeId`'s to the variables used in `unifier`
    value_vars: HashMap<NodeId, ValueVar>,
    stream_vars: HashMap<NodeId, StreamVar>,
    /// maps function-like nodes (UnOp, BinOp, Func, Method) to the generic parameters
    generic_function_vars: HashMap<NodeId, Vec<ValueVar>>,
}

pub(crate) struct TypeTable {
    value_tt: HashMap<NodeId, ValueTy>,
    stream_tt: HashMap<NodeId, StreamTy>,
    func_tt: HashMap<NodeId, Vec<ValueTy>>,
    acti_cond: HashMap<NodeId, Activation<crate::ir::StreamReference>>,
}

impl TypeTable {
    pub(crate) fn get_value_type(&self, nid: NodeId) -> &ValueTy {
        &self.value_tt[&nid]
    }

    pub(crate) fn get_stream_type(&self, nid: NodeId) -> &StreamTy {
        &self.stream_tt[&nid]
    }

    pub(crate) fn get_func_arg_types(&self, nid: NodeId) -> &Vec<ValueTy> {
        &self.func_tt[&nid]
    }

    pub(crate) fn get_acti_cond(&self, nid: NodeId) -> &Activation<crate::ir::StreamReference> {
        &self.acti_cond[&nid]
    }
}

impl<'a, 'b, 'c> TypeAnalysis<'a, 'b, 'c> {
    pub(crate) fn new(handler: &'b Handler, declarations: &'c mut DeclarationTable<'a>) -> TypeAnalysis<'a, 'b, 'c> {
        TypeAnalysis {
            handler,
            declarations,
            method_lookup: MethodLookup::new(),
            unifier: ValueUnifier::new(),
            stream_unifier: ValueUnifier::new(),
            value_vars: HashMap::new(),
            stream_vars: HashMap::new(),
            generic_function_vars: HashMap::new(),
        }
    }

    pub(crate) fn check(&mut self, spec: &'a LolaSpec) -> Option<TypeTable> {
        self.imports(spec);
        self.infer_types(spec);
        if self.handler.contains_error() {
            return None;
        }

        self.assign_types(spec);

        Some(self.extract_type_table(spec))
    }

    fn extract_type_table(&mut self, spec: &'a LolaSpec) -> TypeTable {
        let value_nids: Vec<NodeId> = self.value_vars.keys().cloned().collect();
        let vtt: HashMap<NodeId, ValueTy> = value_nids.into_iter().map(|nid| (nid, self.get_type(nid))).collect();

        // Note: If there is a `None` for a stream, `get_stream_type` would report the error.
        let stream_nids: Vec<NodeId> = self.stream_vars.keys().cloned().collect();
        let stt: HashMap<NodeId, StreamTy> =
            stream_nids.iter().flat_map(|&nid| self.get_stream_type(nid).map(|ty| (nid, ty))).collect();

        let mut func_tt = HashMap::with_capacity(self.generic_function_vars.len());
        let source = &self.generic_function_vars;
        let unifier = &mut self.unifier;
        for (&key, val) in source {
            func_tt.insert(
                key,
                val.iter()
                    .map(|&var| unifier.get_normalized_type(var).unwrap_or(ValueTy::Error).replace_constr())
                    .collect(),
            );
        }

        let acti_cond =
            stream_nids.iter().flat_map(|&nid| self.get_activation_condition(spec, nid).map(|ac| (nid, ac))).collect();

        TypeTable { value_tt: vtt, stream_tt: stt, func_tt, acti_cond }
    }

    fn imports(&mut self, spec: &'a LolaSpec) {
        for import in &spec.imports {
            match import.name.name.as_str() {
                "math" => stdlib::import_math_method(&mut self.method_lookup),
                "regex" => stdlib::import_regex_method(&mut self.method_lookup),
                n => self.handler.error_with_span(
                    &format!("unresolved import `{}`", n),
                    LabeledSpan::new(import.name.span, &format!("no `{}` in the root", n), true),
                ),
            }
        }
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

    fn new_value_var(&mut self, node: NodeId) -> ValueVar {
        assert!(!self.value_vars.contains_key(&node));
        let var = self.unifier.new_var();
        self.value_vars.insert(node, var);
        var
    }

    fn new_stream_var(&mut self, node: NodeId) -> StreamVar {
        assert!(!self.stream_vars.contains_key(&node));
        let var = self.stream_unifier.new_var();
        self.stream_vars.insert(node, var);
        var
    }

    fn infer_constant(&mut self, constant: &'a Constant) -> Result<(), ()> {
        trace!("infer type for {} (NodeId = {})", constant, constant.id);
        let var = self.new_value_var(constant.id);

        if let Some(ast_ty) = &constant.ty {
            self.infer_type(&ast_ty)?;
            let ty_var = self.value_vars[&ast_ty.id];
            self.unifier.unify_var_var(var, ty_var).expect("cannot fail as `var` is a fresh var");
        }

        // generate constraint from literal
        self.unifier
            .unify_var_ty(var, self.get_constraint_for_literal(&constant.literal))
            .map_err(|err| self.handle_error(err, constant.literal.span))?;
        Ok(())
    }

    fn infer_input(&mut self, input: &'a Input) -> Result<(), ()> {
        trace!("infer type for {} (NodeId = {})", input, input.id);

        // value type
        let var = self.new_value_var(input.id);

        self.infer_type(&input.ty)?;
        let ty_var = self.value_vars[&input.ty.id];

        self.unifier.unify_var_var(var, ty_var).map_err(|err| self.handle_error(err, input.name.span))?;

        // stream type
        let stream_var = self.new_stream_var(input.id);

        // determine parameters
        let mut param_types = Vec::new();
        for param in &input.params {
            self.infer_type(&param.ty)?;
            let param_ty_var = self.value_vars[&param.ty.id];
            let param_var = self.new_value_var(param.id);
            self.unifier.unify_var_var(param_var, param_ty_var).expect("cannot fail as `param_var` is fresh");
            param_types.push(ValueTy::Infer(param_var));
        }

        assert!(param_types.is_empty(), "parametric types are currently not supported");

        self.stream_unifier
            .unify_var_ty(stream_var, StreamTy::new_event(Activation::Stream(stream_var)))
            .map_err(|err| self.handle_error(err, input.name.span))
    }

    fn infer_output(&mut self, output: &'a Output) -> Result<(), ()> {
        trace!("infer type for {} (NodeId = {})", output, output.id);

        // value type
        let var = self.new_value_var(output.id);

        // generate constraint in case a type is annotated
        self.infer_type(&output.ty)?;
        let ty_var = self.value_vars[&output.ty.id];

        self.unifier.unify_var_var(var, ty_var).map_err(|err| self.handle_error(err, output.name.span))?;

        // stream type
        let stream_var = self.new_stream_var(output.id);

        // collect parameters
        let mut param_types = Vec::new();
        for param in &output.params {
            self.infer_type(&param.ty)?;
            let param_ty_var = self.value_vars[&param.ty.id];
            let param_var = self.new_value_var(param.id);
            self.unifier.unify_var_var(param_var, param_ty_var).expect("cannot fail as `param_var` is fresh");
            param_types.push(ValueTy::Infer(param_var));
        }

        // check if stream has timing infos
        let mut frequency = None;
        let mut activation = None;
        if let Some(expr) = &output.extend.expr {
            if let Ok(freq) = expr.parse_frequency() {
                frequency = Some(Freq::new(freq));
            } else if let Some(act) = self.parse_activation_condition(expr) {
                activation = Some(act);
            } else {
                self.handler.error_with_span(
                    "expected frequency or activation conditon",
                    LabeledSpan::new(
                        expr.span,
                        "allowed are frequencies (`1Hz`) or Boolean conditions (`x | y`)",
                        true,
                    ),
                );
            }
        }

        assert!(param_types.is_empty(), "Parametric outputs are currently not supported by type checker");

        // determine whether stream is timed or event based or should be infered
        if let Some(f) = frequency {
            self.stream_unifier
                .unify_var_ty(stream_var, StreamTy::new_periodic(f))
                .map_err(|err| self.handle_error(err, output.name.span))
        } else if let Some(act) = activation {
            self.stream_unifier
                .unify_var_ty(stream_var, StreamTy::new_event(act))
                .map_err(|err| self.handle_error(err, output.name.span))
        } else {
            // stream type should be inferred
            self.stream_unifier
                .unify_var_ty(stream_var, StreamTy::new_inferred())
                .map_err(|err| self.handle_error(err, output.name.span))
        }
    }

    fn parse_activation_condition(&mut self, expr: &Expression) -> Option<Activation<StreamVar>> {
        match &expr.kind {
            ExpressionKind::Ident(_) => match self.declarations[&expr.id] {
                Declaration::In(input) => Some(Activation::Stream(self.stream_vars[&input.id])),
                Declaration::Out(output) => Some(Activation::Stream(self.stream_vars[&output.id])),
                _ => None,
            },
            ExpressionKind::Binary(op, left, right) => {
                let (left, right) =
                    match (self.parse_activation_condition(left), self.parse_activation_condition(right)) {
                        (Some(left), Some(right)) => (left, right),
                        _ => return None,
                    };
                match op {
                    BinOp::And => Some(Activation::Conjunction(vec![left, right])),
                    BinOp::Or => Some(Activation::Disjunction(vec![left, right])),
                    _ => None,
                }
            }
            ExpressionKind::ParenthesizedExpression(_, expr, _) => self.parse_activation_condition(expr),
            _ => None,
        }
    }

    fn infer_type(&mut self, ast_ty: &'a Type) -> Result<(), ()> {
        trace!("infer type for {}", ast_ty);
        let ty_var = self.new_value_var(ast_ty.id);
        match &ast_ty.kind {
            TypeKind::Inferred => Ok(()),
            TypeKind::Simple(_) => {
                match self.declarations[&ast_ty.id] {
                    Declaration::Type(ty) => {
                        // ?ty_var = `ty`
                        self.unifier
                            .unify_var_ty(ty_var, (*ty).clone())
                            .map_err(|err| self.handle_error(err, ast_ty.span))
                    }
                    _ => unreachable!(),
                }
            }
            TypeKind::Tuple(tuple) => {
                let ty = self.get_tuple_type(tuple);
                // ?ty_var = `ty`
                self.unifier.unify_var_ty(ty_var, ty).map_err(|err| self.handle_error(err, ast_ty.span))
            }
            _ => unreachable!(),
        }
    }

    fn infer_output_expression(&mut self, output: &'a Output) -> Result<(), ()> {
        trace!("infer type for {}", output);
        let out_var = self.value_vars[&output.id];

        // check template specification
        /*if let Some(template_spec) = &output.template_spec {
            if let Some(invoke) = &template_spec.inv {
                self.infer_expression(&invoke.target, None, StreamVarOrTy::Ty(StreamTy::new(TimingInfo::Event)))?;
                let inv_var = self.value_vars[&invoke.target.id];
                if output.params.len() == 1 {
                    // ?param_var = ?inv_bar
                    let param_var = self.value_vars[&output.params[0].id];
                    self.unifier
                        .unify_var_ty(inv_var, ValueTy::Infer(param_var))
                        .map_err(|err| self.handle_error(err, invoke.target.span))?;
                } else {
                    let target_ty = ValueTy::Tuple(
                        output
                            .params
                            .iter()
                            .map(|p| {
                                let param_var = self.value_vars[&p.id];
                                ValueTy::Infer(param_var)
                            })
                            .collect(),
                    );
                    self.unifier
                        .unify_var_ty(inv_var, target_ty)
                        .map_err(|err| self.handle_error(err, invoke.target.span))?;
                }

                // check that condition is boolean
                if let Some(cond) = &invoke.condition {
                    self.infer_expression(
                        cond,
                        Some(ValueTy::Bool),
                        StreamVarOrTy::Ty(StreamTy::new(TimingInfo::Event)),
                    )?;
                }
            }
            if let Some(extend) = &template_spec.ext {
                // check that condition is boolean
                self.infer_expression(
                    &extend.target,
                    Some(ValueTy::Bool),
                    StreamVarOrTy::Ty(StreamTy::new(TimingInfo::Event)),
                )?;
            }
            if let Some(terminate) = &template_spec.ter {
                // check that condition is boolean
                self.infer_expression(
                    &terminate.target,
                    Some(ValueTy::Bool),
                    StreamVarOrTy::Ty(StreamTy::new(TimingInfo::Event)),
                )?;
            }
        }*/

        let out_stream_var = self.stream_vars[&output.id];

        // generate constraint for expression
        self.infer_expression(
            &output.expression,
            Some(ValueTy::Infer(out_var)),
            Some(StreamVarOrTy::Var(out_stream_var)),
        )
    }

    fn infer_trigger_expression(&mut self, trigger: &'a Trigger) -> Result<(), ()> {
        trace!("infer type for {}", trigger);

        // make sure that NodeId of trigger is assigned Bool
        let var = self.new_value_var(trigger.id);
        self.unifier.unify_var_ty(var, ValueTy::Bool).expect("cannot fail as `var` is fresh");

        let stream_var = self.new_stream_var(trigger.id);

        self.infer_expression(&trigger.expression, Some(ValueTy::Infer(var)), Some(StreamVarOrTy::Var(stream_var)))
    }

    fn get_constraint_for_literal(&self, lit: &Literal) -> ValueTy {
        use crate::ast::LitKind::*;
        match &lit.kind {
            Str(_) | RawStr(_) => ValueTy::String,
            Bool(_) => ValueTy::Bool,
            Numeric(val, unit) => {
                assert!(unit.is_none());
                if val.contains('.') {
                    // Floating Point
                    ValueTy::Constr(TypeConstraint::FloatingPoint)
                } else if val.starts_with('-') {
                    ValueTy::Constr(TypeConstraint::SignedInteger)
                } else {
                    ValueTy::Constr(TypeConstraint::Integer)
                }
            }
        }
    }

    fn infer_expression(
        &mut self,
        expr: &'a Expression,
        target: Option<ValueTy>,
        stream_ty: Option<StreamVarOrTy>,
    ) -> Result<(), ()> {
        use crate::ast::ExpressionKind::*;
        trace!("infer expression {} (NodeId = {})", expr, expr.id);

        let var = self.new_value_var(expr.id);
        if let Some(target_ty) = &target {
            self.unifier.unify_var_ty(var, target_ty.clone()).expect("unification cannot fail as `var` is fresh");
        }

        let stream_var = self.new_stream_var(expr.id);
        match stream_ty.as_ref() {
            Some(StreamVarOrTy::Ty(ty)) => self
                .stream_unifier
                .unify_var_ty(stream_var, ty.clone())
                .expect("unification cannot fail as `var` is fresh"),
            Some(&StreamVarOrTy::Var(var)) => {
                self.stream_unifier.unify_var_var(stream_var, var).expect("unification cannot fail as `var` is fresh")
            }
            None => {}
        }

        match &expr.kind {
            Lit(l) => {
                // generate value type constraint from literal
                self.unifier
                    .unify_var_ty(var, self.get_constraint_for_literal(&l))
                    .map_err(|err| self.handle_error(err, expr.span))?;

                // no stream type constraint is needed
            }
            Ident(_) => {
                let decl = self.declarations[&expr.id];

                match decl {
                    Declaration::Const(constant) => {
                        let const_var = self.value_vars[&constant.id];
                        self.unifier.unify_var_var(var, const_var).map_err(|err| self.handle_error(err, expr.span))?
                    }
                    Declaration::In(input) => {
                        // value type
                        let in_var = self.value_vars[&input.id];
                        self.unifier.unify_var_var(var, in_var).map_err(|err| self.handle_error(err, expr.span))?;

                        // stream type
                        let in_var = self.stream_vars[&input.id];
                        self.stream_unifier
                            .unify_var_var(stream_var, in_var)
                            .map_err(|err| self.handle_error(err, expr.span))?;
                    }
                    Declaration::Out(output) => {
                        // value type
                        let out_var = self.value_vars[&output.id];
                        self.unifier.unify_var_var(var, out_var).map_err(|err| self.handle_error(err, expr.span))?;

                        // stream type
                        let out_var = self.stream_vars[&output.id];
                        self.stream_unifier
                            .unify_var_var(stream_var, out_var)
                            .map_err(|err| self.handle_error(err, expr.span))?;
                    }
                    Declaration::Param(param) => {
                        // value type only
                        let param_var = self.value_vars[&param.id];
                        self.unifier.unify_var_var(var, param_var).map_err(|err| self.handle_error(err, expr.span))?;
                    }
                    _ => unreachable!("unreachable ident {:?}", decl),
                }
            }
            Offset(inner, offset) => self.infer_offset_expr(var, stream_var, expr.span, inner, offset)?,
            SlidingWindowAggregation { expr: inner, duration, aggregation } => {
                self.infer_sliding_window_expression(var, stream_var, expr.span, inner, duration, *aggregation)?;
            }
            Ite(cond, left, right) => {
                // value type constraints
                // * `cond` = Bool
                // * `left` = `var`
                // * `right` = `var`
                // stream type constraints: all should be the same
                self.infer_expression(cond, Some(ValueTy::Bool), Some(StreamVarOrTy::Var(stream_var)))?;
                self.infer_expression(left, Some(ValueTy::Infer(var)), Some(StreamVarOrTy::Var(stream_var)))?;
                self.infer_expression(right, Some(ValueTy::Infer(var)), Some(StreamVarOrTy::Var(stream_var)))?;
            }
            Unary(op, appl) => {
                self.infer_function_application(
                    expr.id,
                    var,
                    stream_var,
                    expr.span,
                    &op.get_func_decl(),
                    &[],
                    &[appl],
                )?;
            }
            Binary(op, left, right) => {
                self.infer_function_application(
                    expr.id,
                    var,
                    stream_var,
                    expr.span,
                    &op.get_func_decl(),
                    &[],
                    &[left, right],
                )?;
            }
            Default(left, right) => {
                self.infer_expression(
                    left,
                    Some(ValueTy::Option(ValueTy::Infer(var).into())),
                    Some(StreamVarOrTy::Var(stream_var)),
                )?;
                self.infer_expression(right, Some(ValueTy::Infer(var)), Some(StreamVarOrTy::Var(stream_var)))?
            }
            StreamAccess(inner, access_type) => {
                // result type is an optional value
                let target_var = self.unifier.new_var();
                self.unifier
                    .unify_var_ty(var, ValueTy::Option(ValueTy::Infer(target_var).into()))
                    .map_err(|err| self.handle_error(err, inner.span))?;

                // the stream type of `inner` is unconstrained
                self.infer_expression(inner, Some(ValueTy::Infer(var)), None)?;

                let function = match access_type {
                    StreamAccessKind::Hold => "hold()",
                    StreamAccessKind::Optional => "get()",
                };

                // check that stream types are not compatible
                let inner_var = self.stream_vars[&inner.id];
                match (
                    self.stream_unifier.get_type(stream_var).map(|ty| ty.normalize_ty(&mut self.stream_unifier)),
                    self.stream_unifier.get_type(inner_var),
                ) {
                    (Some(StreamTy::Event(left)), Some(StreamTy::Event(right))) => {
                        if left.implies_valid(&right, &mut self.stream_unifier) {
                            self.handler.warn_with_span(
                                &format!("Unnecessary `.{}`", function),
                                LabeledSpan::new(expr.span, &format!("remove `.{}`", function), true),
                            )
                        }
                    }
                    (Some(StreamTy::RealTime(left)), Some(StreamTy::RealTime(right))) => {
                        if right.is_multiple_of(&left) {
                            self.handler.warn_with_span(
                                &format!("Unnecessary `.{}`", function),
                                LabeledSpan::new(expr.span, &format!("remove `.{}`", function), true),
                            )
                        }
                    }
                    _ => {
                        if let StreamAccessKind::Optional = access_type {
                            self.handler.error_with_span(
                                "`get()` can be only used when both streams are event-based or periodic",
                                LabeledSpan::new(expr.span, "`get()` not possible here", true),
                            )
                        }
                    }
                }
            }
            Function(_name, types, params) => {
                let decl = self.declarations[&expr.id];
                let fun_decl = match decl {
                    Declaration::Func(fun_decl) => fun_decl,
                    _ => unreachable!("expected function declaration"),
                };
                if params.len() != fun_decl.parameters.len() {
                    self.handler.error_with_span(
                        &format!(
                            "this function takes {} parameters but {} parameters were supplied",
                            fun_decl.parameters.len(),
                            params.len()
                        ),
                        LabeledSpan::new(
                            expr.span,
                            &format!("expected {} parameters", fun_decl.parameters.len()),
                            true,
                        ),
                    );
                    return Err(());
                }
                assert_eq!(params.len(), fun_decl.parameters.len());

                //                println!("{:?}", fun_decl);

                for ty in types {
                    self.infer_type(ty)?;
                }

                let params: Vec<&Expression> = params.iter().map(Box::as_ref).collect();

                self.infer_function_application(
                    expr.id,
                    var,
                    stream_var,
                    expr.span,
                    fun_decl,
                    types.as_slice(),
                    params.as_slice(),
                )?;
            }
            Method(base, name, types, params) => {
                // recursion
                self.infer_expression(base, None, Some(StreamVarOrTy::Var(stream_var)))?;

                if let Some(infered) = self.unifier.get_type(self.value_vars[&base.id]) {
                    debug!("{} {}", base, infered);
                    if let Some(fun_decl) = self.method_lookup.get(&infered, &name) {
                        debug!("{:?}", fun_decl);

                        for ty in types {
                            self.infer_type(ty)?;
                        }

                        let parameters: Vec<_> = std::iter::once(base).chain(params).map(Box::as_ref).collect();

                        self.infer_function_application(
                            expr.id,
                            var,
                            stream_var,
                            expr.span,
                            fun_decl,
                            types.as_slice(),
                            parameters.as_slice(),
                        )?;

                        self.declarations.insert(expr.id, fun_decl.into());
                    } else {
                        self.handler.error_with_span(
                            &format!("unknown method `{}`", name),
                            LabeledSpan::new(expr.span, &format!("no method `{}` for `{}`", name, infered), true),
                        );
                    }
                } else {
                    self.handler.error_with_span(
                        &format!("could not determine type of `{}`", base),
                        LabeledSpan::new(base.span, "consider giving a type annotation", true),
                    );
                }
            }
            Tuple(expressions) => {
                // recursion
                let mut tuples: Vec<ValueTy> = Vec::with_capacity(expressions.len());
                for element in expressions {
                    self.infer_expression(element, None, Some(StreamVarOrTy::Var(stream_var)))?;
                    let inner = self.unifier.get_type(self.value_vars[&element.id]).expect("should have type");
                    tuples.push(inner);
                }
                // ?var = Tuple(?expr1, ?expr2, ..)
                self.unifier
                    .unify_var_ty(var, ValueTy::Tuple(tuples))
                    .map_err(|err| self.handle_error(err, expr.span))?;
            }
            Field(base, ident) => {
                // recursion
                self.infer_expression(base, None, Some(StreamVarOrTy::Var(stream_var)))?;

                let ty = self.unifier.get_type(self.value_vars[&base.id]).expect("should have type at this point");
                let infered = ty.normalize_ty(&mut self.unifier);

                debug!("{} {}", base, infered);
                match infered {
                    ValueTy::Tuple(inner) => {
                        let num: usize = ident.name.parse::<usize>().expect("checked in AST verifier");
                        if num >= inner.len() {
                            self.handler.error_with_span(
                                &format!("Try to access tuple at position {}", num),
                                LabeledSpan::new(ident.span, "", true),
                            );
                            return Err(());
                        }
                        // ?var = inner[num]
                        self.unifier
                            .unify_var_ty(var, inner[num].clone())
                            .map_err(|err| self.handle_error(err, expr.span))?;
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
            ParenthesizedExpression(_, expr, _) => self.infer_expression(expr, target, stream_ty)?,
            MissingExpression => {
                // we simply ignore missing expressions and continue with type analysis
            }
        }
        Ok(())
    }

    fn infer_offset_expr(
        &mut self,
        var: ValueVar,
        stream_var: StreamVar,
        span: Span,
        expr: &'a Expression,
        offset: &'a Offset,
    ) -> Result<(), ()> {
        // check if offset is discrete or time-based
        match offset {
            Offset::Discrete(offset) => {
                let negative_offset = offset.is_negative();

                // result type is an optional value if offset is negative
                let target_var = self.unifier.new_var();
                if negative_offset {
                    self.unifier
                        .unify_var_ty(var, ValueTy::Option(ValueTy::Infer(target_var).into()))
                        .map_err(|err| self.handle_error(err, span))?;
                } else {
                    self.unifier.unify_var_var(var, target_var).map_err(|err| self.handle_error(err, span))?;
                }

                // As the recursion checks that the stream types match, any integer offset will match as well.
                self.infer_expression(expr, Some(ValueTy::Infer(target_var)), Some(StreamVarOrTy::Var(stream_var)))
            }
            Offset::RealTime(time, _) => {
                // target value type
                let target_value_var = self.unifier.new_var();
                if time.is_negative() {
                    self.unifier
                        .unify_var_ty(var, ValueTy::Option(ValueTy::Infer(target_value_var).into()))
                        .map_err(|err| self.handle_error(err, span))?;
                } else {
                    self.unifier.unify_var_var(var, target_value_var).map_err(|err| self.handle_error(err, span))?;
                }
                // get frequency
                let time = offset.to_uom_time().expect("guaranteed to be real-time");
                let freq = UOM_Frequency::new::<hertz>(time.get::<second>().abs().inv());

                // target stream type
                let target_stream_ty = StreamTy::new_periodic(Freq::new(freq));

                // recursion
                // stream types have to match
                self.infer_expression(
                    expr,
                    Some(ValueTy::Infer(target_value_var)),
                    Some(StreamVarOrTy::Ty(target_stream_ty)),
                )
            }
        }
    }

    fn infer_sliding_window_expression(
        &mut self,
        var: ValueVar,
        stream_var: StreamVar,
        span: Span,
        expr: &'a Expression,
        _duration: &'a Expression,
        window_op: WindowOperation,
    ) -> Result<(), ()> {
        // the stream variable has to be real-time
        match self.stream_unifier.get_normalized_type(stream_var) {
            Some(StreamTy::RealTime(_)) => {}
            _ => {
                self.handler.error_with_span(
                    "Sliding windows are only allowed in real-time streams",
                    LabeledSpan::new(span, "unexpected sliding window", true),
                );
                return Err(());
            }
        }

        // value type depends on the aggregation function
        // stream type is not restricted
        use WindowOperation::*;
        match window_op {
            Count => {
                // The value type of the inner stream is not restricted
                self.infer_expression(expr, None, None)?;
                // resulting type is a integer value
                self.unifier
                    .unify_var_ty(var, ValueTy::Constr(TypeConstraint::UnsignedInteger))
                    .map_err(|err| self.handle_error(err, span))
            }
            Sum => {
                // The value type of the inner stream has to be numeric
                self.infer_expression(expr, Some(ValueTy::Constr(TypeConstraint::Numeric)), None)?;
                // resulting type depends on the inner type
                let inner_var = self.value_vars[&expr.id];
                self.unifier.unify_var_ty(var, ValueTy::Infer(inner_var)).map_err(|err| self.handle_error(err, span))
            }
            Average | Product | Integral => {
                // The value type of the inner stream has to be numeric
                self.infer_expression(expr, Some(ValueTy::Constr(TypeConstraint::Numeric)), None)?;
                // resulting type depends on the inner type
                let inner_var = self.value_vars[&expr.id];
                self.unifier
                    .unify_var_ty(var, ValueTy::Option(ValueTy::Infer(inner_var).into()))
                    .map_err(|err| self.handle_error(err, span))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn infer_function_application(
        &mut self,
        node_id: NodeId,
        var: ValueVar,
        stream_var: StreamVar,
        span: Span,
        fun_decl: &FuncDecl,
        types: &[Type],
        params: &[&'a Expression],
    ) -> Result<(), ()> {
        trace!("infer function application");
        assert!(types.len() <= fun_decl.generics.len());
        assert_eq!(params.len(), fun_decl.parameters.len());

        // build symbolic names for generic arguments
        let generics: Vec<ValueVar> = fun_decl
            .generics
            .iter()
            .map(|gen| {
                let var = self.unifier.new_var();
                match &gen.constraint {
                    ValueTy::Constr(_) => {}
                    _ => unreachable!("currently, only constraints are allowed for generic types"),
                }
                self.unifier.unify_var_ty(var, gen.constraint.clone()).expect("cannot fail as var is freshly created");
                var
            })
            .collect();
        for (provided_type, &generic) in types.iter().zip(&generics) {
            println!("{} {}", provided_type, generic);
            self.unifier
                .unify_var_var(generic, self.value_vars[&provided_type.id])
                .map_err(|err| self.handle_error(err, provided_type.span))?;
        }

        for (type_param, parameter) in fun_decl.parameters.iter().zip(params) {
            let ty = type_param.replace_params(&generics);
            if let Some(&param_var) = self.value_vars.get(&parameter.id) {
                // for method calls, we have to infer type for first argument
                // ?param_var = `ty`
                self.unifier.unify_var_ty(param_var, ty).map_err(|err| self.handle_error(err, parameter.span))?;
            } else {
                // otherwise, we have to check it now
                self.infer_expression(parameter, Some(ty), Some(StreamVarOrTy::Var(stream_var)))?;
            }
        }
        // return type
        let ty = fun_decl.return_type.replace_params(&generics);

        // ?param_var = `ty`
        self.unifier.unify_var_ty(var, ty).map_err(|err| self.handle_error(err, span))?;

        // store generic parameters for later lookup
        self.generic_function_vars.insert(node_id, generics);
        Ok(())
    }

    fn get_tuple_type(&mut self, tuple: &[Type]) -> ValueTy {
        let mut inner = Vec::new();
        for t in tuple {
            match &t.kind {
                TypeKind::Simple(_) => {
                    let ty = self.declarations[&t.id];
                    match ty {
                        Declaration::Type(ty) => inner.push(ty.clone()),
                        _ => unreachable!(),
                    }
                }
                TypeKind::Tuple(types) => inner.push(self.get_tuple_type(&types)),
                _ => unreachable!(),
            }
        }
        ValueTy::Tuple(inner)
    }

    /// Assigns types as infered
    fn assign_types(&mut self, spec: &LolaSpec) {
        for constant in &spec.constants {
            debug!(
                "{} has type {}",
                constant,
                self.unifier.get_normalized_type(self.value_vars[&constant.id]).unwrap()
            );
        }
        for input in &spec.inputs {
            debug!("{} has type {}", input, self.unifier.get_normalized_type(self.value_vars[&input.id]).unwrap());
        }
        for output in &spec.outputs {
            debug!("{} has type {}", output, self.unifier.get_normalized_type(self.value_vars[&output.id]).unwrap());
        }
    }

    fn handle_error(&mut self, mut err: InferError, span: Span) {
        err.normalize_types(&mut self.unifier);
        match err {
            InferError::ValueTypeMismatch(ty_l, ty_r) => {
                self.handler.error_with_span(
                    &format!("Type mismatch between `{}` and `{}`", ty_l, ty_r),
                    LabeledSpan::new(span, &format!("expected `{}`, found `{}`", ty_l, ty_r), true),
                );
            }
            InferError::ConflictingConstraint(left, right) => {
                self.handler.error_with_span(
                    &format!("Conflicting constraints `{}` and `{}`", left, right),
                    LabeledSpan::new(span, &format!("no concrete type satisfies `{}` and `{}`", left, right), true),
                );
            }
            InferError::CyclicDependency => {
                self.handler.error_with_span(
                    "Cannot infer type",
                    LabeledSpan::new(span, "consider using a type annotation", true),
                );
            }
            InferError::StreamTypeMismatch(ty_l, ty_r, hint) => {
                let mut diagnostics = self.handler.build_error_with_span(
                    &format!("Type mismatch between `{}` and `{}`", ty_l, ty_r),
                    LabeledSpan::new(span, &format!("expected `{}`, found `{}`", ty_l, ty_r), true),
                );
                if let Some(hint) = hint {
                    diagnostics.add_labeled_span(LabeledSpan::new(span, &format!("hint: {}", hint), false))
                }
                diagnostics.emit();
            }
        }
    }

    pub(crate) fn get_type(&mut self, id: NodeId) -> ValueTy {
        if let Some(&var) = self.value_vars.get(&id) {
            if let Some(t) = self.unifier.get_normalized_type(var) {
                t.replace_constr()
            } else {
                ValueTy::Error
            }
        } else {
            ValueTy::Error
        }
    }

    pub(crate) fn get_stream_type(&mut self, id: NodeId) -> Option<StreamTy> {
        match self.stream_vars.get(&id) {
            Some(&var) => match self.stream_unifier.get_normalized_type(var) {
                // make stream types default to event stream with empty conjunction, i.e., true
                Some(StreamTy::Infer(_)) | None => Some(StreamTy::Event(Activation::True)),
                Some(mut stream_ty) => {
                    stream_ty.simplify();
                    Some(stream_ty)
                }
            },
            None => Some(StreamTy::Event(Activation::True)),
        }
    }

    /// Returns the activation condition (AC) to be used in Lowering/Evaluator.
    /// The resulting AC is a Boolean Condition over Inpuut Stream References.
    pub(crate) fn get_activation_condition(
        &mut self,
        spec: &'a LolaSpec,
        id: NodeId,
    ) -> Option<Activation<crate::ir::StreamReference>> {
        let mut stream_ty = if let Some(stream_ty) = self.get_stream_type(id) {
            stream_ty
        } else {
            return None;
        };
        self.normalize_stream_ty_to_inputs(&mut stream_ty);
        match &stream_ty {
            StreamTy::Event(ac) => Some(self.translate_activation_condition(spec, ac)),
            _ => None,
        }
    }

    fn normalize_stream_ty_to_inputs(&mut self, ty: &mut StreamTy) {
        match ty {
            StreamTy::Event(ac) => self.normalize_activation_condition(ac),
            StreamTy::RealTime(_) => {}
            StreamTy::Infer(_) => unreachable!(),
        }
    }

    fn normalize_activation_condition(&mut self, ac: &mut Activation<StreamVar>) {
        match ac {
            Activation::Conjunction(args) | Activation::Disjunction(args) => {
                args.iter_mut().for_each(|arg| self.normalize_activation_condition(arg))
            }
            Activation::Stream(var) => {
                *ac = match self.stream_unifier.get_normalized_type(*var) {
                    // make stream types default to event stream with empty conjunction, i.e., true
                    Some(StreamTy::Infer(_)) | None => Activation::True,
                    Some(StreamTy::Event(Activation::Stream(v))) if *var == v => {
                        return;
                    }
                    Some(StreamTy::Event(ac)) => ac,
                    _ => unreachable!(),
                };
                self.normalize_activation_condition(ac);
            }
            Activation::True => {}
        }
    }

    fn translate_activation_condition(
        &self,
        spec: &'a LolaSpec,
        ac: &Activation<StreamVar>,
    ) -> Activation<crate::ir::StreamReference> {
        match ac {
            Activation::Conjunction(args) => {
                Activation::Conjunction(args.iter().map(|ac| self.translate_activation_condition(spec, ac)).collect())
            }
            Activation::Disjunction(args) => {
                Activation::Disjunction(args.iter().map(|ac| self.translate_activation_condition(spec, ac)).collect())
            }
            Activation::Stream(var) => {
                let idx = spec
                    .inputs
                    .iter()
                    .enumerate()
                    .find(|(_, val)| self.stream_vars[&val.id] == *var)
                    .expect("activation condition is assumed to be over input-streams only")
                    .0;
                Activation::Stream(crate::ir::StreamReference::InRef(idx))
            }
            Activation::True => Activation::True,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::analysis::naming::*;
    use crate::parse::*;
    use crate::reporting::Handler;
    use crate::ty::{FloatTy, IntTy, UIntTy};
    use std::path::PathBuf;

    fn num_type_errors(spec: &str) -> usize {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let spec = match parse(spec, &handler) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        let mut na = NamingAnalysis::new(&handler);
        let mut decl_table = na.check(&spec);
        assert!(!handler.contains_error(), "Spec produces errors in naming analysis.");
        let mut type_analysis = TypeAnalysis::new(&handler, &mut decl_table);
        type_analysis.check(&spec);
        handler.emitted_errors()
    }

    fn num_type_warnings(spec: &str) -> usize {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let spec = match parse(spec, &handler) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        let mut na = NamingAnalysis::new(&handler);
        let mut decl_table = na.check(&spec);
        assert!(!handler.contains_error(), "Spec produces errors in naming analysis.");
        let mut type_analysis = TypeAnalysis::new(&handler, &mut decl_table);
        type_analysis.check(&spec);
        handler.emitted_warnings()
    }

    /// Returns the type of the last output of the given spec
    fn get_type(spec: &str) -> ValueTy {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let spec = match parse(spec, &handler) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        let mut na = NamingAnalysis::new(&handler);
        let mut decl_table = na.check(&spec);
        assert!(!handler.contains_error(), "Spec produces errors in naming analysis.");
        let mut type_analysis = TypeAnalysis::new(&handler, &mut decl_table);
        type_analysis.check(&spec);
        type_analysis.get_type(spec.outputs.last().expect("spec needs at least one output").id)
    }

    fn type_check(spec: &str) -> TypeTable {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let spec = match parse(spec, &handler) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        let mut na = NamingAnalysis::new(&handler);
        let mut decl_table = na.check(&spec);
        assert!(!handler.contains_error(), "Spec produces errors in naming analysis.");
        let mut type_analysis = TypeAnalysis::new(&handler, &mut decl_table);
        type_analysis.check(&spec);
        type_analysis.extract_type_table(&spec)
    }

    #[test]
    fn simple_input() {
        let spec = "input i: Int8";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    #[ignore] // parametric streams need new design after syntax revision
    fn parametric_input() {
        let spec = "input i<a: Int8, b: Bool>: Int8\noutput o := i(1,false)[0].defaults(to: 42)";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn simple_const_float() {
        let spec = "constant c: Float32 := 2.1";
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
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
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
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn simple_binary_input() {
        let spec = "input i: Int8\noutput o: Int8 := 3 + i";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn simple_unary() {
        let spec = "output o: Bool := !false";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Bool);
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
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn simple_ite_compare() {
        let spec = "output e := if 1 == 0 then 0 else -1";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I32));
    }

    #[test]
    fn underspecified_ite_type() {
        let spec = "output o := if !false then 1.3 else -2.0";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Float(FloatTy::F32));
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
        let spec = "input s: String\noutput o: Bool := s[-1].defaults(to: \"\") == \"a\"";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Bool);
    }

    #[test]
    fn test_underspecified_type() {
        let spec = "output o := 2";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I32));
    }

    #[test]
    fn test_trigonometric() {
        let spec = "import math\noutput o: Float32 := sin(2.0)";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Float(FloatTy::F32));
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
    fn test_regex_function() {
        let spec = "import regex\ninput s: String\noutput o: Bool := matches(s[0], regex: r\"(a+b)\")";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Bool);
    }

    #[test]
    fn test_regex_method() {
        let spec = "import regex\ninput s: String\noutput o: Bool := s.matches(regex: r\"(a+b)\")";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Bool);
    }

    #[test]
    fn test_input_lookup() {
        let spec = "input a: UInt8\n output b: UInt8 := a";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::UInt(UIntTy::U8));
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
        assert_eq!(get_type(spec), ValueTy::UInt(UIntTy::U8));
    }

    #[test]
    fn test_stream_lookup_faulty() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_stream_lookup_dft() {
        let spec = "output a: UInt8 := 3\n output b: UInt8 := a[-1].defaults(to: 3)";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::UInt(UIntTy::U8));
    }

    #[test]
    fn test_stream_lookup_dft_fault() {
        let spec = "output a: UInt8 := 3\n output b: Bool := a[-1].defaults(to: false)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    #[ignore] // paramertic streams need new design after syntax revision
    fn test_invoke_type() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    #[ignore] // paramertic streams need new design after syntax revision
    fn test_invoke_type_two_params() {
        let spec = "input in: Int8\n output a<p1: Int8, p2: Int8>: Int8 { invoke (in, in) } := 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    #[ignore] // paramertic streams need new design after syntax revision
    fn test_invoke_type_faulty() {
        let spec = "input in: Bool\n output a<p1: Int8>: Int8 { invoke in } := 3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    #[ignore] // paramertic streams need new design after syntax revision
    fn test_extend_type() {
        let spec = "input in: Bool\n output a: Int8 { extend in } := 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    #[ignore] // paramertic streams need new design after syntax revision
    fn test_extend_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 { extend in } := 3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    #[ignore] // paramertic streams need new design after syntax revision
    fn test_terminate_type() {
        let spec = "input in: Bool\n output a: Int8 { terminate in } := 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    #[ignore] // paramertic streams need new design after syntax revision
    fn test_terminate_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 { terminate in } := 3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    #[ignore] // parametric streams need new design after syntax revision
    fn test_param_spec() {
        let spec =
            "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: Int8 := a(3)[-2].defaults(to: 1)";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    #[ignore] // parametric streams need new design after syntax revision
    fn test_param_spec_faulty() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: Int8 := a(true)[-2].defaults(to: 1)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    #[ignore] // parametric streams need new design after syntax revision
    fn test_param_spec_wrong_parameters() {
        let spec = "input in<a: Int8, b: Int8>: Int8\noutput x := in(1)[0]";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    #[ignore] // parametric streams need new design after syntax revision
    fn test_lookup_incomp() {
        let spec =
            "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: UInt8 := a(3)[2].defaults(to: 1)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_tuple() {
        let spec = "output out: (Int8, Bool) := (14, false)";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Tuple(vec![ValueTy::Int(IntTy::I8), ValueTy::Bool]));
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
        assert_eq!(get_type(spec), ValueTy::Bool);
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
        assert_eq!(get_type(spec), ValueTy::UInt(UIntTy::U8));
    }

    #[test]
    fn test_tuple_of_tuples() {
        let spec = "input in: (Int8, (UInt8, Bool))\noutput out: Int16 := in[0].0";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I16));
    }

    #[test]
    fn test_tuple_of_tuples2() {
        let spec = "input in: (Int8, (UInt8, Bool))\noutput out: Bool := in[0].1.1";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Bool);
    }

    #[test]
    fn test_window_widening() {
        let spec = "input in: Int8\n output out: Int64 @5Hz:= in.aggregate(over: 3s, using: Σ).defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_window() {
        let spec = "input in: Int8\n output out: Int8 @5Hz := in.aggregate(over: 3s, using: Σ).defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_window_untimed() {
        let spec = "input in: Int8\n output out: Int16 := in.aggregate(over: 3s, using: Σ).defaults(to: 5)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_window_faulty() {
        let spec = "input in: Int8\n output out: Bool @5Hz := in.aggregate(over: 3s, using: Σ).defaults(to: 5)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_timed() {
        let spec = "output o1: Bool @10Hz:= false\noutput o2: Bool @10Hz:= o1";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_timed_faster() {
        let spec = "output o1: Bool @20Hz := false\noutput o2: Bool @10Hz:= o1";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_timed_incompatible() {
        let spec = "output o1: Bool @3Hz := false\noutput o2: Bool@10Hz:= o1";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_timed_binary() {
        let spec = "output o1: Bool @10Hz:= false\noutput o2: Bool @10Hz:= o1 && true";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_involved() {
        let spec = "input velo: Float32\n output avg: Float64 @5Hz := velo.aggregate(over: 1h, using: avg).defaults(to: 10000.0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_rt_offset() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @1Hz := a[-1s].defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_rt_offset_regression() {
        let spec = "output a @10Hz := a.offset(by: -100ms).defaults(to: 0) + 1";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_rt_offset_skip() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @0.5Hz := a[-1s].defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }
    #[test]
    fn test_rt_offset_skip2() {
        let spec = "output a: Int8 @1Hz := 1\noutput b: Int8 @0.5Hz := a[-2s].defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_rt_offset_fail() {
        let spec = "output a: Int8 @0.5Hz := 1\noutput b: Int8 @1Hz := a[-1s].defaults(to: 0)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_noop() {
        let spec = "input x: UInt8\noutput y: UInt8 := 3.hold().defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_sync() {
        let spec = "input x: UInt8\noutput y: UInt8 := x.hold().defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_useful() {
        let spec = "input x: UInt8\noutput y: UInt8 @1Hz := x.hold().defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_casting_implicit_types() {
        let spec = "input x: UInt8\noutput y: Float32 := cast(x)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_casting_explicit_types() {
        let spec = "input x: Int32\noutput y: UInt32 := cast<Int32,UInt32>(x)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_missing_expression() {
        // should not produce an error as we want to be able to handle incomplete specs in analysis
        let spec = "input x: Bool\noutput y: Bool := \ntrigger (y || x)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn infinite_recursion_regression() {
        // this should fail in type checking as the value type of `c` cannot be determined.
        let spec = "output c := c[0].defaults(to:0)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_function_arguments_regression() {
        let spec = "input a: Int32\ntrigger a > 50";
        let type_table = type_check(spec);
        // expression `a > 50` has NodeId = 3
        assert_eq!(type_table.get_value_type(NodeId::new(3)), &ValueTy::Bool);
        assert_eq!(type_table.get_func_arg_types(NodeId::new(3)), &vec![ValueTy::Int(IntTy::I32)]);
    }

    #[test]
    fn test_conjunctive_stream_types() {
        let spec = "input a: Int32\ninput b: Int32\noutput x := a + b";
        let type_table = type_check(spec);
        // input `a` has NodeId = 0, StreamVar = 0
        // input `b` has NodeId = 2, StreamVar = 1
        // output `x` has NodeId = 4
        assert_eq!(type_table.get_value_type(NodeId::new(4)), &ValueTy::Int(IntTy::I32));
        assert_eq!(
            type_table.get_stream_type(NodeId::new(4)),
            &StreamTy::Event(Activation::Conjunction(vec![
                Activation::Stream(StreamVar::new(0)),
                Activation::Stream(StreamVar::new(1))
            ]))
        );
    }

    #[test]
    fn test_activation_condition() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a | b) := 1";
        let type_table = type_check(spec);
        // input `a` has NodeId = 0, StreamVar = 0
        // input `b` has NodeId = 2, StreamVar = 1
        // output `x` has NodeId = 4
        assert_eq!(type_table.get_value_type(NodeId::new(4)), &ValueTy::Int(IntTy::I32));
        assert_eq!(
            type_table.get_stream_type(NodeId::new(4)),
            &StreamTy::Event(Activation::Disjunction(vec![
                Activation::Stream(StreamVar::new(0)),
                Activation::Stream(StreamVar::new(1))
            ]))
        );
    }

    #[test]
    fn test_get() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a | b) := a.get().defaults(to: 0)";
        let type_table = type_check(spec);
        // input `a` has NodeId = 0, StreamVar = 0
        // input `b` has NodeId = 2, StreamVar = 1
        // output `x` has NodeId = 4
        assert_eq!(type_table.get_value_type(NodeId::new(4)), &ValueTy::Int(IntTy::I32));
        assert_eq!(
            type_table.get_stream_type(NodeId::new(4)),
            &StreamTy::Event(Activation::Disjunction(vec![
                Activation::Stream(StreamVar::new(0)),
                Activation::Stream(StreamVar::new(1))
            ]))
        );
    }

    #[test]
    fn test_no_direct_access_possible() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a | b) := a";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_warn_not_needed_hold() {
        let spec = "input a: Int32\noutput x @a := a.hold().defaults(to: 0)";
        assert_eq!(1, num_type_warnings(spec));

        let spec = "output x: Int32 @ 2Hz := 1\noutput y: Int32 @1Hz := x.hold().defaults(to: 0)";
        assert_eq!(1, num_type_warnings(spec));
    }

    #[test]
    fn test_get_not_possible() {
        // it should not be possible to use get with RealTime and EventBased streams
        let spec = "input a: Int32\noutput x: Int32 @ 1Hz := 0\noutput y:Int32 @ a := x.get().defaults(to: 0)";
        assert_eq!(1, num_type_errors(spec));
        let spec = "input a: Int32\noutput x: Int32 @ a := 0\noutput y:Int32 @ 1Hz := x.get().defaults(to: 0)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_normalization_event_streams() {
        let spec = "input a: Int32\ninput b: Int32\ninput c: Int32\noutput x := a + b\noutput y := x + x + c";
        let type_table = type_check(spec);
        //  input `a` has NodeId =  0, StreamVar = 0
        //  input `b` has NodeId =  2, StreamVar = 1
        //  input `c` has NodeId =  4, StreamVar = 2
        // output `x` has NodeId =  6, StreamVar = 3
        // output `y` has NodeId = 12, StreamVar = 4
        let stream_ty_x = type_table.get_stream_type(NodeId::new(6));
        assert_eq!(
            stream_ty_x,
            &StreamTy::Event(Activation::Conjunction(vec![
                Activation::Stream(StreamVar::new(0)),
                Activation::Stream(StreamVar::new(1))
            ]))
        );

        let stream_ty_y = type_table.get_stream_type(NodeId::new(12));
        assert_eq!(
            stream_ty_y,
            &StreamTy::Event(Activation::Conjunction(vec![
                Activation::Stream(StreamVar::new(0)),
                Activation::Stream(StreamVar::new(1)),
                Activation::Stream(StreamVar::new(2))
            ]))
        );

        use crate::ir::StreamReference;
        assert_eq!(
            type_table.get_acti_cond(NodeId::new(6)),
            &Activation::Conjunction(vec![
                Activation::Stream(StreamReference::InRef(0)),
                Activation::Stream(StreamReference::InRef(1)),
            ])
        );
        assert_eq!(
            type_table.get_acti_cond(NodeId::new(12)),
            &Activation::Conjunction(vec![
                Activation::Stream(StreamReference::InRef(0)),
                Activation::Stream(StreamReference::InRef(1)),
                Activation::Stream(StreamReference::InRef(2)),
            ])
        );
    }

    #[test]
    fn test_realtime_stream_integer_offset() {
        let spec = "output b @2Hz := b[-1].defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_realtime_stream_integer_offset_faster() {
        let spec = "output a @4Hz := 0\noutput b @2Hz := b[-1].defaults(to: 0) + a[-1].defaults(to: 0)";
        // equivalent to b[-500ms].defaults(to: 0) + a[-250ms].defaults(to: 0)
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_realtime_stream_integer_offset_incompatible() {
        let spec = "output a @3Hz := 0\noutput b @2Hz := b[-1].defaults(to: 0) + a[-1].defaults(to: 0)";
        // does not work, a[-1] is not guaranteed to exist
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_realtime_stream_integer_offset_sample_and_hold() {
        let spec = "output a @3Hz := 0\noutput b @2Hz := b[-1].defaults(to: 0) + a[-1].hold().defaults(to: 0)";
        // workaround using sample and hold
        assert_eq!(0, num_type_errors(spec));
    }
}
