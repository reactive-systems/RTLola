//! Implementation of the Hindley-Milner type inference through unification
//!
//! Relevant references:
//! * [Introduction to Hindley-Milner type inference](https://eli.thegreenplace.net/2018/type-inference/)
//! * [Unification algorithm](https://eli.thegreenplace.net/2018/unification/)
//! * [Unification in Rust](http://smallcultfollowing.com/babysteps/blog/2017/03/25/unification-in-chalk-part-1/)
//! * [Ena (union-find package)](https://crates.io/crates/ena)

use super::naming::{Declaration, DeclarationTable};
use crate::ast::LolaSpec;
use crate::ast::{
    Constant, Expression, ExpressionKind, Input, LitKind, Literal, Offset, Output, StreamInstance,
    TimeSpec, Trigger, Type, TypeKind, WindowOperation,
};
use crate::parse::NodeId;
use crate::parse::Span;
use crate::reporting::Handler;
use crate::reporting::LabeledSpan;
use crate::stdlib::{FuncDecl, MethodLookup};
use crate::ty::{Freq, StreamTy, TimingInfo, TypeConstraint, ValueTy};
use ena::unify::{
    EqUnifyValue, InPlace, InPlaceUnificationTable, Snapshot, UnificationStore, UnificationTable, UnifyKey, UnifyValue,
};
use log::{debug, trace};
use num::Signed;
use std::collections::HashMap;

pub(crate) struct TypeAnalysis<'a, 'b> {
    handler: &'b Handler,
    declarations: &'a DeclarationTable<'a>,
    method_lookup: MethodLookup,
    unifier: ValueUnifier<ValueTy>,
    stream_unifier: ValueUnifier<StreamTy>,
    /// maps `NodeId`'s to the variables used in `unifier`
    value_vars: HashMap<NodeId, ValueVar>,
    stream_vars: HashMap<NodeId, StreamVar>,
    /// maps function-like nodes (UnOp, BinOp, Func, Mathod) to the generic parameters
    generic_function_vars: HashMap<NodeId, Vec<ValueVar>>,
}

pub(crate) struct TypeTable {
    value_tt: HashMap<NodeId, ValueTy>,
    stream_tt: HashMap<NodeId, StreamTy>,
    func_tt: HashMap<NodeId, Vec<ValueTy>>,
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
}

impl<'a, 'b> TypeAnalysis<'a, 'b> {
    pub(crate) fn new(handler: &'b Handler, declarations: &'a DeclarationTable<'a>) -> TypeAnalysis<'a, 'b> {
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
        self.infer_types(spec);
        if self.handler.contains_error() {
            return None;
        }

        self.assign_types(spec);

        Some(self.extract_type_table())
    }

    fn extract_type_table(&mut self) -> TypeTable {
        let value_nids: Vec<NodeId> = self.value_vars.keys().cloned().collect();
        let vtt: HashMap<NodeId, ValueTy> = value_nids.into_iter().map(|nid| (nid, self.get_type(nid))).collect();

        // Note: If there is a `None` for a stream, `get_stream_type` would report the error.
        let stream_nids: Vec<NodeId> = self.stream_vars.keys().cloned().collect();
        let stt: HashMap<NodeId, StreamTy> =
            stream_nids.into_iter().flat_map(|nid| self.get_stream_type(nid).map(|ty| (nid, ty))).collect();

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

        TypeTable { value_tt: vtt, stream_tt: stt, func_tt }
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
        trace!("infer type for {}", constant);
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
        trace!("infer type for {}", input);

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

        if param_types.is_empty() {
            self.stream_unifier
                .unify_var_ty(stream_var, StreamTy::new(TimingInfo::Event))
                .map_err(|err| self.handle_error(err, input.name.span))
        } else {
            self.stream_unifier
                .unify_var_ty(stream_var, StreamTy::new_parametric(param_types, TimingInfo::Event))
                .map_err(|err| self.handle_error(err, input.name.span))
        }
    }

    fn infer_output(&mut self, output: &'a Output) -> Result<(), ()> {
        trace!("infer type for {}", output);

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
        if let Some(expr) = &output.extend.expr {
            if let Some(time_spec) = expr.parse_timespec() {
                let time_spec: TimeSpec = time_spec;
                frequency = Some(Freq::new(&format!("{}", time_spec), time_spec.exact_period.clone()));
            } else {
                unimplemented!("only frequency annotations are currently implemented for activation conditions");
            }
        }

        // determine whether stream is timed or event based
        let timing: TimingInfo = if let Some(f) = frequency { TimingInfo::RealTime(f) } else { TimingInfo::Event };

        if param_types.is_empty() {
            self.stream_unifier
                .unify_var_ty(stream_var, StreamTy::new(timing))
                .map_err(|err| self.handle_error(err, output.name.span))
        } else {
            self.stream_unifier
                .unify_var_ty(stream_var, StreamTy::new_parametric(param_types, timing))
                .map_err(|err| self.handle_error(err, output.name.span))
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
        if let Some(template_spec) = &output.template_spec {
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
        }

        let out_stream_var = self.stream_vars[&output.id];

        // generate constraint for expression
        self.infer_expression(&output.expression, Some(ValueTy::Infer(out_var)), StreamVarOrTy::Var(out_stream_var))
    }

    fn infer_trigger_expression(&mut self, trigger: &'a Trigger) -> Result<(), ()> {
        trace!("infer type for {}", trigger);

        // make sure that NodeId of trigger is assigned Bool
        let var = self.new_value_var(trigger.id);
        self.unifier.unify_var_ty(var, ValueTy::Bool).expect("cannot fail as `var` is fresh");

        let stream_var = self.new_stream_var(trigger.id);

        self.infer_expression(&trigger.expression, Some(ValueTy::Infer(var)), StreamVarOrTy::Var(stream_var))
    }

    fn get_constraint_for_literal(&self, lit: &Literal) -> ValueTy {
        use crate::ast::LitKind::*;
        match &lit.kind {
            Str(_) | RawStr(_) => ValueTy::String,
            Bool(_) => ValueTy::Bool,
            Numeric(val, unit) => {
                assert!(unit.is_none());
                if val.contains(".") {
                    // Floating Point
                    ValueTy::Constr(TypeConstraint::FloatingPoint)
                } else if val.starts_with("-") {
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
        stream_ty: StreamVarOrTy,
    ) -> Result<(), ()> {
        use crate::ast::ExpressionKind::*;
        trace!("infer expression {}", expr);

        let var = self.new_value_var(expr.id);
        if let Some(target_ty) = &target {
            self.unifier.unify_var_ty(var, target_ty.clone()).expect("unification cannot fail as `var` is fresh");
        }

        let stream_var = self.new_stream_var(expr.id);
        match &stream_ty {
            StreamVarOrTy::Ty(ty) => self
                .stream_unifier
                .unify_var_ty(stream_var, ty.clone())
                .expect("unification cannot fail as `var` is fresh"),
            &StreamVarOrTy::Var(var) => {
                self.stream_unifier.unify_var_var(stream_var, var).expect("unification cannot fail as `var` is fresh")
            }
        }

        match &expr.kind {
            Lit(l) => {
                // generate constraint from literal
                self.unifier
                    .unify_var_ty(var, self.get_constraint_for_literal(&l))
                    .map_err(|err| self.handle_error(err, expr.span))?;
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
            StreamAccess(_, _) => unimplemented!(),
            Offset(_, _) => unimplemented!(),
            SlidingWindowAggregation { expr: _expr, duration: _duration, aggregation: _aggregation } => {
                unimplemented!()
            }
            Ite(cond, left, right) => {
                // constraints
                // - ?cond = EventStream<bool>
                // - ?left = ?expr
                // - ?right = ?expr
                self.infer_expression(cond, Some(ValueTy::Bool), StreamVarOrTy::Var(stream_var))?;
                self.infer_expression(left, Some(ValueTy::Infer(var)), StreamVarOrTy::Var(stream_var))?;
                self.infer_expression(right, Some(ValueTy::Infer(var)), StreamVarOrTy::Var(stream_var))?;
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
            Lookup(stream, offset, aggregation) => {
                self.infer_lookup_expr(var, stream_var, expr.span, stream, offset, *aggregation)?
            }
            Default(left, right) => {
                self.infer_expression(
                    left,
                    Some(ValueTy::Option(ValueTy::Infer(var).into())),
                    StreamVarOrTy::Var(stream_var),
                )?;
                self.infer_expression(
                    right,
                    Some(ValueTy::Infer(var)),
                    StreamVarOrTy::Ty(StreamTy::new(TimingInfo::Event)),
                )?
            }
            Hold(left, right) => {
                // verify that `var` is infered
                if self.stream_unifier.get_type(stream_var).is_none() {
                    self.handler.error_with_span(
                        "Cannot use `!` operator when stream type (event or real-time) is unknown",
                        LabeledSpan::new(expr.span, "Consider removing `!` operator", true),
                    );
                    return Err(());
                }

                let new_var = self.stream_unifier.new_var();
                self.infer_expression(
                    left,
                    Some(ValueTy::Option(ValueTy::Infer(var).into())),
                    StreamVarOrTy::Var(new_var),
                )?;
                // TODO: check if new_var != stream_var => warn that sample and hold is not needed.
                self.infer_expression(
                    right,
                    Some(ValueTy::Infer(var)),
                    StreamVarOrTy::Ty(StreamTy::new(TimingInfo::Event)),
                )?
            }
            Function(_, types, params) => {
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

                let params: Vec<&Expression> = params.iter().map(|e| e.as_ref()).collect();

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
                self.infer_expression(base, None, StreamVarOrTy::Var(stream_var))?;

                if let Some(infered) = self.unifier.get_type(self.value_vars[&base.id]) {
                    debug!("{} {}", base, infered);
                    if let Some(fun_decl) = self.method_lookup.get(&infered, &name) {
                        debug!("{:?}", fun_decl);

                        for ty in types {
                            self.infer_type(ty)?;
                        }

                        let mut parameters = vec![base.as_ref()];
                        parameters.extend(params.iter().map(|e| e.as_ref()));

                        self.infer_function_application(
                            expr.id,
                            var,
                            stream_var,
                            expr.span,
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
                let mut tuples: Vec<ValueTy> = Vec::with_capacity(expressions.len());
                for element in expressions {
                    self.infer_expression(element, None, StreamVarOrTy::Var(stream_var))?;
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
                self.infer_expression(base, None, StreamVarOrTy::Var(stream_var))?;

                let ty = self.unifier.get_type(self.value_vars[&base.id]).expect("should have type at this point");
                let infered = ty.normalize_ty(&mut self.unifier);

                debug!("{} {}", base, infered);
                match infered {
                    ValueTy::Tuple(inner) => {
                        let num: usize = ident.name.parse::<usize>().expect("verify that this is checked earlier");
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

    fn infer_lookup_expr(
        &mut self,
        var: ValueVar,
        stream_var: StreamVar,
        span: Span,
        stream: &'a StreamInstance,
        offset: &'a Offset,
        window_op: Option<WindowOperation>,
    ) -> Result<(), ()> {
        match (offset, window_op) {
            (Offset::DiscreteOffset(off_expr), None) => {
                // recursion
                self.infer_expression(
                    off_expr,
                    Some(ValueTy::Constr(TypeConstraint::Integer)),
                    StreamVarOrTy::Ty(StreamTy::new(TimingInfo::Event)),
                )?;

                let negative_offset = match &off_expr.kind {
                    ExpressionKind::Lit(l) => match &l.kind {
                        //LitKind::Int(i) => i.is_negative(),
                        _ => unreachable!("offset expressions have to be integers"),
                    },
                    _ => unreachable!("offset expressions have to be literal"),
                };

                // need to derive "inner" type
                let decl = self.declarations[&stream.id];
                let decl_id: NodeId = match decl {
                    Declaration::In(input) => input.id,
                    Declaration::Out(output) => output.id,
                    _ => unreachable!(),
                };
                let decl_stream_var: StreamVar = self.stream_vars[&decl_id];
                let decl_stream_type = self.stream_unifier.get_type(decl_stream_var).unwrap();

                if decl_stream_type.parameters.len() != stream.arguments.len() {
                    self.handler.error_with_span(
                        &format!(
                            "this stream takes {} parameters but {} parameters were supplied",
                            decl_stream_type.parameters.len(),
                            stream.arguments.len()
                        ),
                        LabeledSpan::new(
                            span,
                            &format!("expected {} parameters", decl_stream_type.parameters.len()),
                            true,
                        ),
                    );
                    return Err(());
                }

                assert_eq!(decl_stream_type.parameters.len(), stream.arguments.len(),);

                let mut params = Vec::with_capacity(stream.arguments.len());
                for (arg_expr, target) in stream.arguments.iter().zip(decl_stream_type.parameters) {
                    self.infer_expression(
                        arg_expr,
                        Some(target.clone()),
                        StreamVarOrTy::Ty(StreamTy::new(TimingInfo::Event)),
                    )?;
                    params.push(target);
                }
                let target = if params.is_empty() {
                    StreamTy::new(TimingInfo::Event)
                } else {
                    StreamTy::new_parametric(params, TimingInfo::Event)
                };

                // stream type

                // ?decl_stream_var = target
                self.stream_unifier
                    .unify_var_ty(decl_stream_var, target)
                    .map_err(|err| self.handle_error(err, stream.span))?;
                // ?stream_var = Non-parametric Event
                self.stream_unifier
                    .unify_var_ty(stream_var, StreamTy::new(TimingInfo::Event))
                    .map_err(|err| self.handle_error(err, stream.span))?;

                // value type
                // constraints
                // off_expr = {integer}
                // stream = ?inner
                // if negative_offset || parametric { expr = Option<?inner> } else { expr = ?inner }
                let decl_var: ValueVar = self.value_vars[&decl_id];
                if negative_offset || !stream.arguments.is_empty() {
                    self.unifier
                        .unify_var_ty(var, ValueTy::Option(ValueTy::Infer(decl_var).into()))
                        .map_err(|err| self.handle_error(err, span))
                } else {
                    self.unifier.unify_var_var(var, decl_var).map_err(|err| self.handle_error(err, span))
                }
            }
            (Offset::RealTimeOffset(time_spec), None) => {
                assert!(stream.arguments.is_empty(), "parameterized timed streams currently not implemented");

                // need to derive "inner" type
                let decl = self.declarations[&stream.id];
                let decl_id: NodeId = match decl {
                    Declaration::In(input) => input.id,
                    Declaration::Out(output) => output.id,
                    _ => unreachable!(),
                };
                let decl_stream_var: StreamVar = self.stream_vars[&decl_id];
                let decl_stream_type = self.stream_unifier.get_type(decl_stream_var).unwrap();

                assert_eq!(
                    decl_stream_type.parameters.len(),
                    stream.arguments.len(),
                    "TODO: transform into error message"
                );
                let target = StreamTy::new(TimingInfo::RealTime(Freq::new(
                    &format!("{:?}", time_spec.period),
                    time_spec.exact_period.abs().clone(),
                )));

                // stream type

                let target_var = self.stream_unifier.new_var();
                self.stream_unifier.unify_var_ty(target_var, target).expect("cannot fail since `target_var` is fresh");

                // verify that accessed stream is compatible
                self.stream_unifier
                    .unify_var_var(target_var, decl_stream_var)
                    .map_err(|err| self.handle_error(err, stream.span))?;
                // an offset expression does not pose restrictions on `stream_var`

                // value type
                let negative_offset = time_spec.exact_period.is_negative();

                let decl_var: ValueVar = self.value_vars[&decl_id];
                if negative_offset || !stream.arguments.is_empty() {
                    self.unifier
                        .unify_var_ty(var, ValueTy::Option(ValueTy::Infer(decl_var).into()))
                        .map_err(|err| self.handle_error(err, span))
                } else {
                    self.unifier.unify_var_var(var, decl_var).map_err(|err| self.handle_error(err, span))
                }
            }
            (Offset::RealTimeOffset(_), Some(_window_op)) => {
                assert!(stream.arguments.is_empty(), "parameterized timed streams currently not implemented");

                // need to derive "inner" type
                let decl = self.declarations[&stream.id];
                let decl_id: NodeId = match decl {
                    Declaration::In(input) => input.id,
                    Declaration::Out(output) => output.id,
                    _ => unreachable!(),
                };

                // stream type
                let decl_var: StreamVar = self.stream_vars[&decl_id];
                assert!(!self.stream_unifier.get_type(decl_var).unwrap().is_parametric());
                match self.stream_unifier.get_type(stream_var).expect("type has to be known at this point").timing {
                    TimingInfo::RealTime(_) => {}
                    _ => {
                        self.handler.error_with_span(
                            "Sliding windows are only allowed in real-time streams",
                            LabeledSpan::new(span, "unexpected sliding window", true),
                        );
                        return Err(());
                    }
                }

                // value type
                let decl_var: ValueVar = self.value_vars[&decl_id];
                self.unifier
                    .unify_var_ty(var, ValueTy::Option(ValueTy::Infer(decl_var).into()))
                    .map_err(|err| self.handle_error(err, span))
            }
            _ => unreachable!("lookup {:?} {:?}", offset, window_op),
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
                self.infer_expression(parameter, Some(ty), StreamVarOrTy::Var(stream_var))?;
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
            InferError::StreamTypeMismatch(ty_l, ty_r) => {
                self.handler.error_with_span(
                    &format!("Type mismatch between `{}` and `{}`", ty_l, ty_r),
                    LabeledSpan::new(span, &format!("expected `{}`, found `{}`", ty_l, ty_r), true),
                );
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
            Some(&var) => self.stream_unifier.get_normalized_type(var),
            None => None,
        }
    }
}

/// Main data structure for the type unfication.
/// Implemented using a union-find data structure where the keys (`ValueVar`)
/// represent type variables. Those keys have associated values (`ValueVarVal`)
/// that represent whether the type variable is unbounded (`ValueVarVal::Unkown`)
/// or bounded (`ValueVarVal::Know(ty)` where `ty` has type `Ty`).
struct ValueUnifier<T: UnifiableTy> {
    /// union-find data structure representing the current state of the unification
    table: InPlaceUnificationTable<T::V>,
}

impl<T: UnifiableTy> ValueUnifier<T> {
    fn new() -> ValueUnifier<T> {
        ValueUnifier { table: UnificationTable::new() }
    }

    /// Returns type where every inference variable is substituted.
    /// If an infer variable remains, it is replaced by `ValueTy::Error`.
    fn get_normalized_type(&mut self, var: T::V) -> Option<T> {
        self.get_type(var).map(|t| t.normalize_ty(self))
    }
}

pub trait Unifier {
    type Var: UnifyKey;
    type Ty;
    type Store: UnificationStore;

    fn new_var(&mut self) -> Self::Var;
    fn get_type(&mut self, var: Self::Var) -> Option<Self::Ty>;
    fn unify_var_var(&mut self, left: Self::Var, right: Self::Var) -> InferResult;
    fn unify_var_ty(&mut self, var: Self::Var, ty: Self::Ty) -> InferResult;
    fn vars_equal(&mut self, l: Self::Var, r: Self::Var) -> bool;
    fn snapshot(&mut self) -> Snapshot<Self::Store>;
    fn commit(&mut self, ss: Snapshot<Self::Store>);
    fn rollback_to(&mut self, ss: Snapshot<Self::Store>);
}

impl<T: UnifiableTy> Unifier for ValueUnifier<T> {
    type Var = T::V;
    type Ty = T;
    type Store = InPlace<Self::Var>;

    fn new_var(&mut self) -> Self::Var {
        self.table.new_key(ValueVarVal::Unknown)
    }

    /// Unifies two variables.
    /// Cannot fail if one of them is unbounded.
    /// If both are bounded, we try to unify their types (recursively over the `Ty` type).
    /// If this fails as well, we try to coerce them, i.e., transform one type into the other.
    fn unify_var_var(&mut self, left: Self::Var, right: Self::Var) -> InferResult {
        debug!("unify var var {} {}", left, right);
        match (self.table.probe_value(left), self.table.probe_value(right)) {
            (ValueVarVal::Known(ty_l), ValueVarVal::Known(ty_r)) => {
                // if both variables have values, we try to unify them recursively
                if let Some(ty) = ty_l.equal_to(self, &ty_r) {
                    // proceed with unification
                    self.table
                        .unify_var_value(left, ValueVarVal::Concretize(ty.clone()))
                        .expect("overwrite cannot fail");
                    self.table.unify_var_value(right, ValueVarVal::Concretize(ty)).expect("overwrite cannot fail");
                } else if ty_l.coerces_with(self, &ty_r) {
                    return Ok(());
                } else {
                    return Err(ty_l.conflicts_with(ty_r));
                }
            }
            (ValueVarVal::Unknown, ValueVarVal::Known(ty)) => {
                if ty.contains_var(self, left) {
                    return Err(InferError::CyclicDependency);
                }
            }
            (ValueVarVal::Known(ty), ValueVarVal::Unknown) => {
                if ty.contains_var(self, right) {
                    return Err(InferError::CyclicDependency);
                }
            }
            _ => {}
        }

        self.table.unify_var_var(left, right)
    }

    /// Unifies a variable with a type.
    /// Cannot fail if the variable is unbounded.
    /// Prevents infinite recursion by checking if `var` appears in `ty`.
    /// Uses the same strategy to merge types as `unify_var_var` (in case `var` is bounded).
    fn unify_var_ty(&mut self, var: Self::Var, ty: Self::Ty) -> InferResult {
        debug!("unify var ty {} {}", var, ty);
        if let Some(other) = ty.is_inferred() {
            return self.unify_var_var(var, other);
        }
        assert!(ty.is_inferred().is_none(), "internal error: entered unreachable code");
        if ty.contains_var(self, var) {
            return Err(InferError::CyclicDependency);
        }
        if let ValueVarVal::Known(val) = self.table.probe_value(var) {
            if let Some(ty) = val.equal_to(self, &ty) {
                self.table.unify_var_value(var, ValueVarVal::Concretize(ty))
            } else if val.coerces_with(self, &ty) {
                return Ok(());
            } else {
                Err(val.conflicts_with(ty))
            }
        } else {
            self.table.unify_var_value(var, ValueVarVal::Known(ty))
        }
    }

    /// Returns current value of inference variable if it exists, `None` otherwise.
    fn get_type(&mut self, var: Self::Var) -> Option<Self::Ty> {
        if let ValueVarVal::Known(ty) = self.table.probe_value(var) {
            Some(ty)
        } else {
            None
        }
    }

    fn vars_equal(&mut self, l: Self::Var, r: Self::Var) -> bool {
        self.table.unioned(l, r)
    }

    fn snapshot(&mut self) -> Snapshot<Self::Store> {
        self.table.snapshot()
    }
    fn commit(&mut self, ss: Snapshot<Self::Store>) {
        self.table.commit(ss);
    }
    fn rollback_to(&mut self, ss: Snapshot<Self::Store>) {
        self.table.rollback_to(ss);
    }
}

pub trait UnifiableTy: std::marker::Sized + std::fmt::Display + Clone + std::cmp::PartialEq + std::fmt::Debug {
    type V: UnifyKey<Value = ValueVarVal<Self>> + std::fmt::Display;
    fn normalize_ty<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U) -> Self;
    fn coerces_with<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, right: &Self) -> bool;
    fn equal_to<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, right: &Self) -> Option<Self>;
    fn contains_var<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, var: Self::V) -> bool;
    fn is_inferred(&self) -> Option<Self::V>;
    fn conflicts_with(self, other: Self) -> InferError;
}

impl UnifiableTy for ValueTy {
    type V = ValueVar;

    /// Removes occurrences of inference variables by the inferred type.
    fn normalize_ty<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U) -> ValueTy {
        match self {
            ValueTy::Infer(var) => match unifier.get_type(*var) {
                None => self.clone(),
                Some(other_ty) => other_ty.normalize_ty(unifier),
            },
            ValueTy::Tuple(t) => ValueTy::Tuple(t.iter().map(|el| el.normalize_ty(unifier)).collect()),
            ValueTy::Option(ty) => ValueTy::Option(Box::new(ty.normalize_ty(unifier))),
            _ if self.is_primitive() => self.clone(),
            ValueTy::Constr(_) => self.clone(),
            ValueTy::Param(_, _) => self.clone(),
            _ => unreachable!("cannot normalize {}", self),
        }
    }

    /// Checks recursively if the `right` type can be transformed to match `self`.
    fn coerces_with<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, right: &ValueTy) -> bool {
        debug!("coerce {} {}", self, right);

        // Rule 1: Any type `T` can be coerced into `Option<T>`
        if let ValueTy::Option(ty) = self {
            // Take snapshot before checking equality to potentially rollback the side effects.
            let ss = unifier.snapshot();
            if ty.equal_to(unifier, right).is_some() {
                unifier.commit(ss);
                return true;
            } else {
                unifier.rollback_to(ss);
            }
        }

        // Rule 2: Bit-width increase is allowed
        match right {
            ValueTy::Int(lower) => match self {
                ValueTy::Int(upper) => lower <= upper,
                _ => false,
            },
            ValueTy::UInt(lower) => match self {
                ValueTy::UInt(upper) => lower <= upper,
                _ => false,
            },
            ValueTy::Float(lower) => match self {
                ValueTy::Float(upper) => lower <= upper,
                _ => false,
            },
            _ => false,
        }
    }

    /// Checks recursively if types are equal. Tries to unify type parameters if possible.
    /// Returns the unified, i.e., more concrete type if possible.
    fn equal_to<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, right: &ValueTy) -> Option<ValueTy> {
        trace!("comp {} {}", self, right);
        match (self, right) {
            (&ValueTy::Infer(l), &ValueTy::Infer(r)) => {
                if unifier.vars_equal(l, r) {
                    Some(ValueTy::Infer(l))
                } else {
                    // try to unify values
                    if unifier.unify_var_var(l, r).is_ok() {
                        Some(ValueTy::Infer(l))
                    } else {
                        None
                    }
                }
            }
            (&ValueTy::Infer(var), ty) | (ty, &ValueTy::Infer(var)) => {
                // try to unify
                if unifier.unify_var_ty(var, ty.clone()).is_ok() {
                    Some(ValueTy::Infer(var))
                } else {
                    None
                }
            }
            (ValueTy::Constr(constr_l), ValueTy::Constr(constr_r)) => {
                constr_l.conjunction(constr_r).map(|c| ValueTy::Constr(c.clone()))
            }
            (ValueTy::Constr(constr), other) => {
                if other.satisfies(constr) {
                    Some(other.clone())
                } else {
                    None
                }
            }
            (other, ValueTy::Constr(constr)) => {
                if other.satisfies(constr) {
                    Some(other.clone())
                } else {
                    None
                }
            }
            (ValueTy::Option(l), ValueTy::Option(r)) => l.equal_to(unifier, r).map(|ty| ValueTy::Option(ty.into())),
            (ValueTy::Tuple(l), ValueTy::Tuple(r)) => {
                if l.len() != r.len() {
                    return None;
                }
                let params: Vec<ValueTy> = l.iter().zip(r).flat_map(|(l, r)| l.equal_to(unifier, r)).collect();
                if params.len() != l.len() {
                    return None;
                }
                Some(ValueTy::Tuple(params))
            }
            (l, r) => {
                if l == r {
                    Some(l.clone())
                } else {
                    None
                }
            }
        }
    }

    /// Checks if `var` occurs in `self`
    fn contains_var<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, var: ValueVar) -> bool {
        trace!("check occurrence {} {}", var, self);
        match self {
            ValueTy::Infer(t) => unifier.vars_equal(var, *t),
            ValueTy::Tuple(t) => t.iter().any(|e| e.contains_var(unifier, var)),
            ValueTy::Option(t) => t.contains_var(unifier, var),
            _ => false,
        }
    }

    fn is_inferred(&self) -> Option<ValueVar> {
        if let ValueTy::Infer(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    fn conflicts_with(self, other: Self) -> InferError {
        InferError::ValueTypeMismatch(self, other)
    }
}

/// Representation of key for unification of `ValueTy`
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub struct ValueVar(u32);

// used in UnifcationTable
impl UnifyKey for ValueVar {
    type Value = ValueVarVal<ValueTy>;
    fn index(&self) -> u32 {
        self.0
    }
    fn from_index(u: u32) -> ValueVar {
        ValueVar(u)
    }
    fn tag() -> &'static str {
        "ValueVar"
    }
}

impl UnifiableTy for StreamTy {
    type V = StreamVar;

    fn normalize_ty<U: Unifier<Var = Self::V, Ty = Self>>(&self, _unifier: &mut U) -> Self {
        self.clone()
    }

    fn coerces_with<U: Unifier<Var = Self::V, Ty = Self>>(&self, _unifier: &mut U, right: &Self) -> bool {
        debug!("coerce {} {}", self, right);

        if self.is_parametric() || right.is_parametric() {
            return false;
        }

        // RealTime<freq_self> -> RealTime<freq_right> if freq_left is multiple of freq_right
        match (&self.timing, &right.timing) {
            (TimingInfo::RealTime(target), TimingInfo::RealTime(other)) => other.is_multiple_of(target),
            _ => false,
        }
    }

    fn equal_to<U: Unifier<Var = Self::V, Ty = Self>>(&self, _unifier: &mut U, right: &Self) -> Option<Self> {
        trace!("comp {} {}", self, right);
        if self == right {
            Some(self.clone())
        } else {
            None
        }
    }

    fn contains_var<U: Unifier<Var = Self::V, Ty = Self>>(&self, _unifier: &mut U, var: Self::V) -> bool {
        trace!("check occurrence {} {}", var, self);
        false
    }

    fn is_inferred(&self) -> Option<Self::V> {
        None
    }

    fn conflicts_with(self, other: Self) -> InferError {
        InferError::StreamTypeMismatch(self, other)
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ValueVarVal<Ty: UnifiableTy> {
    Known(Ty),
    Unknown,
    /// Used to overwrite known values
    Concretize(Ty),
}

/// Implements how the types are merged during unification
impl<Ty: UnifiableTy> UnifyValue for ValueVarVal<Ty> {
    type Error = InferError;

    /// the idea of the unification is to merge two types and always taking the more concrete value
    fn unify_values(left: &Self, right: &Self) -> Result<Self, InferError> {
        use self::ValueVarVal::*;
        match (left, right) {
            (Known(_), Unknown) => Ok(left.clone()),
            (Unknown, Known(_)) => Ok(right.clone()),
            (Unknown, Unknown) => Ok(Unknown),
            (Known(_), Concretize(ty)) => Ok(Known(ty.clone())),
            (Known(l), Known(r)) => {
                assert!(l == r);
                Ok(left.clone())
            }
            _ => unreachable!("unify values {:?} {:?}", left, right),
        }
    }
}

impl EqUnifyValue for ValueTy {}

impl std::fmt::Display for ValueVar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Representation of key for unification of `StreamTy`
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub struct StreamVar(u32);

impl UnifyKey for StreamVar {
    type Value = ValueVarVal<StreamTy>;
    fn index(&self) -> u32 {
        self.0
    }
    fn from_index(u: u32) -> Self {
        StreamVar(u)
    }
    fn tag() -> &'static str {
        "StreamVar"
    }
}

impl EqUnifyValue for StreamTy {}

impl std::fmt::Display for StreamVar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone)]
enum StreamVarOrTy {
    Var(StreamVar),
    Ty(StreamTy),
}

type InferResult = Result<(), InferError>;

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum InferError {
    ValueTypeMismatch(ValueTy, ValueTy),
    StreamTypeMismatch(StreamTy, StreamTy),
    ConflictingConstraint(TypeConstraint, TypeConstraint),
    CyclicDependency,
}

impl InferError {
    fn normalize_types(&mut self, unifier: &mut ValueUnifier<ValueTy>) {
        if let InferError::ValueTypeMismatch(ref mut expected, ref mut found) = self {
            std::mem::replace(expected, expected.normalize_ty(unifier));
            std::mem::replace(found, found.normalize_ty(unifier));
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

        let spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        let mut na = NamingAnalysis::new(&handler);
        let decl_table = na.check(&spec);
        assert!(!handler.contains_error(), "Spec produces errors in naming analysis.");
        let mut type_analysis = TypeAnalysis::new(&handler, &decl_table);
        type_analysis.check(&spec);
        handler.emitted_errors()
    }

    /// Returns the type of the last output of the given spec
    fn get_type(spec: &str) -> ValueTy {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        let mut na = NamingAnalysis::new(&handler);
        let decl_table = na.check(&spec);
        assert!(!handler.contains_error(), "Spec produces errors in naming analysis.");
        let mut type_analysis = TypeAnalysis::new(&handler, &decl_table);
        type_analysis.check(&spec);
        type_analysis.get_type(spec.outputs.last().expect("spec needs at least one output").id)
    }

    fn type_check(spec: &str) -> TypeTable {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        let mut na = NamingAnalysis::new(&handler);
        let decl_table = na.check(&spec);
        assert!(!handler.contains_error(), "Spec produces errors in naming analysis.");
        let mut type_analysis = TypeAnalysis::new(&handler, &decl_table);
        type_analysis.check(&spec);
        type_analysis.extract_type_table()
    }

    #[test]
    fn simple_input() {
        let spec = "input i: Int8";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
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
    fn test_trigonometric_faulty_3() {
        let spec = "import math\noutput o: Float64 := cos()";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_regex() {
        let spec = "import regex\ninput s: String\noutput o: Bool := matches_regex(s[0], r\"(a+b)\")";
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
    fn test_invoke_type() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn test_invoke_type_two_params() {
        let spec = "input in: Int8\n output a<p1: Int8, p2: Int8>: Int8 { invoke (in, in) } := 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
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
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
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
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn test_terminate_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 { terminate in } := 3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_param_spec() {
        let spec =
            "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: Int8 := a(3)[-2].defaults(to: 1)";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn test_param_spec_faulty() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: Int8 := a(true)[-2].defaults(to: 1)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_param_spec_wrong_parameters() {
        let spec = "input in<a: Int8, b: Int8>: Int8\noutput x := in(1)[0]";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
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
        let spec = "output a: Int8 @1s := 1\noutput b: Int8 @1s := a[-1s].defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_rt_offset_skip() {
        let spec = "output a: Int8 @1s := 1\noutput b: Int8 @2s := a[-1s].defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }
    #[test]
    fn test_rt_offset_skip2() {
        let spec = "output a: Int8 @1s := 1\noutput b: Int8 @2s := a[-2s].defaults(to: 0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_rt_offset_fail() {
        let spec = "output a: Int8 @2s := 1\noutput b: Int8 @1s := a[-1s].defaults(to: 0)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_noop() {
        let spec = "input x: UInt8\noutput y: UInt8 := 3 ! 0";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_sync() {
        let spec = "input x: UInt8\noutput y: UInt8 := x ! 0";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_sample_and_hold_useful() {
        let spec = "input x: UInt8\noutput y: UInt8 @1Hz := x ! 0";
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
        let spec = "output c := c[0]?0";
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
}
