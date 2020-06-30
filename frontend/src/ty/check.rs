//! Implementation of the Hindley-Milner type inference through unification
//!
//! Relevant references:
//! * [Introduction to Hindley-Milner type inference](https://eli.thegreenplace.net/2018/type-inference/)
//! * [Unification algorithm](https://eli.thegreenplace.net/2018/unification/)
//! * [Unification in Rust](http://smallcultfollowing.com/babysteps/blog/2017/03/25/unification-in-chalk-part-1/)
//! * [Ena (union-find package)](https://crates.io/crates/ena)

use super::unifier::{InferError, UnifiableTy, Unifier, ValueUnifier, ValueVar};
use super::{Activation, Freq, StreamTy, TypeConstraint, ValueTy};
use crate::analysis::naming::{Declaration, DeclarationTable};
use crate::ast::{
    BinOp, Constant, Expression, ExpressionKind, FunctionName, Input, Literal, Offset, Output, RTLolaAst,
    StreamAccessKind, Trigger, Type, TypeKind, WindowOperation,
};
use crate::parse::{NodeId, Span};
use crate::reporting::{Handler, LabeledSpan};
use crate::stdlib;
use crate::stdlib::{FuncDecl, MethodLookup};
use log::{debug, trace};
use num::traits::ops::inv::Inv;
use num::Signed;
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};
use uom::si::frequency::hertz;
use uom::si::rational64::Frequency as UOM_Frequency;
use uom::si::time::second;

pub(crate) struct TypeAnalysis<'a, 'b, 'c> {
    handler: &'b Handler,
    declarations: &'c mut DeclarationTable,
    method_lookup: MethodLookup<'a>,
    unifier: ValueUnifier<ValueTy>,
    /// maps `NodeId`'s to the variables used in `unifier`
    value_vars: HashMap<NodeId, ValueVar>,
    /// maps function-like nodes (UnOp, BinOp, Func, Method) to the generic parameters
    generic_function_vars: HashMap<NodeId, Vec<ValueVar>>,
    stream_ty: HashMap<NodeId, StreamTy>,
}

#[derive(Debug)]
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
    pub(crate) fn new(handler: &'b Handler, declarations: &'c mut DeclarationTable) -> TypeAnalysis<'a, 'b, 'c> {
        TypeAnalysis {
            handler,
            declarations,
            method_lookup: MethodLookup::new(),
            unifier: ValueUnifier::new(),
            value_vars: HashMap::new(),
            generic_function_vars: HashMap::new(),
            stream_ty: HashMap::new(),
        }
    }

    pub(crate) fn check(&mut self, spec: &'a RTLolaAst) -> Option<TypeTable> {
        self.imports(spec);
        self.infer_types(spec);
        if self.handler.contains_error() {
            return None;
        }

        self.assign_types(spec);

        let type_table = self.extract_type_table(spec);
        if self.handler.contains_error() {
            None
        } else {
            Some(type_table)
        }
    }

    fn extract_type_table(&mut self, spec: &'a RTLolaAst) -> TypeTable {
        let value_nids: Vec<NodeId> = self.value_vars.keys().cloned().collect();
        let vtt: HashMap<NodeId, ValueTy> = value_nids.into_iter().map(|nid| (nid, self.get_type(nid))).collect();
        let stt: HashMap<NodeId, StreamTy> = self.stream_ty.clone();

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

        let acti_cond = self
            .stream_ty
            .clone()
            .keys()
            .flat_map(|&nid| self.get_activation_condition(spec, nid).map(|ac| (nid, ac)))
            .collect();

        TypeTable { value_tt: vtt, stream_tt: stt, func_tt, acti_cond }
    }

    fn imports(&mut self, spec: &'a RTLolaAst) {
        stdlib::import_implicit_method(&mut self.method_lookup);
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

    fn infer_types(&mut self, spec: &'a RTLolaAst) {
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

        let mut stream_ty_inference_failed = false;
        for output in &spec.outputs {
            self.infer_output_clock(output).unwrap_or_else(|_| {
                debug!("stream type inference failed for {}", output);
                stream_ty_inference_failed = true;
            });
        }

        if stream_ty_inference_failed {
            return;
        }

        for output in &spec.outputs {
            self.concretize_output_clock(output.id).unwrap_or_else(|_| {
                debug!("stream type concretization failed for {}", output);
            });
        }
        for trigger in &spec.trigger {
            self.concretize_output_clock(trigger.id).unwrap_or_else(|_| {
                debug!("stream type concretization failed for {}", trigger);
            });
        }

        for output in &spec.outputs {
            self.check_output_clock(output).unwrap_or_else(|_| {
                debug!("stream type check failed for {}", output);
            });
        }
        for trigger in &spec.trigger {
            self.check_output_clock_expression(&self.stream_ty[&trigger.id].clone(), &trigger.expression)
                .unwrap_or_else(|_| {
                    debug!("stream type check for {}", trigger);
                });
        }
    }

    fn new_value_var(&mut self, node: NodeId) -> ValueVar {
        assert!(!self.value_vars.contains_key(&node));
        let var = self.unifier.new_var();
        self.value_vars.insert(node, var);
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
        if let Some(constraint) = self.get_constraint_for_literal(&constant.literal) {
            self.unifier.unify_var_ty(var, constraint).map_err(|err| self.handle_error(err, constant.literal.span))?;
            Ok(())
        } else {
            Err(())
        }
    }

    fn infer_input(&mut self, input: &'a Input) -> Result<(), ()> {
        trace!("infer type for {} (NodeId = {})", input, input.id);

        // value type
        let var = self.new_value_var(input.id);

        self.infer_type(&input.ty)?;
        let ty_var = self.value_vars[&input.ty.id];

        self.unifier.unify_var_var(var, ty_var).map_err(|err| self.handle_error(err, input.name.span))?;

        // stream type
        self.stream_ty.insert(input.id, StreamTy::new_event(Activation::Stream(input.id)));

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
        Ok(())
    }

    /// infers value types if given (`:` expression)
    fn infer_output(&mut self, output: &'a Output) -> Result<(), ()> {
        trace!("infer type for {} (NodeId = {})", output, output.id);

        // value type
        let var = self.new_value_var(output.id);

        // generate constraint in case a type is annotated
        self.infer_type(&output.ty)?;
        let ty_var = self.value_vars[&output.ty.id];

        self.unifier.unify_var_var(var, ty_var).map_err(|err| self.handle_error(err, output.name.span))?;

        // collect parameters
        let mut param_types = Vec::new();
        for param in &output.params {
            self.infer_type(&param.ty)?;
            let param_ty_var = self.value_vars[&param.ty.id];
            let param_var = self.new_value_var(param.id);
            self.unifier.unify_var_var(param_var, param_ty_var).expect("cannot fail as `param_var` is fresh");
            param_types.push(ValueTy::Infer(param_var));
        }

        Ok(())
    }

    /// infers stream types if given (`@` expression)
    fn infer_output_clock(&mut self, output: &'a Output) -> Result<(), ()> {
        trace!("infer output clock for {} (NodeId = {})", output, output.id);

        // check if stream has timing infos
        let frequency;
        let activation;

        if let Ok((f, a)) = self.parse_at_expression(output) {
            frequency = f;
            activation = a;
        } else {
            return Err(());
        }

        // determine whether stream is timed or event based or should be infered
        if let Some(f) = frequency {
            self.stream_ty.insert(output.id, StreamTy::new_periodic(f));
        } else if let Some(act) = activation {
            self.stream_ty.insert(output.id, StreamTy::new_event(act));
        } else {
            // stream type should be inferred
            let mut inner = Vec::new();
            if let Some(termination) = output.termination.as_ref() {
                self.infer_stream_ty_from_expression(termination, &mut inner);
            }
            self.infer_stream_ty_from_expression(&output.expression, &mut inner);
            self.stream_ty.insert(output.id, StreamTy::Infer(inner));
        }
        Ok(())
    }

    fn infer_stream_ty_from_expression(&mut self, expression: &Expression, inner: &mut Vec<NodeId>) {
        use crate::ast::ExpressionKind::*;
        match &expression.kind {
            Lit(_) => {}
            Ident(_) => {
                let decl = self.declarations[&expression.id].clone();

                match decl {
                    Declaration::Const(_) => {}
                    Declaration::In(input) => {
                        // stream type
                        inner.push(input.id)
                    }
                    Declaration::Out(output) => {
                        // stream type
                        inner.push(output.id)
                    }
                    Declaration::Param(_) => {}
                    Declaration::Type(_) | Declaration::Func(_) | Declaration::ParamOut(_) => {
                        unreachable!("ensured by naming analysis {:?}", decl)
                    }
                }
            }
            Offset(expr, _) => {
                self.infer_stream_ty_from_expression(&expr, inner);
            }
            SlidingWindowAggregation { .. } => {}
            Ite(cond, left, right) => {
                self.infer_stream_ty_from_expression(&cond, inner);
                self.infer_stream_ty_from_expression(&left, inner);
                self.infer_stream_ty_from_expression(&right, inner);
            }
            Unary(_, appl) => {
                self.infer_stream_ty_from_expression(&appl, inner);
            }
            Binary(_, left, right) => {
                self.infer_stream_ty_from_expression(&left, inner);
                self.infer_stream_ty_from_expression(&right, inner);
            }
            Default(left, right) => {
                self.infer_stream_ty_from_expression(&left, inner);
                self.infer_stream_ty_from_expression(&right, inner);
            }
            StreamAccess(_, _) => {
                // inner does not influence stream ty
            }
            Function(_name, _, params) => {
                for param in params {
                    self.infer_stream_ty_from_expression(&param, inner);
                }
            }
            Method(base, _, _, params) => {
                self.infer_stream_ty_from_expression(&base, inner);
                for param in params {
                    self.infer_stream_ty_from_expression(&param, inner);
                }
            }
            Tuple(expressions) => {
                for element in expressions {
                    self.infer_stream_ty_from_expression(&element, inner);
                }
            }
            Field(base, _) => {
                self.infer_stream_ty_from_expression(&base, inner);
            }
            ParenthesizedExpression(_, expr, _) => {
                self.infer_stream_ty_from_expression(&expr, inner);
            }
            MissingExpression => {
                // we simply ignore missing expressions and continue with type analysis
            }
        }
    }

    /// computes actual type from inferred stream vars
    fn concretize_output_clock(&mut self, node_id: NodeId) -> Result<(), ()> {
        trace!("concretize and normalize for (NodeId = {})", node_id);

        let mut concretized = match &self.stream_ty[&node_id] {
            StreamTy::RealTime(_) => {
                // already conrete and normalized
                self.stream_ty[&node_id].clone()
            }
            StreamTy::Event(_) => {
                // concrete but not normalized
                self.stream_ty[&node_id].clone()
            }
            StreamTy::Infer(_) => {
                // neither concrete nor normalized
                match self.concretize_stream_ty(&self.stream_ty[&node_id].clone()) {
                    Some(ty) => ty,
                    None => return Err(()),
                }
            }
        };
        concretized.simplify();

        self.stream_ty.insert(node_id, concretized);

        Ok(())
    }

    /// Returns a vector of (non-inferred) StreamTy from a stream variable
    ///
    /// Precondition: every stream variable contained has a StreamTy
    /// In order to prevent infinite recursion, `seen_vars` is used to record already seen `var`s
    fn stream_var_to_vec_stream_ty(&mut self, node_id: NodeId, seen_vars: &mut HashSet<NodeId>) -> Vec<StreamTy> {
        match &self.stream_ty[&node_id].clone() {
            StreamTy::Infer(vars) => vars
                .iter()
                .flat_map(|&svar| {
                    if seen_vars.insert(svar) {
                        Some(self.stream_var_to_vec_stream_ty(svar, seen_vars))
                    } else {
                        None
                    }
                })
                .flatten()
                .collect(),
            t => vec![t.clone()],
        }
    }

    /// Transates StreamTy::Infer to conrete form
    fn concretize_stream_ty(&mut self, stream_ty: &StreamTy) -> Option<StreamTy> {
        match stream_ty {
            StreamTy::RealTime(_) | StreamTy::Event(_) => Some(stream_ty.clone()),
            StreamTy::Infer(vars) => {
                let mut seen_vars = HashSet::new();
                let stream_types: Vec<StreamTy> =
                    vars.iter().map(|&svar| self.stream_var_to_vec_stream_ty(svar, &mut seen_vars)).flatten().collect();
                if stream_types.is_empty() {
                    return Some(StreamTy::Event(Activation::True));
                }
                let mut result_ty = None;
                for stream_ty in stream_types {
                    result_ty = match (result_ty.as_mut(), &stream_ty) {
                        (None, ty) => Some(ty.clone()),
                        (Some(StreamTy::Event(l)), StreamTy::Event(r)) => Some(StreamTy::Event(l.conjunction(r))),
                        (Some(StreamTy::RealTime(l)), StreamTy::RealTime(r)) => {
                            Some(StreamTy::RealTime(l.conjunction(r)))
                        }
                        (Some(_), _) => {
                            // Incompatible types
                            return None;
                        }
                    }
                }
                result_ty
            }
        }
    }

    /// check stream types of expression match the inferred/annotated stream_ty
    fn check_output_clock(&mut self, output: &'a Output) -> Result<(), ()> {
        trace!("check output clock for {} (NodeId = {})", output, output.id);

        if let StreamTy::Infer(_) = self.stream_ty[&output.id] {
            unreachable!("stream types should be concrete at this point");
        }

        if let Some(termination) = output.termination.as_ref() {
            self.check_output_clock_expression(&self.stream_ty[&output.id].clone(), termination)?;
        }

        self.check_output_clock_expression(&self.stream_ty[&output.id].clone(), &output.expression)
    }

    fn check_output_clock_expression(&mut self, stream_ty: &StreamTy, expr: &'a Expression) -> Result<(), ()> {
        use crate::ast::ExpressionKind::*;

        match &expr.kind {
            Lit(_) => {}
            Ident(_) => {
                let decl = self.declarations[&expr.id].clone();

                match decl {
                    Declaration::Const(_) => {}
                    Declaration::In(input) => {
                        // stream type
                        let in_ty = &self.stream_ty[&input.id];
                        self.check_stream_types_are_compatible(stream_ty, in_ty, expr.span)?;
                    }
                    Declaration::Out(output) => {
                        // stream type
                        let out_ty = &self.stream_ty[&output.id];
                        self.check_stream_types_are_compatible(stream_ty, out_ty, expr.span)?;
                    }
                    Declaration::Param(_) => {}
                    Declaration::Type(_) | Declaration::Func(_) | Declaration::ParamOut(_) => {
                        unreachable!("ensured by naming analysis {:?}", decl)
                    }
                }
            }
            Offset(inner, offset) => self.check_offset_expr(stream_ty, expr.span, inner, offset)?,
            SlidingWindowAggregation { expr: inner, duration, aggregation, .. } => {
                self.check_sliding_window_expression(stream_ty, expr.span, inner, duration, *aggregation)?;
            }
            Ite(cond, left, right) => {
                self.check_output_clock_expression(stream_ty, cond)?;
                self.check_output_clock_expression(stream_ty, left)?;
                self.check_output_clock_expression(stream_ty, right)?;
            }
            Unary(_, appl) => {
                self.check_output_clock_expression(stream_ty, appl)?;
            }
            Binary(_, left, right) => {
                self.check_output_clock_expression(stream_ty, left)?;
                self.check_output_clock_expression(stream_ty, right)?;
            }
            Default(left, right) => {
                self.check_output_clock_expression(stream_ty, left)?;
                self.check_output_clock_expression(stream_ty, right)?;
            }
            StreamAccess(inner, access_type) => {
                let inner_ty = match self.declarations[&inner.id].clone() {
                    Declaration::In(input) => &self.stream_ty[&input.id],
                    Declaration::Out(output) => &self.stream_ty[&output.id],
                    _ => unreachable!(),
                };

                let function = match access_type {
                    StreamAccessKind::Sync => unreachable!("only used in IR after lowering"),
                    StreamAccessKind::Hold => "hold()",
                    StreamAccessKind::Optional => "get()",
                };

                // check that stream types are not compatible (otherwise one can use stream directly)
                match (stream_ty, inner_ty) {
                    (StreamTy::Event(left), StreamTy::Event(right)) => {
                        if left.implies_valid(&right) {
                            self.handler.warn_with_span(
                                &format!("Unnecessary `.{}`", function),
                                LabeledSpan::new(expr.span, &format!("remove `.{}`", function), true),
                            )
                        }
                    }
                    (StreamTy::RealTime(left), StreamTy::RealTime(right)) => {
                        if right.is_multiple_of(&left) == Ok(true) {
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
            Function(_name, _, params) => {
                for param in params {
                    self.check_output_clock_expression(stream_ty, param)?;
                }
            }
            Method(base, _, _, params) => {
                // recursion
                self.check_output_clock_expression(stream_ty, base)?;
                for param in params {
                    self.check_output_clock_expression(stream_ty, param)?;
                }
            }
            Tuple(expressions) => {
                for expression in expressions {
                    self.check_output_clock_expression(stream_ty, expression)?;
                }
            }
            Field(base, _) => {
                self.check_output_clock_expression(stream_ty, base)?;
            }
            ParenthesizedExpression(_, expr, _) => {
                self.check_output_clock_expression(stream_ty, expr)?;
            }
            MissingExpression => {
                // we simply ignore missing expressions and continue with type analysis
            }
        }
        Ok(())
    }

    fn check_offset_expr(
        &mut self,
        stream_ty: &StreamTy,
        span: Span,
        expr: &'a Expression,
        offset: &'a Offset,
    ) -> Result<(), ()> {
        // check if offset is discrete or time-based
        match offset {
            Offset::Discrete(_) => self.check_output_clock_expression(stream_ty, expr),
            Offset::RealTime(_, _) => {
                // get frequency
                let time = offset.to_uom_time().expect("guaranteed to be real-time");
                let freq = UOM_Frequency::new::<hertz>(time.get::<second>().abs().inv());

                // target stream type
                let target_stream_ty = StreamTy::new_periodic(Freq::new(freq));

                // recursion
                // stream types have to match
                if let ExpressionKind::Ident(_) = &expr.kind {
                    let decl = self.declarations[&expr.id].clone();

                    let id = match decl {
                        Declaration::In(input) => input.id,
                        Declaration::Out(output) => output.id,
                        _ => unreachable!("ensured by naming analysis {:?}", decl),
                    };
                    self.check_stream_types_are_compatible(stream_ty, &self.stream_ty[&id], span)?;
                    self.check_stream_types_are_compatible(&target_stream_ty, &self.stream_ty[&id], span)
                } else {
                    unreachable!();
                }
            }
        }
    }

    fn check_sliding_window_expression(
        &mut self,
        stream_ty: &StreamTy,
        span: Span,
        _expr: &'a Expression,
        duration: &'a Expression,
        _window_op: WindowOperation,
    ) -> Result<(), ()> {
        // the stream variable has to be real-time
        let _f = match stream_ty {
            StreamTy::RealTime(f) => f,
            _ => {
                self.handler.error_with_span(
                    "Sliding windows are only allowed in real-time streams",
                    LabeledSpan::new(span, "unexpected sliding window", true),
                );
                return Err(());
            }
        };

        // check duration
        let _duration = match duration.parse_duration() {
            Err(message) => {
                self.handler.error_with_span("expected duration", LabeledSpan::new(duration.span, &message, true));
                return Err(());
            }
            Ok(d) => d,
        };
        /*if duration < f.period {
            self.handler
                .warn_with_span("period is smaller than duration in sliding window", LabeledSpan::new(span, "", true));
        }*/
        Ok(())
    }

    /// Produeces user facing error message in case the stream types are incompatible
    fn check_stream_types_are_compatible(&self, left: &StreamTy, right: &StreamTy, span: Span) -> Result<(), ()> {
        match left.is_valid(right) {
            Ok(true) => Ok(()),
            Ok(false) => {
                self.handler.error_with_span(
                    "stream types are incompatible",
                    LabeledSpan::new(span, &format!("expected `{}`, found `{}`", left, right), true),
                );
                Err(())
            }
            Err(s) => {
                self.handler.error_with_span("stream types are incompatible", LabeledSpan::new(span, s.as_ref(), true));
                Err(())
            }
        }
    }

    fn parse_at_expression(&mut self, output: &'a Output) -> Result<(Option<Freq>, Option<Activation<NodeId>>), ()> {
        if let Some(expr) = &output.extend.expr {
            match &expr.kind {
                ExpressionKind::Lit(_) => match expr.parse_freqspec() {
                    Ok(f) => Ok((Some(Freq::new(f)), None)),
                    Err(s) => {
                        self.handler.error_with_span(&s, LabeledSpan::new(expr.span, "", true));
                        Err(())
                    }
                },
                _ => match self.parse_activation_condition(output.id, expr) {
                    Ok(act) => Ok((None, Some(act))),
                    Err(_) => Err(()),
                },
            }
        } else {
            Ok((None, None))
        }
    }

    fn parse_activation_condition(&mut self, out_id: NodeId, expr: &Expression) -> Result<Activation<NodeId>, ()> {
        match &expr.kind {
            ExpressionKind::Ident(_) => match self.declarations[&expr.id].clone() {
                Declaration::In(input) => Ok(Activation::Stream(input.id)),
                Declaration::Out(output) => {
                    if output.id == out_id {
                        self.handler.error_with_span(
                            "self-references are not allowed in activation conditions",
                            LabeledSpan::new(expr.span, "", true),
                        );
                        Err(())
                    } else {
                        Ok(Activation::Stream(output.id))
                    }
                }
                _ => {
                    self.handler.error_with_span("expected stream", LabeledSpan::new(expr.span, "", true));
                    Err(())
                }
            },
            ExpressionKind::Binary(op, left, right) => {
                let (left, right) = match (
                    self.parse_activation_condition(out_id, left),
                    self.parse_activation_condition(out_id, right),
                ) {
                    (Ok(left), Ok(right)) => (left, right),
                    (Err(_), _) => return Err(()),
                    (_, Err(_)) => return Err(()),
                };
                match op {
                    BinOp::And => Ok(Activation::Conjunction(vec![left, right])),
                    BinOp::Or => Ok(Activation::Disjunction(vec![left, right])),
                    _ => {
                        self.handler.error_with_span(
                            "only disjunctions and conjunctions are allowed for activation conditions",
                            LabeledSpan::new(expr.span, "", true),
                        );
                        Err(())
                    }
                }
            }
            ExpressionKind::ParenthesizedExpression(_, expr, _) => self.parse_activation_condition(out_id, expr),
            _ => {
                self.handler.error_with_span(
                    "only variables, disjunctions, and conjunctions are allowed for activation conditions",
                    LabeledSpan::new(expr.span, "", true),
                );
                Err(())
            }
        }
    }

    fn infer_type(&mut self, ast_ty: &'a Type) -> Result<ValueVar, ()> {
        trace!("infer type for {}", ast_ty);
        let ty_var = self.new_value_var(ast_ty.id);
        match &ast_ty.kind {
            TypeKind::Inferred => {}
            TypeKind::Simple(_) => {
                match self.declarations[&ast_ty.id].clone() {
                    Declaration::Type(ty) => {
                        // ?ty_var = `ty`
                        self.unifier.unify_var_ty(ty_var, (*ty).clone()).expect("cannot fail as `ty_var` is fresh");
                    }
                    _ => unreachable!("ensured by naming analysis"),
                }
            }
            TypeKind::Tuple(tuple) => {
                let inner: Vec<ValueTy> =
                    tuple.iter().map(|ele| self.infer_type(ele).unwrap()).map(ValueTy::Infer).collect();
                let ty = ValueTy::Tuple(inner);
                // ?ty_var = `ty`
                self.unifier.unify_var_ty(ty_var, ty).expect("cannot fail as `ty_var` is fresh");
            }
            TypeKind::Optional(ty) => {
                self.infer_type(ty)?;
                let inner = self.value_vars[&ty.id];
                // ?ty_var = `ty?`
                self.unifier
                    .unify_var_ty(ty_var, ValueTy::Option(ValueTy::Infer(inner).into()))
                    .expect("cannot fail as `ty_var` is fresh");
            }
        }
        Ok(ty_var)
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

        if let Some(terminate) = output.termination.as_ref() {
            // check that condition is boolean
            self.infer_expression(&terminate, Some(ValueTy::Bool))?;
        }

        // generate constraint for expression
        self.infer_expression(&output.expression, Some(ValueTy::Infer(out_var)))
    }

    fn infer_trigger_expression(&mut self, trigger: &'a Trigger) -> Result<(), ()> {
        trace!("infer type for {}", trigger);

        // for triggers, there are no stream type annotations, so just infer
        let mut inner = Vec::new();
        self.infer_stream_ty_from_expression(&trigger.expression, &mut inner);
        self.stream_ty.insert(trigger.id, StreamTy::Infer(inner));

        // make sure that NodeId of trigger is assigned Bool
        let var = self.new_value_var(trigger.id);
        self.unifier.unify_var_ty(var, ValueTy::Bool).expect("cannot fail as `var` is fresh");

        self.infer_expression(&trigger.expression, Some(ValueTy::Infer(var)))
    }

    fn get_constraint_for_literal(&self, lit: &Literal) -> Option<ValueTy> {
        use crate::ast::LitKind::*;
        Some(match &lit.kind {
            Str(_) | RawStr(_) => ValueTy::String,
            Bool(_) => ValueTy::Bool,
            Numeric(val, unit) => {
                if let Some(unit) = unit {
                    self.handler.error_with_span(
                        &format!("unexpected unit `{}`", unit),
                        LabeledSpan::new(lit.span, "remove unit from numeric value", true),
                    );
                    return None;
                }
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
        })
    }

    fn infer_expression(&mut self, expr: &'a Expression, target: Option<ValueTy>) -> Result<(), ()> {
        use crate::ast::ExpressionKind::*;
        trace!("infer expression {} (NodeId = {})", expr, expr.id);

        let var = self.new_value_var(expr.id);
        if let Some(target_ty) = &target {
            self.unifier.unify_var_ty(var, target_ty.clone()).expect("unification cannot fail as `var` is fresh");
        }

        match &expr.kind {
            Lit(l) => {
                // generate value type constraint from literal
                if let Some(constraint) = self.get_constraint_for_literal(&l) {
                    self.unifier.unify_var_ty(var, constraint).map_err(|err| self.handle_error(err, expr.span))?;
                } else {
                    return Err(());
                }

                // no stream type constraint is needed
            }
            Ident(_) => {
                let decl = self.declarations[&expr.id].clone();

                match decl {
                    Declaration::Const(constant) => {
                        let const_var = self.value_vars[&constant.id];
                        self.unifier.unify_var_var(var, const_var).map_err(|err| self.handle_error(err, expr.span))?
                    }
                    Declaration::In(input) => {
                        // value type
                        let in_var = self.value_vars[&input.id];
                        self.unifier.unify_var_var(var, in_var).map_err(|err| self.handle_error(err, expr.span))?;
                    }
                    Declaration::Out(output) => {
                        // value type
                        let out_var = self.value_vars[&output.id];
                        self.unifier.unify_var_var(var, out_var).map_err(|err| self.handle_error(err, expr.span))?;
                    }
                    Declaration::Param(param) => {
                        // value type only
                        let param_var = self.value_vars[&param.id];
                        self.unifier.unify_var_var(var, param_var).map_err(|err| self.handle_error(err, expr.span))?;
                    }
                    Declaration::Type(_) | Declaration::Func(_) | Declaration::ParamOut(_) => {
                        unreachable!("ensured by naming analysis {:?}", decl)
                    }
                }
            }
            Offset(inner, offset) => self.infer_offset_expr(var, expr.span, inner, offset)?,
            SlidingWindowAggregation { expr: inner, duration, wait, aggregation } => {
                self.infer_sliding_window_expression(var, expr.span, inner, duration, *wait, *aggregation)?;
            }
            Ite(cond, left, right) => {
                // value type constraints
                // * `cond` = Bool
                // * `left` = `var`
                // * `right` = `var`
                // stream type constraints: all should be the same
                self.infer_expression(cond, Some(ValueTy::Bool))?;
                self.infer_expression(left, Some(ValueTy::Infer(var)))?;
                self.infer_expression(right, Some(ValueTy::Infer(var)))?;
            }
            Unary(op, appl) => {
                self.infer_function_application(expr.id, var, expr.span, &op.get_func_decl(), &[], &[appl])?;
            }
            Binary(op, left, right) => {
                self.infer_function_application(expr.id, var, expr.span, &op.get_func_decl(), &[], &[left, right])?;
            }
            Default(left, right) => {
                self.infer_expression(left, Some(ValueTy::Option(ValueTy::Infer(var).into())))?;
                self.infer_expression(right, Some(ValueTy::Infer(var)))?
            }
            StreamAccess(inner, _) => {
                // result type is an optional value
                let target_var = self.unifier.new_var();
                self.unifier
                    .unify_var_ty(var, ValueTy::Option(ValueTy::Infer(target_var).into()))
                    .map_err(|err| self.handle_error(err, inner.span))?;

                // the stream type of `inner` is unconstrained
                self.infer_expression(inner, Some(ValueTy::Infer(target_var)))?;
            }
            Function(_name, types, params) => {
                let decl = self.declarations[&expr.id].clone();
                let fun_decl = match decl {
                    Declaration::Func(fun_decl) => fun_decl,
                    Declaration::ParamOut(out) => {
                        // create matching function declaration
                        assert!(!out.params.is_empty());
                        Rc::new(FuncDecl {
                            name: FunctionName { name: out.name.clone(), arg_names: vec![None; out.params.len()] },
                            generics: Vec::new(),
                            parameters: out
                                .params
                                .iter()
                                .map(|param| ValueTy::Infer(self.value_vars[&param.ty.id]))
                                .collect(),
                            return_type: ValueTy::Infer(self.value_vars[&out.id]),
                        })
                    }
                    _ => unreachable!("ensured by naming analysis"),
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

                for ty in types {
                    self.infer_type(ty)?;
                }

                let params: Vec<&Expression> = params.iter().map(Box::as_ref).collect();

                self.infer_function_application(
                    expr.id,
                    var,
                    expr.span,
                    &fun_decl,
                    types.as_slice(),
                    params.as_slice(),
                )?;
            }
            Method(base, name, types, params) => {
                // recursion
                self.infer_expression(base, None)?;

                if let Some(inferred) = self.unifier.get_type(self.value_vars[&base.id]) {
                    debug!("{} {}", base, inferred);
                    if let Some(fun_decl) = self.method_lookup.get(&inferred, &name) {
                        debug!("{:?}", fun_decl);

                        for ty in types {
                            self.infer_type(ty)?;
                        }

                        let parameters: Vec<_> = std::iter::once(base).chain(params).map(Box::as_ref).collect();

                        self.infer_function_application(
                            expr.id,
                            var,
                            expr.span,
                            fun_decl,
                            types.as_slice(),
                            parameters.as_slice(),
                        )?;

                        self.declarations.insert(expr.id, Declaration::Func(Rc::new(fun_decl.clone())));
                    } else {
                        self.handler.error_with_span(
                            &format!("unknown method `{}`", name),
                            LabeledSpan::new(expr.span, &format!("no method `{}` for `{}`", name, inferred), true),
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
                    self.infer_expression(element, None)?;
                    let inner = ValueTy::Infer(self.value_vars[&element.id]);
                    tuples.push(inner);
                }
                // ?var = Tuple(?expr1, ?expr2, ..)
                self.unifier
                    .unify_var_ty(var, ValueTy::Tuple(tuples))
                    .map_err(|err| self.handle_error(err, expr.span))?;
            }
            Field(base, ident) => {
                // recursion
                self.infer_expression(base, None)?;

                let ty = match self.unifier.get_type(self.value_vars[&base.id]) {
                    Some(ty) => ty,
                    None => {
                        // could not determine type, thus, there was a previous error
                        return Err(());
                    }
                };
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
            ParenthesizedExpression(_, expr, _) => {
                self.infer_expression(expr, target)?;
                self.unifier
                    .unify_var_var(var, self.value_vars[&expr.id])
                    .map_err(|err| self.handle_error(err, expr.span))?;
            }
            MissingExpression => {
                // we simply ignore missing expressions and continue with type analysis
            }
        }
        Ok(())
    }

    fn infer_offset_expr(
        &mut self,
        var: ValueVar,
        span: Span,
        expr: &'a Expression,
        offset: &'a Offset,
    ) -> Result<(), ()> {
        // check if offset is discrete or time-based
        match offset {
            Offset::Discrete(offset) => {
                // TODO: Discuss how to handle positive offsets in terms of timing constraints.
                let sync_access = *offset == 0;

                // result type is an optional value if offset is negative
                let target_var = self.unifier.new_var();
                if sync_access {
                    self.unifier.unify_var_var(var, target_var).map_err(|err| self.handle_error(err, span))?;
                } else {
                    self.unifier
                        .unify_var_ty(var, ValueTy::Option(ValueTy::Infer(target_var).into()))
                        .map_err(|err| self.handle_error(err, span))?;
                }

                // As the recursion checks that the stream types match, any integer offset will match as well.
                self.infer_expression(expr, Some(ValueTy::Infer(target_var)))
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

                // recursion
                // stream types have to match
                self.infer_expression(expr, Some(ValueTy::Infer(target_value_var)))
            }
        }
    }

    fn infer_sliding_window_expression(
        &mut self,
        var: ValueVar,
        span: Span,
        expr: &'a Expression,
        duration: &'a Expression,
        wait: bool,
        window_op: WindowOperation,
    ) -> Result<(), ()> {
        // check duration
        if let Err(message) = duration.parse_duration() {
            self.handler.error_with_span("expected duration", LabeledSpan::new(duration.span, &message, true));
        }

        // value type depends on the aggregation function
        // stream type is not restricted
        use WindowOperation::*;
        match window_op {
            Count => {
                // The value type of the inner stream is not restricted
                self.infer_expression(expr, None)?;
                // resulting type is an unsigned integer value, optional if wait
                let inner_ty = ValueTy::Constr(TypeConstraint::UnsignedInteger);
                if wait {
                    self.unifier
                        .unify_var_ty(var, ValueTy::Option(inner_ty.into()))
                        .map_err(|err| self.handle_error(err, span))
                } else {
                    self.unifier.unify_var_ty(var, inner_ty).map_err(|err| self.handle_error(err, span))
                }
            }
            Sum | Product => {
                // The value type of the inner stream has to be numeric
                let ss = self.unifier.snapshot();
                match self.infer_expression(expr, Some(ValueTy::Constr(TypeConstraint::Numeric))) {
                    Ok(()) => self.unifier.commit(ss),
                    Err(()) => {
                        self.unifier.rollback_to(ss);
                        self.infer_expression(expr, Some(ValueTy::Bool))?
                    }
                }
                // resulting type depends on the inner type, optional if wait
                let inner_var = self.value_vars[&expr.id];
                if wait {
                    self.unifier
                        .unify_var_ty(var, ValueTy::Option(ValueTy::Infer(inner_var).into()))
                        .map_err(|err| self.handle_error(err, span))
                } else {
                    self.unifier
                        .unify_var_ty(var, ValueTy::Infer(inner_var))
                        .map_err(|err| self.handle_error(err, span))
                }
            }
            Min | Max | Average => {
                // The value type of the inner stream has to be numeric
                self.infer_expression(expr, Some(ValueTy::Constr(TypeConstraint::Numeric)))?;
                // resulting type depends on the inner type
                let inner_var = self.value_vars[&expr.id];
                self.unifier
                    .unify_var_ty(var, ValueTy::Option(ValueTy::Infer(inner_var).into()))
                    .map_err(|err| self.handle_error(err, span))
            }
            Disjunction | Conjunction => {
                // The value type of the inner stream has to be boolean
                self.infer_expression(expr, Some(ValueTy::Bool))?;
                // resulting type is boolean as well.
                let inner_ty = ValueTy::Bool;
                if wait {
                    self.unifier
                        .unify_var_ty(var, ValueTy::Option(inner_ty.into()))
                        .map_err(|err| self.handle_error(err, span))
                } else {
                    self.unifier.unify_var_ty(var, inner_ty).map_err(|err| self.handle_error(err, span))
                }
            }
            Integral => {
                // The value type of the inner stream has to be numeric
                self.infer_expression(expr, Some(ValueTy::Constr(TypeConstraint::Numeric)))?;
                // resulting type is floating point, optional if wait
                let inner_ty = ValueTy::Constr(TypeConstraint::FloatingPoint);
                if wait {
                    self.unifier
                        .unify_var_ty(var, ValueTy::Option(inner_ty.into()))
                        .map_err(|err| self.handle_error(err, span))
                } else {
                    self.unifier.unify_var_ty(var, inner_ty).map_err(|err| self.handle_error(err, span))
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn infer_function_application(
        &mut self,
        node_id: NodeId,
        var: ValueVar,
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
                match &gen {
                    ValueTy::Constr(_) => {}
                    _ => unreachable!("function declarations are not user-definable and currently, only constraints are allowed for generic types"),
                }
                self.unifier.unify_var_ty(var, gen.clone()).expect("cannot fail as var is freshly created");
                var
            })
            .collect();
        for (provided_type, &generic) in types.iter().zip(&generics) {
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
                self.infer_expression(parameter, Some(ty))?;
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

    /// Assigns types as infered
    fn assign_types(&mut self, spec: &RTLolaAst) {
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
            self.check_literal_sizes(&output.expression);
        }
        for trigger in &spec.trigger {
            self.check_literal_sizes(&trigger.expression);
        }
    }

    /// Check if literals fit the infered bit-width
    fn check_literal_sizes(&mut self, expression: &Expression) {
        use crate::ast::LitKind::*;
        use crate::ty::{IntTy, UIntTy};
        expression.iter().for_each(|e| {
            if let ExpressionKind::Lit(l) = &e.kind {
                if let Numeric(val, unit) = &l.kind {
                    if !unit.is_none() {
                        return;
                    }
                    if !val.contains('.') {
                        // integer
                        match self.get_type(e.id) {
                            ValueTy::Int(IntTy::I8) if val.parse::<i8>().is_err() => {
                                self.handler.error_with_span(
                                    "literal out of range for `Int8`",
                                    LabeledSpan::new(e.span, "", true),
                                );
                            }
                            ValueTy::Int(IntTy::I16) if val.parse::<i16>().is_err() => {
                                self.handler.error_with_span(
                                    "literal out of range for `Int16`",
                                    LabeledSpan::new(e.span, "", true),
                                );
                            }
                            ValueTy::Int(IntTy::I32) if val.parse::<i32>().is_err() => {
                                self.handler.error_with_span(
                                    "literal out of range for `Int32`",
                                    LabeledSpan::new(e.span, "", true),
                                );
                            }
                            ValueTy::Int(IntTy::I64) if val.parse::<i64>().is_err() => {
                                self.handler.error_with_span(
                                    "literal out of range for `Int64`",
                                    LabeledSpan::new(e.span, "", true),
                                );
                            }
                            ValueTy::UInt(UIntTy::U8) if val.parse::<u8>().is_err() => {
                                self.handler.error_with_span(
                                    "literal out of range for `UInt8`",
                                    LabeledSpan::new(e.span, "", true),
                                );
                            }
                            ValueTy::UInt(UIntTy::U16) if val.parse::<u16>().is_err() => {
                                self.handler.error_with_span(
                                    "literal out of range for `UInt16`",
                                    LabeledSpan::new(e.span, "", true),
                                );
                            }
                            ValueTy::UInt(UIntTy::U32) if val.parse::<u32>().is_err() => {
                                self.handler.error_with_span(
                                    "literal out of range for `UInt32`",
                                    LabeledSpan::new(e.span, "", true),
                                );
                            }
                            ValueTy::UInt(UIntTy::U64) if val.parse::<u64>().is_err() => {
                                self.handler.error_with_span(
                                    "literal out of range for `UInt64`",
                                    LabeledSpan::new(e.span, "", true),
                                );
                            }
                            _ => {}
                        }
                    }
                }
            }
        })
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

    /// Returns the activation condition (AC) to be used in Lowering/Evaluator.
    /// The resulting AC is a Boolean Condition over Inpuut Stream References.
    pub(crate) fn get_activation_condition(
        &mut self,
        spec: &'a RTLolaAst,
        id: NodeId,
    ) -> Option<Activation<crate::ir::StreamReference>> {
        if !self.stream_ty.contains_key(&id) {
            return None;
        }
        let mut stream_ty = self.stream_ty[&id].clone();
        stream_ty.simplify();
        let mut stream_vars = HashSet::new();
        stream_vars.insert(id);
        if !self.normalize_stream_ty_to_inputs(&mut stream_ty, &mut stream_vars) {
            return None;
        }
        match &stream_ty {
            StreamTy::Event(ac) => self.translate_activation_condition(spec, ac),
            _ => None,
        }
    }

    /// Returns false if the normalization failed
    fn normalize_stream_ty_to_inputs(&mut self, ty: &mut StreamTy, stream_vars: &mut HashSet<NodeId>) -> bool {
        match ty {
            StreamTy::Event(ac) => self.normalize_activation_condition(ac, stream_vars),
            StreamTy::RealTime(_) => false,
            StreamTy::Infer(_) => unreachable!("ensured as only called after type inference is complete"),
        }
    }

    /// Returns false if the normalization failed
    fn normalize_activation_condition(
        &mut self,
        ac: &mut Activation<NodeId>,
        stream_vars: &mut HashSet<NodeId>,
    ) -> bool {
        match ac {
            Activation::Conjunction(args) | Activation::Disjunction(args) => {
                args.iter_mut().all(|arg| self.normalize_activation_condition(arg, stream_vars))
            }
            Activation::Stream(var) => {
                if stream_vars.contains(var) {
                    return true;
                }
                stream_vars.insert(*var);
                *ac = match &self.stream_ty[var] {
                    // make stream types default to event stream with empty conjunction, i.e., true
                    StreamTy::Infer(_) => unreachable!(),
                    StreamTy::Event(Activation::Stream(v)) if *var == *v => {
                        return true;
                    }
                    StreamTy::Event(ac) => ac.clone(),
                    StreamTy::RealTime(_) => {
                        self.handler.error("real-time streams cannot be used in activation conditions");
                        return false;
                    }
                };
                self.normalize_activation_condition(ac, stream_vars)
            }
            Activation::True => true,
        }
    }

    fn translate_activation_condition(
        &self,
        spec: &'a RTLolaAst,
        ac: &Activation<NodeId>,
    ) -> Option<Activation<crate::ir::StreamReference>> {
        Some(match ac {
            Activation::Conjunction(args) => Activation::Conjunction(
                args.iter().flat_map(|ac| self.translate_activation_condition(spec, ac)).collect(),
            ),
            Activation::Disjunction(args) => Activation::Disjunction(
                args.iter().flat_map(|ac| self.translate_activation_condition(spec, ac)).collect(),
            ),
            Activation::Stream(var) => {
                if let Some(idx) = spec.inputs.iter().enumerate().find(|(_, val)| val.id == *var).map(|x| x.0) {
                    Activation::Stream(crate::ir::StreamReference::InRef(idx))
                } else {
                    // ignore remaining output variables, they are self-refrences
                    Activation::True
                }
            }
            Activation::True => Activation::True,
        })
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::analysis::naming::*;
    use crate::parse::*;
    use crate::reporting::Handler;
    use crate::ty::{FloatTy, IntTy, UIntTy};
    use crate::FrontendConfig;
    use num::rational::Rational64 as Rational;
    use num::FromPrimitive;
    use std::path::PathBuf;

    fn num_type_errors(spec: &str) -> usize {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let spec = match parse(spec, &handler, FrontendConfig::default()) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        let mut na = NamingAnalysis::new(&handler, FrontendConfig::default());
        let mut decl_table = na.check(&spec);
        assert!(!handler.contains_error(), "Spec produces errors in naming analysis.");
        let mut type_analysis = TypeAnalysis::new(&handler, &mut decl_table);
        type_analysis.check(&spec);
        handler.emitted_errors()
    }

    fn num_type_warnings(spec: &str) -> usize {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let spec = match parse(spec, &handler, FrontendConfig::default()) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        let mut na = NamingAnalysis::new(&handler, FrontendConfig::default());
        let mut decl_table = na.check(&spec);
        assert!(!handler.contains_error(), "Spec produces errors in naming analysis.");
        let mut type_analysis = TypeAnalysis::new(&handler, &mut decl_table);
        type_analysis.check(&spec);
        handler.emitted_warnings()
    }

    /// Returns the type of the last output of the given spec
    fn get_type(spec: &str) -> ValueTy {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let spec = match parse(spec, &handler, FrontendConfig::default()) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        let mut na = NamingAnalysis::new(&handler, FrontendConfig::default());
        let mut decl_table = na.check(&spec);
        assert!(!handler.contains_error(), "Spec produces errors in naming analysis.");
        let mut type_analysis = TypeAnalysis::new(&handler, &mut decl_table);
        type_analysis.check(&spec);
        type_analysis.get_type(spec.outputs.last().expect("spec needs at least one output").id)
    }

    fn type_check(spec: &str) -> TypeTable {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let spec = match parse(spec, &handler, FrontendConfig::default()) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        let mut na = NamingAnalysis::new(&handler, FrontendConfig::default());
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
        let spec = "input i(a: Int8, b: Bool): Int8\noutput o := i(1,false)[0].defaults(to: 42)";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn parametric_declaration() {
        let spec = "output x(a: UInt8, b: Bool): Int8 := 1 output y := x(1, false)";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn simple_const_float() {
        let spec = "constant c: Float32 := 2.1";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn simple_const_float16() {
        let spec = "constant c: Float16 := 2.1";
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
    fn simple_trigger() {
        //let spec = "trigger never := false";
        //assert_eq!(0, num_type_errors(spec));
        let spec = "trigger false";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn faulty_trigger() {
        //let spec = "trigger failed := 1";
        //assert_eq!(1, num_type_errors(spec));
        let spec = "trigger 1";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    #[ignore] // for now we don't allow named triggers
    fn reuse_trigger() {
        let spec = "output a: Float64 @1Hz := 0.0\ntrigger t := a > 5.0\noutput b := if t then 2 else 3";
        assert_eq!(0, num_type_errors(spec));
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
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I64));
    }

    #[test]
    fn underspecified_ite_type() {
        let spec = "output o := if !false then 1.3 else -2.0";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Float(FloatTy::F64));
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
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I64));
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
    fn test_terminate_type() {
        let spec = "input in: Bool\n output a(b: Bool): Int8 close in := 3";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn test_terminate_type_faulty() {
        let spec = "input in: Int8\n output a(b: Bool): Int8 close in := 3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_terminate_type_faulty_ac() {
        // stream type is not compatible
        let spec = "input in: Int8 input in2: Bool output a(b: Bool): Int8 @in close in2 := 3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_param_spec() {
        let spec = "output a(p1: Int8): Int8 := 3 output b: Int8 := a(3)";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn test_param_spec_faulty() {
        let spec = "output a(p1: Int8): Int8:= 3 output b: Int8 := a(true)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_param_inferred() {
        let spec = "input i: Int8 output x(param): Int8 := 3 output y: Int8 := x(i)";
        assert_eq!(0, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn test_param_inferred_conflicting() {
        let spec = "input i: Int8, j: UInt8 output x(param): Int8 := 3 output y: Int8 := x(i) output z: Int8 := x(j)";
        assert_eq!(1, num_type_errors(spec));
        assert_eq!(get_type(spec), ValueTy::Int(IntTy::I8));
    }

    #[test]
    fn test_lookup_incomp() {
        let spec = "output a(p1: Int8): Int8 := 3\n output b: UInt8 := a(3)";
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
    fn test_optional_type() {
        let spec = "input in: Int8\noutput out: Int8? := in.offset(by: -1)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_optional_type_faulty() {
        let spec = "input in: Int8\noutput out: Int8? := in";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_input_offset() {
        let spec = "input a: UInt8\n output b: UInt8 := a[3].defaults(to: 10)";
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
        let spec = "input in: Int8\n output out: Int64 @5Hz:= in.aggregate(over: 3s, using: Σ)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_window() {
        let spec = "input in: Int8\n output out: Int8 @5Hz := in.aggregate(over: 3s, using: Σ)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_window_untimed() {
        let spec = "input in: Int8\n output out: Int16 := in.aggregate(over: 3s, using: Σ)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_window_faulty() {
        let spec = "input in: Int8\n output out: Bool @5Hz := in.aggregate(over: 3s, using: Σ)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_window_invalid_duration() {
        let spec = "input in: Int8\n output out: Bool @5Hz := in.aggregate(over: 0s, using: Σ)";
        assert_eq!(1, num_type_errors(spec));
        let spec = "input in: Int8\n output out: Bool @5Hz := in.aggregate(over: -3s, using: Σ)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    #[ignore] // ignore until implemented
    fn test_aggregation_implicit_cast() {
        let spec =
            "input in: UInt8\n output out: Int16 @5Hz := in.aggregate(over_exactly: 3s, using: Σ).defaults(to: 5)";
        assert_eq!(0, num_type_errors(spec));
        let spec =
            "input in: Int8\n output out: Float32 @5Hz := in.aggregate(over_exactly: 3s, using: avg).defaults(to: 5.0)";
        assert_eq!(0, num_type_errors(spec));
        let spec =
            "input in: Int8\n output out: Float32 @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5.0)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_aggregation_integer_integral() {
        let spec =
            "input in: UInt8\n output out: UInt8 @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5)";
        assert_eq!(1, num_type_errors(spec));
        let spec =
            "input in: Int8\n output out: Int8 @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5)";
        assert_eq!(1, num_type_errors(spec));
        let spec =
            "input in: UInt8\n output out @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5.0)";
        assert_eq!(0, num_type_errors(spec));
        let spec =
            "input in: Int8\n output out @5Hz := in.aggregate(over_exactly: 3s, using: integral).defaults(to: 5.0)";
        assert_eq!(0, num_type_errors(spec));
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
        let spec = "input velo: Float32\n output avg: Float64 @5Hz := velo.aggregate(over_exactly: 1h, using: avg).defaults(to: 10000.0)";
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
    fn test_rt_offset_regression2() {
        let spec = "
            output x @ 10Hz := 1
            output x_diff := x - x.offset(by:-1s).defaults(to: x)
        ";
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
        let spec = "input x: UInt8\noutput y: UInt8 @ x := x.hold().defaults(to: 0)";
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
        let spec = "output c := c.defaults(to:0)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_function_arguments_regression() {
        let spec = "input a: Int32\ntrigger a > 50";
        let type_table = type_check(spec);
        // expression `a > 50` has NodeId = 3
        let exp_a_gt_50_id = NodeId::new(5);
        assert_eq!(type_table.get_value_type(exp_a_gt_50_id), &ValueTy::Bool);
        assert_eq!(type_table.get_func_arg_types(exp_a_gt_50_id), &vec![ValueTy::Int(IntTy::I32)]);
    }

    #[test]
    fn test_conjunctive_stream_types() {
        let spec = "input a: Int32\ninput b: Int32\noutput x := a + b";
        let type_table = type_check(spec);
        // input `a` has NodeId = 1
        let a_id = NodeId::new(1);
        // input `b` has NodeId = 3
        let b_id = NodeId::new(3);
        // output `x` has NodeId = 9
        let x_id = NodeId::new(9);
        assert_eq!(type_table.get_value_type(x_id), &ValueTy::Int(IntTy::I32));
        assert_eq!(
            type_table.get_stream_type(x_id),
            &StreamTy::Event(Activation::Conjunction(vec![Activation::Stream(a_id), Activation::Stream(b_id)]))
        );
    }

    #[test]
    fn test_activation_condition() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a || b) := 1";
        let type_table = type_check(spec);
        // node ids can be verified using `rtlola-analyze spec.lola ast`
        // input `a` has NodeId = 1
        let a_id = NodeId::new(1);
        // input `b` has NodeId = 3
        let b_id = NodeId::new(3);
        // output `x` has NodeId = 14
        let x_id = NodeId::new(14);
        assert_eq!(type_table.get_value_type(x_id), &ValueTy::Int(IntTy::I64));
        assert_eq!(
            type_table.get_stream_type(x_id),
            &StreamTy::Event(Activation::Disjunction(vec![Activation::Stream(a_id), Activation::Stream(b_id)]))
        );
    }

    #[test]
    fn test_realtime_activation_condition() {
        let spec = "output a: Int32 @10Hz := 0\noutput b: Int32 @5Hz := 0\noutput x := a+b";
        let type_table = type_check(spec);
        // node ids can be verified using `rtlola-analyze spec.lola ast`
        // output `a` has NodeId = 6
        let _a_id = NodeId::new(6);
        // output `b` has NodeId = 13
        let _b_id = NodeId::new(13);
        // output `x` has NodeId = 19
        let x_id = NodeId::new(19);

        assert_eq!(type_table.get_value_type(x_id), &ValueTy::Int(IntTy::I32));

        assert_eq!(
            type_table.get_stream_type(x_id),
            &StreamTy::RealTime(Freq::new(UOM_Frequency::new::<hertz>(Rational::from_u8(5).unwrap())))
        );
    }

    #[test]
    fn test_get() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a || b) := a.get().defaults(to: 0)";
        let type_table = type_check(spec);
        // node ids can be verified using `rtlola-analyze spec.lola ast`
        // input `a` has NodeId = 1
        // input `b` has NodeId = 3
        // output `x` has NodeId = 19
        assert_eq!(type_table.get_value_type(NodeId::new(19)), &ValueTy::Int(IntTy::I32));
        assert_eq!(
            type_table.get_stream_type(NodeId::new(19)),
            &StreamTy::Event(Activation::Disjunction(vec![
                Activation::Stream(NodeId::new(1)),
                Activation::Stream(NodeId::new(3))
            ]))
        );
    }

    #[test]
    fn test_no_direct_access_possible() {
        let spec = "input a: Int32\ninput b: Int32\noutput x @(a || b) := a";
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
        // node ids can be verified using `rtlola-analyze spec.lola ast`
        //  input `a` has NodeId =  1
        let a_id = NodeId::new(1);
        //  input `b` has NodeId =  3
        let b_id = NodeId::new(3);
        //  input `c` has NodeId =  5
        let c_id = NodeId::new(5);
        // output `x` has NodeId = 11
        let x_id = NodeId::new(11);
        // output `y` has NodeId = 19
        let y_id = NodeId::new(19);

        let stream_ty_x = type_table.get_stream_type(x_id);

        assert_eq!(
            stream_ty_x,
            &StreamTy::Event(Activation::Conjunction(vec![Activation::Stream(a_id), Activation::Stream(b_id)]))
        );

        let stream_ty_y = type_table.get_stream_type(y_id);
        assert_eq!(
            stream_ty_y,
            &StreamTy::Event(Activation::Conjunction(vec![
                Activation::Stream(a_id),
                Activation::Stream(b_id),
                Activation::Stream(c_id)
            ]))
        );

        use crate::ir::StreamReference;
        assert_eq!(
            type_table.get_acti_cond(x_id),
            &Activation::Conjunction(vec![
                Activation::Stream(StreamReference::InRef(0)),
                Activation::Stream(StreamReference::InRef(1)),
            ])
        );
        assert_eq!(
            type_table.get_acti_cond(y_id),
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
        let spec = "
            output a @3Hz := 0
            output a_offset := a[-1].defaults(to: 0)
            output b @2Hz := b[-1].defaults(to: 0) + a_offset.hold().defaults(to: 0)
        ";
        // workaround using sample and hold
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_to_large_literals() {
        let spec = "output a: Int32 := 1111111111111111111111111110";
        assert_eq!(1, num_type_errors(spec));
    }
}
