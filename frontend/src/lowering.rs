use crate::analysis::naming::DeclarationTable;
use crate::ty::check::TypeTable;
// Only import the unambiguous Nodes, use `ast::`/`ir::` prefix for disambiguation.
use crate::analysis::naming::Declaration;
use crate::ast;
use crate::ast::{ExpressionKind, LolaSpec};
use crate::ir;
use crate::ir::{
    EventDrivenStream, LolaIR, MemorizationBound, ParametrizedStream, StreamReference, TimeDrivenStream,
    WindowReference,
};
use crate::parse::NodeId;
use crate::ty::StreamTy;
use std::collections::HashMap;
use std::time::Duration;

use crate::analysis::graph_based_analysis::evaluation_order::{EvalOrder, EvaluationOrderResult};
use crate::analysis::graph_based_analysis::space_requirements::{
    SpaceRequirements as MemoryTable, TrackingRequirements,
};
//use crate::analysis::graph_based_analysis::future_dependency::FutureDependentStreams;
use crate::analysis::graph_based_analysis::{ComputeStep, RequiredInputs, StorageRequirement, TrackingRequirement};

use self::lowering_state::*;
use crate::analysis::AnalysisResult;

use num::ToPrimitive;

type EvalTable = HashMap<NodeId, u32>;

pub(crate) struct Lowering<'a> {
    ast: &'a LolaSpec,
    ref_lookup: HashMap<NodeId, StreamReference>,
    dt: &'a DeclarationTable<'a>,
    tt: &'a TypeTable,
    et: EvalTable,
    mt: &'a MemoryTable,
    tr: &'a TrackingRequirements,
    ir: LolaIR,
    ri: &'a RequiredInputs,
}

impl<'a> Lowering<'a> {
    pub(crate) fn new(ast: &'a LolaSpec, analysis_result: &'a AnalysisResult<'a>) -> Lowering<'a> {
        let mut ir = LolaIR {
            inputs: Vec::new(),
            outputs: Vec::new(),
            time_driven: Vec::new(),
            event_driven: Vec::new(),
            parametrized: Vec::new(),
            sliding_windows: Vec::new(),
            triggers: Vec::new(),
            feature_flags: Vec::new(),
        };

        ir.inputs.reserve(ast.inputs.len());
        ir.outputs.reserve(ast.outputs.len());
        let graph = analysis_result.graph_analysis_result.as_ref().unwrap();

        Lowering {
            ast,
            ref_lookup: Lowering::create_ref_lookup(&ast.inputs, &ast.outputs),
            dt: analysis_result.declaration_table.as_ref().unwrap(),
            tt: analysis_result.type_table.as_ref().unwrap(),
            et: Self::order_to_table(&graph.evaluation_order),
            mt: &graph.space_requirements,
            tr: &graph.tracking_requirements,
            ir,
            ri: &graph.input_dependencies,
        }
    }

    pub(crate) fn lower(mut self) -> LolaIR {
        self.lower_ast();
        self.ir
    }

    fn lower_ast(&mut self) {
        self.ast.inputs.iter().for_each(|i| self.lower_input(i));
        self.ast.outputs.iter().for_each(|o| self.lower_output(o));
        self.link_windows();
        self.ast.trigger.iter().for_each(|t| self.lower_trigger(t));
    }

    fn link_windows(&mut self) {
        // Extract and copy relevant information before-hand to avoid double burrow.
        let essences: Vec<(StreamReference, WindowReference)> =
            self.ir.sliding_windows.iter().map(|window| (window.target, window.reference)).collect();
        for (target, window) in essences {
            match target {
                StreamReference::InRef(_) => {
                    let windows = &mut self.ir.get_in_mut(target).dependent_windows;
                    windows.push(window);
                }
                StreamReference::OutRef(_) => {
                    let windows = &mut self.ir.get_out_mut(target).dependent_windows;
                    windows.push(window);
                }
            }
        }
    }

    /// Does *not* add dependent windows, yet.
    fn lower_input(&mut self, input: &ast::Input) {
        let nid = input.id;
        let ast_req = self.get_memory(nid);
        let memory_bound = self.lower_storage_req(ast_req);
        let reference = self.get_ref_for_stream(nid);
        let layer = self.get_layer(nid);

        let trackings = self.collect_tracking_info(nid, None);

        let input = ir::InputStream {
            name: input.name.name.clone(),
            ty: self.lower_value_type(nid),
            dependent_streams: trackings,
            dependent_windows: Vec::new(),
            layer, // Not necessarily 0 when parametrized inputs exist.
            memory_bound,
            reference,
        };

        let debug_clone = input.clone();
        self.ir.inputs.push(input);

        assert_eq!(
            self.ir.get_in(reference),
            &debug_clone,
            "Bug in implementation: Output vector in IR changed between creation of reference and insertion of stream."
        );
    }

    fn gather_dependent_inputs(&mut self, node_id: NodeId) -> Vec<StreamReference> {
        self.ri[&node_id].iter().map(|input_id| self.get_ref_for_stream(*input_id)).collect()
    }

    fn lower_trigger(&mut self, trigger: &ast::Trigger) {
        let name = if let Some(msg) = trigger.message.as_ref() {
            format!("trigger_{}", msg.clone().replace(" ", "_"))
        } else {
            String::from("trigger")
        };

        let ty = ir::Type::Bool;
        let expr = self.lower_expression(&trigger.expression, &ty);
        let reference = StreamReference::OutRef(self.ir.outputs.len());
        let outgoing_dependencies = self.find_dependencies(&trigger.expression);
        let input_dependencies = self.gather_dependent_inputs(trigger.id);
        let output = ir::OutputStream {
            name,
            ty,
            expr,
            dependent_streams: Vec::new(),
            dependent_windows: Vec::new(),
            memory_bound: MemorizationBound::Bounded(1),
            layer: self.get_layer(trigger.id),
            reference,
            outgoing_dependencies,
            input_dependencies,
        };
        self.ir.outputs.push(output);
        let trig = ir::Trigger { message: trigger.message.clone(), reference };
        match self.check_time_driven(trigger.id, reference) {
            None => self.ir.event_driven.push(EventDrivenStream { reference }),
            Some(tds) => self.ir.time_driven.push(tds),
        }
        self.ir.triggers.push(trig);
    }

    fn collect_tracking_info(&self, nid: NodeId, time_driven: Option<TimeDrivenStream>) -> Vec<ir::Tracking> {
        let dependent = self.find_depending_streams(nid);
        assert!(
            dependent.iter().all(|(_, req)| match req {
                TrackingRequirement::Unbounded => false,
                _ => true,
            }),
            "Unbounded dependencies are not supported, yet."
        );

        dependent.into_iter().map(|(trackee, req)| self.lower_tracking_req(time_driven, trackee, req)).collect()
    }

    /// Does *not* add dependent windows, yet.
    fn lower_output(&mut self, ast_output: &ast::Output) {
        let nid = ast_output.id;
        let ast_req = self.get_memory(nid);
        let memory_bound = self.lower_storage_req(ast_req);
        let layer = self.get_layer(nid);
        let reference = self.get_ref_for_stream(nid);
        let time_driven = self.check_time_driven(ast_output.id, reference);
        let parametrized = self.check_parametrized(&ast_output, reference);

        let trackings = self.collect_tracking_info(nid, time_driven);
        let outgoing_dependencies = self.find_dependencies(&ast_output.expression);
        let mut dep_map: HashMap<StreamReference, Vec<ir::Offset>> = HashMap::new();
        outgoing_dependencies.into_iter().for_each(|dep| {
            dep_map.entry(dep.stream).or_insert_with(Vec::new).extend_from_slice(dep.offsets.as_slice())
        });

        let outgoing_dependencies =
            dep_map.into_iter().map(|(sr, offsets)| ir::Dependency { stream: sr, offsets }).collect();

        let output_type = self.lower_value_type(nid);
        let input_dependencies = self.gather_dependent_inputs(nid);
        let output = ir::OutputStream {
            name: ast_output.name.name.clone(),
            ty: output_type.clone(),
            expr: self.lower_expression(&ast_output.expression, &output_type),
            outgoing_dependencies,
            dependent_streams: trackings,
            dependent_windows: Vec::new(),
            memory_bound,
            layer,
            reference,
            input_dependencies,
        };

        let debug_clone = output.clone();
        self.ir.outputs.push(output);
        assert_eq!(
            self.ir.get_out(reference),
            &debug_clone,
            "Bug in implementation: Output vector in IR changed between creation of reference and insertion of stream."
        );

        if let Some(td_ref) = time_driven {
            self.ir.time_driven.push(td_ref)
        } else {
            self.ir.event_driven.push(EventDrivenStream { reference })
        }

        if let Some(param_ref) = parametrized {
            self.ir.parametrized.push(param_ref);
        }
    }

    /// Returns the flattened result of calling f on each node recursively in `pre_order` or post_order.
    fn collect_expression<T, F>(expr: &'a ast::Expression, f: &F, pre_order: bool) -> Vec<T>
    where
        F: Fn(&'a ast::Expression) -> Vec<T>,
    {
        let recursion = |e| Lowering::collect_expression(e, f, pre_order);
        let pre = if pre_order { f(expr).into_iter() } else { Vec::new().into_iter() };
        let post = || {
            if pre_order {
                Vec::new().into_iter()
            } else {
                f(expr).into_iter()
            }
        };
        match &expr.kind {
            ExpressionKind::Lit(_) => pre.chain(post()).collect(),
            ExpressionKind::StreamAccess(_, _) => unimplemented!(),
            ExpressionKind::Ident(_) => pre.chain(post()).collect(),
            ExpressionKind::Default(e, dft) => pre.chain(recursion(e)).chain(recursion(dft)).chain(post()).collect(),
            ExpressionKind::Offset(_, _) => unimplemented!(),
            ExpressionKind::SlidingWindowAggregation {
                expr: _expr,
                duration: _duration,
                aggregation: _aggregation,
            } => unimplemented!(),
            /*ExpressionKind::Lookup(inst, _, _) => {
                let args = inst.arguments.iter().flat_map(|a| recursion(a));
                pre.chain(args).chain(post()).collect()
            }*/
            ExpressionKind::Binary(_, lhs, rhs) => {
                pre.chain(recursion(lhs)).chain(recursion(rhs)).chain(post()).collect()
            }
            ExpressionKind::Unary(_, operand) => pre.chain(recursion(operand)).chain(post()).collect(),
            ExpressionKind::Ite(cond, cons, alt) => {
                pre.chain(recursion(cond)).chain(recursion(cons)).chain(recursion(alt)).chain(post()).collect()
            }
            ExpressionKind::ParenthesizedExpression(_, e, _) => {
                pre.chain(Lowering::collect_expression(e, f, pre_order)).chain(post()).collect()
            }
            ExpressionKind::MissingExpression => panic!(), // TODO: Eradicate in preceding step.
            ExpressionKind::Tuple(exprs) => {
                let elems = exprs.iter().flat_map(|a| recursion(a));
                pre.chain(elems).chain(post()).collect()
            }
            ExpressionKind::Function(_, _, args) => {
                let args = args.iter().flat_map(|a| recursion(a));
                pre.chain(args).chain(post()).collect()
            }
            ExpressionKind::Field(e, _) => pre.chain(recursion(e)).chain(post()).collect(),
            ExpressionKind::Method(_, _, _, _) => unimplemented!(),
        }
    }

    fn find_dependencies(&self, expr: &ast::Expression) -> Vec<ir::Dependency> {
        let f = |e: &ast::Expression| -> Vec<ir::Dependency> {
            match &e.kind {
                /*ExpressionKind::Lookup(_target, offset, None) => {
                    let sr = self.extract_target_from_lookup(&e.kind);
                    let offset = self.lower_offset(offset);
                    vec![ir::Dependency { stream: sr, offsets: vec![offset] }]
                }*/
                ExpressionKind::Ident(_ident) => {
                    let sr = self.get_ref_for_ident(e.id);
                    let offset = ir::Offset::PastDiscreteOffset(0);
                    vec![ir::Dependency { stream: sr, offsets: vec![offset] }]
                }
                _ => Vec::new(),
            }
        };
        Lowering::collect_expression(expr, &f, true)
    }

    fn lower_tracking_req(
        &self,
        tracker: Option<TimeDrivenStream>,
        trackee: NodeId,
        req: TrackingRequirement,
    ) -> ir::Tracking {
        let trackee = self.get_ref_for_stream(trackee);
        match req {
            TrackingRequirement::Unbounded => ir::Tracking::All(trackee),
            TrackingRequirement::Finite(num) => {
                let rate = tracker.map_or(Duration::from_secs(0), |tds| tds.extend_rate);
                ir::Tracking::Bounded { trackee, num: u128::from(num), rate }
            }
            TrackingRequirement::Future => unimplemented!(),
        }
    }

    fn lower_window(&mut self, expr: &ast::Expression) -> WindowReference {
        unimplemented!();
        /*if let ExpressionKind::Lookup(_, offset, Some(op)) = &expr.kind {
            let duration = match self.lower_offset(offset) {
                ir::Offset::PastDiscreteOffset(_)
                | ir::Offset::FutureDiscreteOffset(_)
                | ir::Offset::PastRealTimeOffset(_) => panic!("Bug: Should be caught in preceeding step."),
                ir::Offset::FutureRealTimeOffset(dur) => dur,
            };
            let ty = self.lower_value_type(expr.id);
            let target = self.extract_target_from_lookup(&expr.kind);
            let reference = WindowReference { ix: self.ir.sliding_windows.len() };
            let op = self.lower_window_op(*op);
            let window = ir::SlidingWindow { target, duration, op, reference, ty };
            self.ir.sliding_windows.push(window);
            reference
        } else {
            panic!("Bug in implementation: Called `lower_window` on non-window expression.")
        }*/
    }

    fn extract_target_from_lookup(&self, lookup: &ExpressionKind) -> StreamReference {
        unimplemented!();
        /*if let ExpressionKind::Lookup(inst, _, _) = lookup {
            let decl = self.get_decl(inst.id);
            let nid = match decl {
                Declaration::Out(out) => out.id,
                Declaration::In(inp) => inp.id,
                Declaration::Const(_) => unimplemented!(),
                Declaration::Param(_) => unimplemented!(),
                Declaration::Type(_) | Declaration::Func(_) => panic!("Bug in implementation: Invalid lookup."),
            };
            *self.ref_lookup.get(&nid).expect("Bug in ReferenceLookup.")
        } else {
            panic!("Bug in implementation: Called `extract_target_from_lookup` on non-lookup expression.")
        }*/
    }

    fn lower_window_op(&self, op: ast::WindowOperation) -> ir::WindowOperation {
        match op {
            ast::WindowOperation::Average => ir::WindowOperation::Average,
            ast::WindowOperation::Count => ir::WindowOperation::Count,
            ast::WindowOperation::Integral => ir::WindowOperation::Integral,
            ast::WindowOperation::Product => ir::WindowOperation::Product,
            ast::WindowOperation::Sum => ir::WindowOperation::Sum,
        }
    }

    fn check_parametrized(
        &mut self,
        ast_output: &ast::Output,
        reference: StreamReference,
    ) -> Option<ParametrizedStream> {
        if let Some(temp_spec) = ast_output.template_spec.as_ref() {
            // Check if it is merely timed, not parametrized.
            if temp_spec.ext.is_none() && temp_spec.inv.is_none() && temp_spec.ter.is_none() {
                None
            } else {
                // TODO: Finalize and implement parametrization.
                Some(ParametrizedStream {
                    reference,
                    params: ast_output.params.iter().map(|p| self.lower_param(p)).collect(),
                    invoke: None,
                    extend: None,
                    terminate: None,
                })
            }
        } else {
            assert!(ast_output.params.is_empty());
            None
        }
    }

    fn lower_param(&mut self, param: &ast::Parameter) -> ir::Parameter {
        ir::Parameter { name: param.name.name.clone(), ty: self.lower_value_type(param.id) }
    }

    fn lower_storage_req(&self, req: StorageRequirement) -> MemorizationBound {
        match req {
            StorageRequirement::Finite(b) => MemorizationBound::Bounded(b as u16),
            StorageRequirement::FutureRef(_) | StorageRequirement::Unbounded => {
                MemorizationBound::Unbounded
            }
        }
    }

    fn lower_value_type(&mut self, id: NodeId) -> ir::Type {
        self.tt.get_value_type(id).into()
    }

    fn lower_expression(&mut self, expression: &ast::Expression, expected_type: &ir::Type) -> ir::Expression {
        let mut state = self.lower_subexpression(expression, LoweringState::empty());
        if state.result_type() != expected_type {
            let convert = ir::Statement {
                target: state.temp_for_type(expected_type),
                op: ir::Op::Convert,
                args: vec![state.get_target()],
            };
            state = state.with_stmt(convert)
        }
        let (stmts, temporaries) = state.finalize();
        ir::Expression { stmts, temporaries }
    }

    fn lower_subexpression(&mut self, expr: &ast::Expression, mut state: LoweringState) -> LoweringState {
        use crate::ir::{Op, Statement};

        let result_type = self.lower_value_type(expr.id);

        match &expr.kind {
            ExpressionKind::Lit(l) => {
                let op = Op::LoadConstant(self.lower_literal(l));
                let args = Vec::new();
                let stmt = Statement { target: state.temp_for_type(&result_type), op, args };
                state.with_stmt(stmt)
            }
            ExpressionKind::Ident(_) => {
                let sr = self.get_ref_for_ident(expr.id);
                self.lower_sync_lookup(state, sr)
            }
            ExpressionKind::StreamAccess(_, _) => unimplemented!(),
            ExpressionKind::Default(e, dft) => {
                unimplemented!();
                /*if let ExpressionKind::Lookup(_, _, _) = &e.kind {
                    let result_type = self.lower_value_type(expr.id);
                    self.lower_lookup_expression(e, dft, state, result_type, false)
                } else {
                    // A "stray" default expression such as `5.defaults(to: 3)` is valid, but a no-op.
                    // Thus, print a warning. Evaluating the expression is necessary, the dft can be skipped.
                    println!("WARNING: No-Op Default operation!");
                    self.lower_subexpression(e, state)
                }*/
            }
            ExpressionKind::Offset(_, _) => unimplemented!(),
            ExpressionKind::SlidingWindowAggregation {
                expr: _expr,
                duration: _duration,
                aggregation: _aggregation,
            } => unimplemented!(),
            /*ExpressionKind::Hold(e, dft) => {
                if let ExpressionKind::Lookup(_, _, _) = &e.kind {
                    let result_type = self.lower_value_type(expr.id);
                    self.lower_lookup_expression(e, dft, state, result_type, true)
                } else {
                    // A "stray" sample and hold expression such as `5 ! 3` is valid, but a no-op.
                    // Thus, print a warning. Evaluating the expression is necessary, the dft can be skipped.
                    println!("WARNING: No-Op Sample and Hold operation!");
                    self.lower_subexpression(e, state)
                }
            }*/
            /*ExpressionKind::Lookup(_, _, _) => {
                // Stray lookup without any default expression surrounding it, see `ExpressionKind::Default` case.
                // This is only valid for sync accesses, i.e. the offset is 0. And this is an `Ident` expression.
                unimplemented!("Assert offset is 0, transform into sync access.")
            }*/
            ExpressionKind::Binary(ast_op, lhs, rhs) => {
                use crate::ast::BinOp::*;
                let req_arg_types = self.tt.get_func_arg_types(expr.id);
                let req_arg_type = if req_arg_types.is_empty() {
                    match ast_op {
                        Add | Sub | Mul | Div | Rem | Pow | Eq | Lt | Le | Ne | Ge | Gt => {
                            panic!("Generic operator not recognized as such.")
                        }
                        And | Or => ir::Type::Bool,
                    }
                } else {
                    assert_eq!(req_arg_types.len(), 1);
                    (&req_arg_types[0]).into()
                };

                state = self.lower_subexpression(lhs, state);
                let lhs_type = state.result_type();
                if lhs_type != &req_arg_type {
                    let conversion_source = state.get_target();
                    let conversion_target = state.temp_for_type(&req_arg_type);
                    state = self.convert_temp(state, conversion_source, conversion_target);
                }
                let lhs_target = state.get_target();

                state = self.lower_subexpression(rhs, state);
                let rhs_type = state.result_type();
                if rhs_type != &req_arg_type {
                    let conversion_source = state.get_target();
                    let conversion_target = state.temp_for_type(&req_arg_type);
                    state = self.convert_temp(state, conversion_source, conversion_target);
                }
                let rhs_target = state.get_target();

                let args = vec![lhs_target, rhs_target];
                let op = Op::ArithLog(Lowering::lower_bin_op(*ast_op));
                let stmt = Statement { target: state.temp_for_type(&result_type), op, args };
                state.with_stmt(stmt)
            }
            ExpressionKind::Unary(ast_op, operand) => {
                state = self.lower_subexpression(operand, state);
                let op = Op::ArithLog(Lowering::lower_un_op(*ast_op));
                let stmt = Statement { target: state.temp_for_type(&result_type), op, args: vec![state.get_target()] };
                state.with_stmt(stmt)
            }
            ExpressionKind::Ite(cond, cons, alt) => {
                // Plan:
                // a) Compute condition.
                // b) Branch state off, lower consequence.
                // c) If apropos, convert consequence.
                // d) Merge temps from branch into state.
                // e) Branch state off, lower alternative.
                // f) Move or convert alternative into result of consequence.
                // g) Merge temps into state.
                // h) Add Ite Statement.

                // a) Compute condition.
                state = self.lower_subexpression(cond, state);
                let cond_target = state.get_target();

                // b) Branch state off, lower consequence.
                let mut cons_state = self.lower_subexpression(cons, state.branch());

                // c) If apropos, convert consequence.
                if &result_type != cons_state.result_type() {
                    let conversion_source = cons_state.get_target();
                    let conversion_target = cons_state.temp_for_type(&result_type);
                    cons_state = self.convert_temp(cons_state, conversion_source, conversion_target);
                }
                let branch_target = cons_state.get_target();

                // d) Merge temps from branch into state.
                let (cons_stmts, cons_temps) = cons_state.destruct();
                state = state.with_temps(cons_temps);

                // e) Branch state off, lower alternative.
                let mut alt_state = self.lower_subexpression(alt, state.branch());

                // f) Move or convert alternative into result of consequence.
                alt_state = if &result_type != alt_state.result_type() {
                    self.convert_temp(alt_state, branch_target, branch_target)
                } else {
                    let move_source = alt_state.get_target();
                    alt_state.with_stmt(ir::Statement { target: branch_target, op: Op::Move, args: vec![move_source] })
                };

                // g) Merge temps into state.
                let (alt_stmts, alt_temps) = alt_state.destruct();
                state = state.with_temps(alt_temps);

                // h) Add Ite Statement.
                let op = Op::Ite { consequence: cons_stmts, alternative: alt_stmts };
                let ite_stmt = Statement { target: branch_target, op, args: vec![cond_target] };

                state.with_stmt(ite_stmt)
            }
            ExpressionKind::ParenthesizedExpression(_, e, _) => self.lower_subexpression(e, state),
            ExpressionKind::MissingExpression => panic!("How wasn't this caught in a preceding step?!"),
            ExpressionKind::Tuple(exprs) => {
                let lowered_list = self.lower_expression_list(exprs, state);
                state = lowered_list.0;
                let stmt = Statement { target: state.temp_for_type(&result_type), op: Op::Tuple, args: lowered_list.1 };
                state.with_stmt(stmt)
            }
            ExpressionKind::Function(name, _, args) if name.name.name == "cast" => {
                // Special case for cast function.
                assert_eq!(args.len(), 1);
                let arg = &args[0];
                let mut state = self.lower_subexpression(arg, state);
                let conv_target = state.temp_for_type(&result_type);
                let conv_source = state.get_target();
                self.convert_temp(state, conv_source, conv_target)
            }
            ExpressionKind::Function(name, _, args) => {
                use crate::ty::ValueTy;
                let ir_kind = name.name.name.clone();

                let req_arg_types = self.tt.get_func_arg_types(expr.id);
                let args = if let Declaration::Func(fd) = self.get_decl(expr.id) {
                    fd.parameters
                        .iter()
                        .map(|param| {
                            if let ValueTy::Param(i, _) = param {
                                (&req_arg_types[*i as usize]).into()
                            } else {
                                param.into()
                            }
                        })
                        .zip(args.iter().map(|e| e.as_ref()))
                        .collect::<Vec<(ir::Type, &ast::Expression)>>()
                } else {
                    panic!()
                };

                let mut arg_temps = Vec::new();
                for (req_type, arg_expr) in args {
                    state = self.lower_subexpression(arg_expr, state);
                    let actual_type = state.result_type();
                    if actual_type != &req_type {
                        let conv_src = state.get_target();
                        let conv_tar = state.temp_for_type(&req_type);
                        state = self.convert_temp(state, conv_src, conv_tar);
                    }
                    arg_temps.push(state.get_target())
                }

                let stmt =
                    Statement { target: state.temp_for_type(&result_type), op: Op::Function(ir_kind), args: arg_temps };
                state.with_stmt(stmt)
            }
            ExpressionKind::Field(_, _) => unimplemented!(),
            ExpressionKind::Method(_, _, _, _) => unimplemented!(),
        }
    }

    fn convert_temp(&self, state: LoweringState, source: ir::Temporary, target: ir::Temporary) -> LoweringState {
        state.with_stmt(ir::Statement { target, op: ir::Op::Convert, args: vec![source] })
    }

    fn lower_sync_lookup(&self, mut state: LoweringState, lu_target_ref: StreamReference) -> LoweringState {
        let lu_target = ir::StreamInstance { reference: lu_target_ref, arguments: Vec::new() };
        let lu_type = match lu_target_ref {
            StreamReference::InRef(_) => &self.ir.get_in(lu_target_ref).ty,
            StreamReference::OutRef(_) => &self.ir.get_out(lu_target_ref).ty,
        };

        let lu_target_temp = state.temp_for_type(lu_type);

        let lu_stmt =
            ir::Statement { target: lu_target_temp, op: ir::Op::SyncStreamLookup(lu_target), args: Vec::new() };

        state.with_stmt(lu_stmt)
    }

    fn lower_expression_list(
        &mut self,
        expressions: &[Box<ast::Expression>],
        state: LoweringState,
    ) -> (LoweringState, Vec<ir::Temporary>) {
        expressions.iter().fold((state, Vec::new()), |(mut state, mut results), e| {
            state = self.lower_subexpression(e, state);
            results.push(state.get_target());
            (state, results)
        })
    }

    fn lower_lookup_expression(
        &mut self,
        lookup_expr: &ast::Expression,
        dft: &ast::Expression,
        mut state: LoweringState,
        desired_return_type: ir::Type,
        is_hold: bool,
    ) -> LoweringState {
        unimplemented!();
        /*if let ExpressionKind::Lookup(instance, offset, op) = &lookup_expr.kind {
            let target_ref = self.extract_target_from_lookup(&lookup_expr.kind);

            // Compute the default value first.
            state = self.lower_subexpression(dft, state);

            let default_type = state.result_type();
            if default_type != &desired_return_type {
                let conversion_source = state.get_target();
                let conversion_target = state.temp_for_type(&desired_return_type);
                state = self.convert_temp(state, conversion_source, conversion_target);
            }
            let default_temp = state.get_target();

            // Compute all arguments.
            let (mut state, lookup_args) = self.lower_expression_list(&instance.arguments, state);

            let op = if op.is_some() {
                assert!(!is_hold);
                let window_ref = self.lower_window(&lookup_expr);
                ir::Op::WindowLookup(window_ref)
            } else {
                let target_instance = ir::StreamInstance { reference: target_ref, arguments: lookup_args };
                if is_hold {
                    ir::Op::SampleAndHoldStreamLookup { instance: target_instance, offset: self.lower_offset(&offset) }
                } else {
                    ir::Op::StreamLookup { instance: target_instance, offset: self.lower_offset(&offset) }
                }
            };

            let lookup_result_type = self.lower_value_type(lookup_expr.id);
            let lookup_type = match lookup_result_type {
                ir::Type::Option(t) => *t.clone(),
                _ => panic!("A non-sync lookups always ought to produce an option."),
            };

            let lookup_stmt = ir::Statement { target: state.temp_for_type(&lookup_type), op, args: vec![default_temp] };
            state = state.with_stmt(lookup_stmt);

            if lookup_type != desired_return_type {
                let conversion_source = state.get_target();
                let conversion_target = state.temp_for_type(&desired_return_type);
                state = self.convert_temp(state, conversion_source, conversion_target);
            }
            state
        } else {
            panic!("Called `lower_lookup_expression` on a non-lookup expression.");
        }*/
    }

    fn extract_literal(&self, e: &ast::Expression) -> ir::Constant {
        match &e.kind {
            ExpressionKind::Lit(l) => self.lower_literal(l),
            _ => unreachable!("Made impossible in preceding step."),
        }
    }

    fn lower_literal(&self, lit: &ast::Literal) -> ir::Constant {
        use crate::ast::LitKind;
        match &lit.kind {
            LitKind::Str(s) | LitKind::RawStr(s) => ir::Constant::Str(s.clone()),
            LitKind::Numeric(_, _) => unimplemented!("need typing information"),
            LitKind::Bool(b) => ir::Constant::Bool(*b),
        }
    }

    fn lower_un_op(ast_op: ast::UnOp) -> ir::ArithLogOp {
        match ast_op {
            ast::UnOp::Neg => ir::ArithLogOp::Neg,
            ast::UnOp::Not => ir::ArithLogOp::Not,
        }
    }

    fn lower_bin_op(ast_op: ast::BinOp) -> ir::ArithLogOp {
        use crate::ast::BinOp::*;
        match ast_op {
            Add => ir::ArithLogOp::Add,
            Sub => ir::ArithLogOp::Sub,
            Mul => ir::ArithLogOp::Mul,
            Div => ir::ArithLogOp::Div,
            Rem => ir::ArithLogOp::Rem,
            Pow => ir::ArithLogOp::Pow,
            And => ir::ArithLogOp::And,
            Or => ir::ArithLogOp::Or,
            Eq => ir::ArithLogOp::Eq,
            Lt => ir::ArithLogOp::Lt,
            Le => ir::ArithLogOp::Le,
            Ne => ir::ArithLogOp::Ne,
            Ge => ir::ArithLogOp::Ge,
            Gt => ir::ArithLogOp::Gt,
        }
    }

    fn check_time_driven(&mut self, stream_id: NodeId, reference: StreamReference) -> Option<TimeDrivenStream> {
        match &self.tt.get_stream_type(stream_id) {
            StreamTy::RealTime(f) => Some(TimeDrivenStream {
                reference,
                extend_rate: Duration::from_nanos(
                    f.ns.to_integer().to_u64().expect("extend duration [ns] does not fit in u64"),
                ),
            }),
            _ => None,
        }
    }

    fn find_depending_streams(&self, nid: NodeId) -> Vec<(NodeId, TrackingRequirement)> {
        self.tr
            .iter()
            .flat_map(|(src_nid, reqs)| -> Vec<(NodeId, TrackingRequirement)> {
                reqs.iter()
                    .filter(|(tar_nid, _)| *tar_nid == nid) // Pick dependencies where `nid` is the target
                    .map(|(_, req)| (*src_nid, *req)) // Forget target, remember source.
                    .collect()
            })
            .collect()
    }

    fn create_ref_lookup(inputs: &[ast::Input], outputs: &[ast::Output]) -> HashMap<NodeId, StreamReference> {
        let ins = inputs.iter().enumerate().map(|(ix, i)| (i.id, StreamReference::InRef(ix)));
        let outs = outputs.iter().enumerate().map(|(ix, o)| (o.id, StreamReference::OutRef(ix))); // Re-start indexing @ 0.
        ins.chain(outs).collect()
    }

    fn order_to_table(eo: &EvaluationOrderResult) -> EvalTable {
        fn extr_id(step: ComputeStep) -> NodeId {
            // TODO: Rework when parameters actually exist.
            use self::ComputeStep::*;
            match step {
                Evaluate(nid) | Extend(nid) | Invoke(nid) | Terminate(nid) => nid,
            }
        }
        let o2t = |eo: &EvalOrder| {
            let mut res = Vec::new();
            for (ix, layer) in eo.iter().enumerate() {
                let vals = layer.iter().map(|s| (extr_id(*s), ix as u32));
                res.extend(vals);
            }
            res.into_iter()
        };
        o2t(&eo.periodic_streams_order)
            .chain(o2t(&eo.event_based_streams_order))
            .collect()
    }

    fn get_decl(&self, nid: NodeId) -> &Declaration {
        self.dt.get(&nid).expect("Bug in DeclarationTable.")
    }

    fn get_layer(&self, nid: NodeId) -> u32 {
        *self.et.get(&nid).expect("Bug in EvaluationOrder.")
    }

    fn get_memory(&self, nid: NodeId) -> StorageRequirement {
        *self.mt.get(&nid).expect("Bug in MemoryTable.")
    }

    fn get_ref_for_stream(&self, nid: NodeId) -> StreamReference {
        *self.ref_lookup.get(&nid).expect("Bug in ReferenceLookup.")
    }

    fn get_ref_for_ident(&self, nid: NodeId) -> StreamReference {
        match self.get_decl(nid) {
            Declaration::In(inp) => self.get_ref_for_stream(inp.id),
            Declaration::Out(out) => self.get_ref_for_stream(out.id),
            Declaration::Param(_) | Declaration::Const(_) => unimplemented!(),
            Declaration::Type(_) | Declaration::Func(_) => panic!("Types and functions are not streams."),
        }
    }
}

mod lowering_state {

    use crate::ir;
    use std::collections::HashMap;

    #[derive(Debug)]
    pub(crate) struct LoweringState {
        stmts: Vec<ir::Statement>,
        temps: HashMap<ir::Temporary, ir::Type>,
        next_temp: u32,
    }

    impl LoweringState {
        pub(crate) fn result_type(&self) -> &ir::Type {
            &self.temps[&self.get_target()]
        }

        pub(crate) fn get_target(&self) -> ir::Temporary {
            self.stmts.last().expect("A expression needs to return a value.").target
        }

        pub(crate) fn branch(&self) -> LoweringState {
            LoweringState { stmts: Vec::new(), temps: self.temps.clone(), next_temp: self.next_temp }
        }

        pub(crate) fn with_temps(mut self, others: HashMap<ir::Temporary, ir::Type>) -> LoweringState {
            for (temp, ty) in others {
                if temp.0 > self.next_temp as usize {
                    self.next_temp = (temp.0 + 1) as u32;
                }
                self.add_temp(temp, &ty);
            }
            self
        }

        pub(crate) fn temp_for_type(&mut self, ty: &ir::Type) -> ir::Temporary {
            let temp = ir::Temporary::from(self.next_temp);
            self.next_temp += 1;
            self.add_temp(temp, ty);
            temp
        }

        fn add_temp(&mut self, temp: ir::Temporary, ty: &ir::Type) {
            if cfg!(debug_assertions) {
                if let Some(other) = self.temps.get(&temp) {
                    assert_eq!(other, ty);
                }
            }
            self.temps.insert(temp, ty.clone());
        }

        pub(crate) fn with_stmt(mut self, stmt: ir::Statement) -> LoweringState {
            self.stmts.push(stmt);
            self
        }

        pub(crate) fn empty() -> LoweringState {
            LoweringState { stmts: Vec::new(), temps: HashMap::new(), next_temp: 0 }
        }

        pub(crate) fn destruct(self) -> (Vec<ir::Statement>, HashMap<ir::Temporary, ir::Type>) {
            (self.stmts, self.temps)
        }

        pub(crate) fn finalize(self) -> (Vec<ir::Statement>, Vec<ir::Type>) {
            let mut temps: Vec<(ir::Temporary, ir::Type)> = self.temps.into_iter().collect();
            temps.sort_unstable_by_key(|(temp, _)| temp.0);
            // Make sure every temporary has its type.
            // Should not be necessary, but this is a crucial property, so better safe than sorry.
            temps.iter().enumerate().for_each(|(i, (temp, _))| {
                assert_eq!(
                    i, temp.0 as usize,
                    "Temporary {} does not have a type! Found type for Temporary {} instead.",
                    i, temp.0
                )
            });
            let temporaries = temps.into_iter().map(|(_, ty)| ty).collect();
            (self.stmts, temporaries)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::*;

    fn spec_to_ir(spec: &str) -> LolaIR {
        crate::parse(spec)
    }

    fn check_stream_number(
        ir: &LolaIR,
        inputs: usize,
        outputs: usize,
        time: usize,
        event: usize,
        param: usize,
        sliding: usize,
        triggers: usize,
    ) {
        assert_eq!(ir.inputs.len(), inputs);
        assert_eq!(ir.outputs.len(), outputs);
        assert_eq!(ir.time_driven.len(), time);
        assert_eq!(ir.event_driven.len(), event);
        assert_eq!(ir.parametrized.len(), param);
        assert_eq!(ir.sliding_windows.len(), sliding);
        assert_eq!(ir.triggers.len(), triggers);
    }

    #[test]
    fn lower_one_input() {
        let ir = spec_to_ir("input a: Int32");
        check_stream_number(&ir, 1, 0, 0, 0, 0, 0, 0);
    }

    #[test]
    fn lower_triggers() {
        let ir = spec_to_ir("input a: Int32\ntrigger a > 50\ntrigger a < 30 \"So low...\"");
        // Note: Each trigger needs to be accounted for as an output stream.
        check_stream_number(&ir, 1, 2, 0, 2, 0, 0, 2);
    }

    #[test]
    fn lower_one_output_event() {
        let ir = spec_to_ir("output a: Int32 := 34");
        check_stream_number(&ir, 0, 1, 0, 1, 0, 0, 0);
    }

    #[test]
    fn lower_one_output_event_float() {
        let ir = spec_to_ir("output a: Float64 := 34.");
        check_stream_number(&ir, 0, 1, 0, 1, 0, 0, 0);
    }

    #[test]
    fn lower_one_output_time() {
        let ir = spec_to_ir("output a: Int32 @1Hz := 34");
        check_stream_number(&ir, 0, 1, 1, 0, 0, 0, 0);
    }

    #[test]
    fn lower_one_sliding() {
        let ir = spec_to_ir("input a: Int32 output b: Int64 @1Hz := a.aggregate(over: 3s, using: sum).defaults(to: 4)");
        check_stream_number(&ir, 1, 1, 1, 0, 0, 1, 0);
    }

    #[test]
    #[ignore] // Trigger needs to be periodic, and if it were event based, the type checker needs to reject the access w/o s&h or default.
    fn lower_multiple_streams_with_windows() {
        let ir = spec_to_ir(
            "\
             input a: Int32 \n\
             input b: Bool \n\
             output c: Int32 := a \n\
             output d: Int64 @1Hz := a[3s, sum].defaults(to: 19) \n\
             output e: Bool := a > 4 && b \n\
             output f: Int64 @1Hz := if (e ! true) then (c ! 0) else 0 \n\
             output g: Float64 @0.1Hz :=  cast(f[10s, avg].defaults(to: 0)) \n\
             trigger g > 17.0 \
             ",
        );
        check_stream_number(&ir, 2, 6, 5, 1, 0, 2, 1);
    }

    #[test]
    fn lower_constant_expression() {
        let ir = spec_to_ir("output a: Int32 := 3+4*7");
        let stream: &OutputStream = &ir.outputs[0];

        let ty = Type::Int(crate::ty::IntTy::I32);

        assert_eq!(stream.ty, ty);

        let expr = &stream.expr;
        // Load constant 3, load constant 4, load constant 7, add, multiply
        let mut mult = None;
        let mut add = None;
        let mut load_3 = None;
        let mut load_4 = None;
        let mut load_7 = None;
        for stmt in &expr.stmts {
            match stmt.op {
                Op::ArithLog(ArithLogOp::Mul) => {
                    assert!(mult.is_none());
                    mult = Some(stmt);
                }
                Op::ArithLog(ArithLogOp::Add) => {
                    assert!(add.is_none());
                    add = Some(stmt);
                }
                Op::LoadConstant(Constant::Int(7)) => {
                    assert!(load_7.is_none());
                    load_7 = Some(stmt);
                }
                Op::LoadConstant(Constant::Int(4)) => {
                    assert!(load_4.is_none());
                    load_4 = Some(stmt);
                }
                Op::LoadConstant(Constant::Int(3)) => {
                    assert!(load_3.is_none());
                    load_3 = Some(stmt);
                }
                _ => panic!("There shouldn't be any other statement."),
            }
        }
        assert!(mult.is_some());
        assert!(add.is_some());
        assert!(load_3.is_some());
        assert!(load_4.is_some());
        assert!(load_7.is_some());

        let mult = mult.unwrap();
        let add = add.unwrap();
        let load_3 = load_3.unwrap();
        let load_4 = load_4.unwrap();
        let load_7 = load_7.unwrap();

        // 3+4*7
        assert_eq!(mult.args[0], load_4.target);
        assert_eq!(mult.args[1], load_7.target);
        assert_eq!(add.args[0], load_3.target);
        assert_eq!(add.args[1], mult.target);

        for ty in &expr.temporaries {
            assert_eq!(ty, &stream.ty);
        }
    }

    #[test]
    fn lower_expr_with_widening() {
        let ir = spec_to_ir("input a: UInt8 output b: UInt16 := a");
        let stream = &ir.outputs[0];

        let expr = &stream.expr;
        let stmts = &expr.stmts;
        assert_eq!(stmts.len(), 2);

        let lu_target = match &stmts[0] {
            Statement { target, op: Op::SyncStreamLookup(_), args } if args.is_empty() => *target,
            _ => panic!("Incorrect stream lookup."),
        };
        let res_target = match &stmts[1] {
            Statement { target, op: Op::Convert, args } if args.len() == 1 => {
                assert_eq!(args[0], lu_target);
                *target
            }
            _ => panic!(),
        };

        let lu_type = Type::UInt(crate::ty::UIntTy::U8);
        let result_type = Type::UInt(crate::ty::UIntTy::U16);

        assert_eq!(lu_type, expr.temporaries[lu_target.0 as usize]);
        assert_eq!(result_type, expr.temporaries[res_target.0 as usize]);
    }

    #[test]
    fn lower_function_expression() {
        let ir = spec_to_ir("import math input a: Float32 output v: Float64 := sqrt(a)");
        let stream: &OutputStream = &ir.outputs[0];

        let ty = Type::Float(crate::ty::FloatTy::F64);

        assert_eq!(stream.ty, ty);

        let expr = &stream.expr;
        assert_eq!(expr.stmts.len(), 2);

        let load = &expr.stmts[0];

        match &load.op {
            Op::SyncStreamLookup(StreamInstance { reference, arguments }) => {
                assert!(arguments.is_empty(), "Lookup does not have arguments.");
                match reference {
                    StreamReference::InRef(0) => {}
                    _ => panic!("Incorrect StreamReference"),
                }
            }
            _ => panic!("Need to load the constant first."),
        };

        let sqrt = &expr.stmts[1];

        match &sqrt.op {
            Op::Function(s) => assert_eq!("sqrt", s),
            _ => panic!("Need to apply the function!"),
        }
    }

    #[test]
    fn lower_cast_expression() {
        let ir = spec_to_ir("input a: Float64 output v: Float32 := cast(a)");
        let stream: &OutputStream = &ir.outputs[0];

        let ty = Type::Float(crate::ty::FloatTy::F32);

        assert_eq!(stream.ty, ty);

        let expr = &stream.expr;
        assert_eq!(expr.stmts.len(), 2);

        let load = &expr.stmts[0];

        match &load.op {
            Op::SyncStreamLookup(StreamInstance { reference, arguments }) => {
                assert!(arguments.is_empty(), "Lookup does not have arguments.");
                match reference {
                    StreamReference::InRef(0) => {}
                    _ => panic!("Incorrect StreamReference"),
                }
            }
            _ => panic!("Need to load the constant first."),
        };

        let cast = &expr.stmts[1];

        match &cast.op {
            Op::Convert => {}
            _ => panic!("Need to convert the value!"),
        }
    }

    #[test]
    fn lower_function_expression_regex() {
        let ir = spec_to_ir("import regex\ninput a: String output v: Bool := matches_regex(a, r\"a*b\")");

        let stream: &OutputStream = &ir.outputs[0];

        let ty = Type::Bool;

        assert_eq!(stream.ty, ty);

        let expr = &stream.expr;
        assert_eq!(expr.stmts.len(), 3);

        let load = &expr.stmts[0];

        match &load.op {
            Op::SyncStreamLookup(StreamInstance { reference, arguments }) => {
                assert!(arguments.is_empty(), "Lookup does not have arguments.");
                match reference {
                    StreamReference::InRef(0) => {}
                    _ => panic!("Incorrect StreamReference"),
                }
            }
            _ => panic!("Need to load the constant first."),
        };

        let constant = &expr.stmts[1];
        match &constant.op {
            Op::LoadConstant(Constant::Str(s)) => assert_eq!(s, "a*b"),
            c => panic!("expected constant, found {:?}", c),
        }

        let regex_match = &expr.stmts[2];

        match &regex_match.op {
            Op::Function(s) => assert_eq!(s, "matches_regex"),
            _ => panic!("Need to apply the function!"),
        }
    }

    #[test]
    fn input_lookup() {
        let ir = spec_to_ir("input a: Int32");
        let inp = &ir.inputs[0];
        assert_eq!(inp, ir.get_in(inp.reference));
    }

    #[test]
    fn output_lookup() {
        let ir = spec_to_ir("output b: Int32 := 3 + 4");
        let outp = &ir.outputs[0];
        assert_eq!(outp, ir.get_out(outp.reference));
    }

    #[test]
    fn window_lookup() {
        let ir = spec_to_ir("input a: Int32 output b: Int32 @1Hz := a.aggregate(over: 3s, using: sum).defaults(to: 3)");
        let window = &ir.sliding_windows[0];
        assert_eq!(window, ir.get_window(window.reference));
    }

    #[test]
    fn test_superfluous_casts_after_lookup() {
        let ir = spec_to_ir("input v: Bool\n output b: Bool := v & v[-1].defaults(to: false)");
        // We need a sync lookup, an async lookup, load false, and the and operation.
        assert_eq!(ir.outputs[0].expr.stmts.len(), 4);
    }

    #[test]
    #[should_panic]
    fn invalid_lookup_no_out() {
        let ir = spec_to_ir("input a: Int32");
        let r = StreamReference::OutRef(0);
        ir.get_in(r);
    }

    #[test]
    #[should_panic]
    fn invalid_lookup_index_oob() {
        let ir = spec_to_ir("input a: Int32");
        let r = StreamReference::InRef(24);
        ir.get_in(r);
    }

    #[test]
    fn dependency_test() {
        let ir = spec_to_ir(
            "input a: Int32\ninput b: Int32\ninput c: Int32\noutput d: Int32 := a + b + (b[-1]?0) + (a[-2]?0) + c",
        );
        let mut in_refs: [StreamReference; 3] =
            [StreamReference::InRef(5), StreamReference::InRef(5), StreamReference::InRef(5)];
        for i in ir.inputs {
            if i.name == "a" {
                in_refs[0] = i.reference;
            }
            if i.name == "b" {
                in_refs[1] = i.reference;
            }
            if i.name == "c" {
                in_refs[2] = i.reference;
            }
        }
        let out_dep = &ir.outputs[0].outgoing_dependencies;
        assert_eq!(out_dep.len(), 3);
        let a_dep = out_dep.into_iter().find(|&x| x.stream == in_refs[0]).expect("a dependencies not found");
        let b_dep = out_dep.into_iter().find(|&x| x.stream == in_refs[1]).expect("b dependencies not found");
        let c_dep = out_dep.into_iter().find(|&x| x.stream == in_refs[2]).expect("c dependencies not found");
        assert_eq!(a_dep.offsets.len(), 2);
        assert_eq!(b_dep.offsets.len(), 2);
        assert_eq!(c_dep.offsets.len(), 1);
        assert!(a_dep.offsets.contains(&Offset::PastDiscreteOffset(0)));
        assert!(a_dep.offsets.contains(&Offset::PastDiscreteOffset(2)));
        assert!(b_dep.offsets.contains(&Offset::PastDiscreteOffset(0)));
        assert!(b_dep.offsets.contains(&Offset::PastDiscreteOffset(1)));
        assert!(c_dep.offsets.contains(&Offset::PastDiscreteOffset(0)));
    }
}
