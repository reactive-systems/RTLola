use crate::analysis::naming::DeclarationTable;
use crate::analysis::typing::TypeTable;
// Only import the unambiguous Nodes, use `ast::`/`ir::` prefix for disambiguation.
use crate::analysis::naming::Declaration;
use crate::ast;
use crate::ast::{ExpressionKind, LolaSpec};
use crate::ir;
use crate::ir::{
    EventDrivenStream, LolaIR, MemorizationBound, ParametrizedStream, StreamReference,
    TimeDrivenStream, WindowReference,
};
use crate::ty::TimingInfo;
use ast_node::{AstNode, NodeId};
use std::collections::HashMap;
use std::time::Duration;

use crate::analysis::graph_based_analysis::evaluation_order::{EvalOrder, EvaluationOrderResult};
use crate::analysis::graph_based_analysis::space_requirements::{
    SpaceRequirements as MemoryTable, TrackingRequirements,
};
//use crate::analysis::graph_based_analysis::future_dependency::FutureDependentStreams;
use crate::analysis::graph_based_analysis::{ComputeStep, StorageRequirement, TrackingRequirement};

use crate::analysis::AnalysisResult;

type EvalTable = HashMap<NodeId, u32>;

/// Amazing macro for hashmap initialization. (credits: https://stackoverflow.com/questions/28392008/more-concise-hashmap-initialization)
/// Use by calling hashmap!['a' => 1, 'b' => 4, 'c' => 7].
macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

pub(crate) struct Lowering<'a> {
    ast: &'a LolaSpec,
    ref_lookup: HashMap<NodeId, StreamReference>,
    dt: &'a DeclarationTable<'a>,
    tt: &'a TypeTable,
    et: EvalTable,
    mt: &'a MemoryTable,
    tr: &'a TrackingRequirements,
    ir: LolaIR,
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
    }

    fn link_windows(&mut self) {
        // Extract and copy relevant information before-hand to avoid double burrow.
        let essences: Vec<(StreamReference, WindowReference)> = self
            .ir
            .sliding_windows
            .iter()
            .map(|window| (window.target, window.reference))
            .collect();
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
        let nid = *input.id();
        let ast_req = self.get_memory(input.id());
        let memory_bound = self.lower_storage_req(ast_req);
        let reference = self.get_ref(input.id());
        let layer = self.get_layer(input.id());

        let trackings = self.collect_tracking_info(nid, None);

        let input = ir::InputStream {
            name: input.name.name.clone(),
            ty: self.lower_value_type(*input.id()),
            dependent_streams: trackings,
            dependent_windows: Vec::new(),
            layer, // Not necessarily 0 when parametrized inputs exist.
            memory_bound,
            reference,
        };

        let debug_clone = input.clone();
        self.ir.inputs.push(input);

        assert_eq!(self.ir.get_in(reference), &debug_clone, "Bug in implementation: Output vector in IR changed between creation of reference and insertion of stream.");
    }

    fn collect_tracking_info(
        &self,
        nid: NodeId,
        time_driven: Option<TimeDrivenStream>,
    ) -> Vec<ir::Tracking> {
        let dependent = self.find_depending_streams(nid);
        assert!(
            dependent.iter().all(|(_, req)| match req {
                TrackingRequirement::Unbounded => false,
                _ => true,
            }),
            "Unbounded dependencies are not supported, yet."
        );

        dependent
            .into_iter()
            .map(|(trackee, req)| self.lower_tracking_req(time_driven, trackee, req))
            .collect()
    }

    /// Does *not* add dependent windows, yet.
    fn lower_output(&mut self, ast_output: &ast::Output) {
        let nid = *ast_output.id();
        let ast_req = self.get_memory(ast_output.id());
        let memory_bound = self.lower_storage_req(ast_req);
        let layer = self.get_layer(ast_output.id());
        let reference = self.get_ref(ast_output.id());
        let time_driven = self.check_time_driven(&ast_output, reference);
        let parametrized = self.check_parametrized(&ast_output, reference);

        let trackings = self.collect_tracking_info(nid, time_driven);

        let output = ir::OutputStream {
            name: ast_output.name.name.clone(),
            ty: self.lower_value_type(*ast_output.id()),
            expr: self.lower_expression(&ast_output.expression),
            dependent_streams: trackings,
            dependent_windows: Vec::new(),
            memory_bound,
            layer,
            reference,
        };

        let debug_clone = output.clone();
        self.ir.outputs.push(output);
        assert_eq!(self.ir.get_out(reference), &debug_clone, "Bug in implementation: Output vector in IR changed between creation of reference and insertion of stream.");

        if let Some(td_ref) = time_driven {
            self.ir.time_driven.push(td_ref)
        } else {
            self.ir.event_driven.push(EventDrivenStream { reference })
        }

        if let Some(param_ref) = parametrized {
            self.ir.parametrized.push(param_ref);
        }
    }

    fn lower_tracking_req(
        &self,
        tracker: Option<TimeDrivenStream>,
        trackee: NodeId,
        req: TrackingRequirement,
    ) -> ir::Tracking {
        let trackee = self.get_ref(&trackee);
        match req {
            TrackingRequirement::Unbounded => ir::Tracking::All(trackee),
            TrackingRequirement::Finite(num) => {
                let rate = tracker
                    .map(|tds| tds.extend_rate)
                    .unwrap_or(Duration::from_secs(0));
                ir::Tracking::Bounded {
                    trackee,
                    num: num as u128,
                    rate,
                }
            }
            TrackingRequirement::Future => unimplemented!(),
        }
    }

    fn lower_window(&mut self, expr: &ast::Expression) -> WindowReference {
        if let ExpressionKind::Lookup(_, offset, Some(op)) = &expr.kind {
            let duration = match self.lower_offset(offset) {
                ir::Offset::PastDiscreteOffset(_)
                | ir::Offset::FutureDiscreteOffset(_)
                | ir::Offset::PastRealTimeOffset(_) => {
                    panic!("Bug: Should be caught in preceeding step.")
                }
                ir::Offset::FutureRealTimeOffset(dur) => dur,
            };
            let target = self.extract_target_from_lookup(&expr.kind);
            let reference = WindowReference {
                ix: self.ir.sliding_windows.len(),
            };
            let window = ir::SlidingWindow {
                target,
                duration,
                op: self.lower_window_op(*op),
                reference,
            };
            self.ir.sliding_windows.push(window);
            reference
        } else {
            panic!("Bug in implementation: Called `lower_window` on non-window expression.")
        }
    }

    fn extract_target_from_lookup(&self, lookup: &ExpressionKind) -> StreamReference {
        if let ExpressionKind::Lookup(inst, _, _) = lookup {
            let decl = self.get_decl(inst.id());
            let nid = match decl {
                Declaration::Out(out) => out.id(),
                Declaration::In(inp) => inp.id(),
                Declaration::Const(_) => unimplemented!(),
                Declaration::Param(_) => unimplemented!(),
                Declaration::Type(_) | Declaration::Func(_) => {
                    panic!("Bug in implementation: Invalid lookup.")
                }
            };
            *self.ref_lookup.get(nid).expect("Bug in ReferenceLookup.")
        } else {
            panic!("Bug in implementation: Called `extract_target_from_lookup` on non-lookup expression.")
        }
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
            if temp_spec
                .ext
                .as_ref()
                .map(|e| e.target.is_none() && e.freq.is_some())
                .unwrap_or(false)
                && temp_spec.inv.is_none()
                && temp_spec.ter.is_none()
            {
                None
            } else {
                // TODO: Finalize and implement parametrization.
                Some(ParametrizedStream {
                    reference,
                    params: ast_output
                        .params
                        .iter()
                        .map(|p| self.lower_param(p))
                        .collect(),
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
        ir::Parameter {
            name: param.name.name.clone(),
            ty: self.lower_value_type(*param.id()),
        }
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

    fn lower_expression(&mut self, expression: &ast::Expression) -> ir::Expression {
        let mut mm = MemoryManager::new();

        let state = self.lower_subexpression(expression, &mut mm, None);

        let mut temps: Vec<(ir::Temporary, ir::Type)> = state.temps.clone().into_iter().collect();
        temps.sort_unstable_by_key(|(temp, _)| temp.0);
        // Make sure every temporary has its type.
        temps.iter().enumerate().for_each(|(i, (temp, _))| {
            assert_eq!(
                i, temp.0 as usize,
                "Temporary {} does not have a type! Found type for Temporary {} instead.",
                i, temp.0
            )
        });
        let temporaries = temps.into_iter().map(|(_, ty)| ty).collect();

        ir::Expression {
            stmts: state.stmts,
            temporaries,
        }
    }

    fn lower_subexpression(
        &mut self,
        expr: &ast::Expression,
        mm: &mut MemoryManager,
        target_temp: Option<ir::Temporary>,
    ) -> LoweringState {
        use crate::ir::{Op, Statement};

        let result_type = self.lower_value_type(expr._id);

        match &expr.kind {
            ExpressionKind::Lit(l) => {
                let result = target_temp.unwrap_or_else(|| mm.temp_for_type(&result_type));
                let op = Op::LoadConstant(self.lower_literal(l));
                let args = Vec::new();
                let stmt = Statement {
                    target: result,
                    op,
                    args,
                };
                LoweringState::new(vec![stmt], hashmap![result => result_type])
            }
            ExpressionKind::Ident(_) => match self.get_decl(expr.id()) {
                Declaration::In(inp) => {
                    let result = target_temp.unwrap_or_else(|| mm.temp_for_type(&result_type));
                    let lu_target = ir::StreamInstance {
                        reference: self.get_ref(inp.id()),
                        arguments: Vec::new(),
                    };
                    let op = Op::SyncStreamLookup {
                        instance: lu_target,
                    };
                    let stmt = Statement {
                        target: result,
                        op,
                        args: Vec::new(),
                    };
                    LoweringState::new(vec![stmt], hashmap![result => result_type])
                }
                Declaration::Out(out) => {
                    let result = target_temp.unwrap_or_else(|| mm.temp_for_type(&result_type));
                    let lu_target = ir::StreamInstance {
                        reference: self.get_ref(out.id()),
                        arguments: Vec::new(),
                    };
                    let op = Op::SyncStreamLookup {
                        instance: lu_target,
                    };
                    let stmt = Statement {
                        target: result,
                        op,
                        args: Vec::new(),
                    };
                    LoweringState::new(vec![stmt], hashmap![result => result_type])
                }
                Declaration::Param(_) | Declaration::Const(_) => unimplemented!(),
                Declaration::Type(_) | Declaration::Func(_) => unreachable!("Bug in TypeChecker."),
            },
            ExpressionKind::Default(e, dft) => {
                if let ExpressionKind::Lookup(_, _, _) = &e.kind {
                    self.lower_lookup_expression(e, dft, mm, target_temp)
                } else {
                    // A "stray" default expression such as `5 ? 3` is valid, but a no-op.
                    // Thus, print a warning. Evaluating the expression is necessary, the dft can be skipped.
                    println!("WARNING: No-Op Default operation!");
                    self.lower_subexpression(e, mm, target_temp)
                }
            }
            ExpressionKind::Lookup(_, _, _) => {
                // Stray lookup without any default expression surrounding it, see `ExpressionKind::Default` case.
                // This is only valid for sync accesses, i.e. the offset is 0.
                // TODO: Double check: is a [0] access even allowed anymore? Or should it be sample&hold or a no-offset lookup instead.
                // TODO: If it is allowed, transform lookup into ExpressionKind::Ident, call recursively. Yes, [0] is a no-op!
                unimplemented!()
            }
            ExpressionKind::Binary(ast_op, lhs, rhs) => {
                let lhs_state = self.lower_subexpression(lhs, mm, None);
                let rhs_state = self.lower_subexpression(rhs, mm, None);
                let args = vec![lhs_state.get_target(), rhs_state.get_target()];
                let state = lhs_state.merge_with(rhs_state);
                let op = Op::ArithLog(Lowering::lower_bin_op(*ast_op));
                let result = target_temp.unwrap_or_else(|| mm.temp_for_type(&result_type));
                let stmt = Statement {
                    target: result,
                    op,
                    args,
                };
                let new_state = LoweringState::new(vec![stmt], hashmap![result => result_type]);
                state.merge_with(new_state)
            }
            ExpressionKind::Unary(ast_op, operand) => {
                let state = self.lower_subexpression(operand, mm, None);
                let op = Op::ArithLog(Lowering::lower_un_op(*ast_op));
                let result = target_temp.unwrap_or_else(|| mm.temp_for_type(&result_type));
                let stmt = Statement {
                    target: result,
                    op,
                    args: vec![state.get_target()],
                };
                let new_state = LoweringState::new(vec![stmt], hashmap![result => result_type]);
                state.merge_with(new_state)
            }
            ExpressionKind::Ite(cond, cons, alt) => {
                let cond_state = self.lower_subexpression(cond, mm, None);
                let result = target_temp.unwrap_or_else(|| mm.temp_for_type(&result_type));
                let cons_state = self.lower_subexpression(cons, mm, Some(result));
                let alt_state = self.lower_subexpression(alt, mm, Some(result));

                let op = Op::Ite {
                    consequence: cons_state.stmts,
                    alternative: alt_state.stmts,
                };
                let stmt = Statement {
                    target: result,
                    op,
                    args: vec![cond_state.get_target()],
                };

                // Result temp and type already registered in cons and alt states.
                cond_state
                    .merge_temps(cons_state.temps)
                    .merge_temps(alt_state.temps)
                    .merge_with(LoweringState::new(vec![stmt], HashMap::new()))
            }
            ExpressionKind::ParenthesizedExpression(_, e, _) => {
                self.lower_subexpression(e, mm, target_temp)
            }
            ExpressionKind::MissingExpression() => {
                panic!("How wasn't this caught in a preceding step?!")
            }
            ExpressionKind::Tuple(exprs) => {
                let (state, values) = self.lower_expression_list(exprs, mm);
                let result = target_temp.unwrap_or_else(|| mm.temp_for_type(&result_type));
                let stmt = Statement {
                    target: result,
                    op: Op::Tuple,
                    args: values,
                };
                state.merge_with(LoweringState::new(
                    vec![stmt],
                    hashmap![result => result_type],
                ))
            }
            ExpressionKind::Function(kind, _, args) => {
                let ir_kind = self.lower_function_kind(kind, args.first());
                let (state, values) = self.lower_expression_list(args, mm);
                let result = target_temp.unwrap_or_else(|| mm.temp_for_type(&result_type));
                let stmt = Statement {
                    target: result,
                    op: Op::Function(ir_kind),
                    args: values,
                };
                state.merge_with(LoweringState::new(
                    vec![stmt],
                    hashmap![result => result_type],
                ))
            }
            ExpressionKind::Field(_, _) => unimplemented!(),
            ExpressionKind::Method(_, _, _, _) => unimplemented!(),
        }
    }

    fn lower_function_kind(
        &self,
        ident: &crate::parse::Ident,
        first_arg: Option<&Box<ast::Expression>>,
    ) -> ir::FunctionKind {
        unimplemented!("check declaration table + typing information instead");
    }

    fn lower_expression_list(
        &mut self,
        expressions: &Vec<Box<ast::Expression>>,
        mm: &mut MemoryManager,
    ) -> (LoweringState, Vec<ir::Temporary>) {
        let states: Vec<LoweringState> = expressions
            .iter()
            .map(|e| self.lower_subexpression(e, mm, None))
            .collect();
        let values: Vec<ir::Temporary> = states.iter().map(|s| s.get_target()).collect();
        let state = states
            .into_iter()
            .fold(LoweringState::empty(), |accu, s| accu.merge_with(s));
        (state, values)
    }

    fn lower_lookup_expression(
        &mut self,
        lookup_expr: &ast::Expression,
        dft: &ast::Expression,
        mm: &mut MemoryManager,
        target_temp: Option<ir::Temporary>,
    ) -> LoweringState {
        if let ExpressionKind::Lookup(instance, offset, op) = &lookup_expr.kind {
            // Translate look-up, add default value.
            let target_ref = self.extract_target_from_lookup(&lookup_expr.kind);

            // Compute all arguments.
            let mut state = LoweringState::empty();
            let mut lookup_args = Vec::new();
            for arg in &instance.arguments {
                state = state.merge_with(self.lower_subexpression(arg.as_ref(), mm, None));
                lookup_args.push(state.get_target());
            }

            // Compute default value. Write it in the target temp to avoid unnecessary copying later.
            state = state.merge_with(self.lower_subexpression(dft, mm, target_temp));
            let default_temp = state.get_target();

            let op = if op.is_some() {
                let window_ref = self.lower_window(&lookup_expr);
                ir::Op::WindowLookup(window_ref)
            } else {
                let target_instance = ir::StreamInstance {
                    reference: target_ref,
                    arguments: lookup_args,
                };
                ir::Op::StreamLookup {
                    instance: target_instance,
                    offset: self.lower_offset(&offset),
                }
            };

            let lookup_type = self.lower_value_type(lookup_expr._id);
            let lookup_result = target_temp.unwrap_or_else(|| mm.temp_for_type(&lookup_type));
            let lookup_stmt = ir::Statement {
                target: lookup_result,
                op,
                args: vec![default_temp],
            };
            let lookup_state =
                LoweringState::new(vec![lookup_stmt], hashmap![lookup_result => lookup_type]);
            state.merge_with(lookup_state)
        } else {
            panic!("Called `lower_lookup_expression` on a non-lookup expression.");
        }
    }

    fn lower_offset(&self, offset: &ast::Offset) -> ir::Offset {
        match offset {
            ast::Offset::RealTimeOffset(e, unit) => {
                let (duration, pos) = self.lower_time_spec(e, unit);
                if pos {
                    ir::Offset::FutureRealTimeOffset(duration)
                } else {
                    ir::Offset::PastRealTimeOffset(duration)
                }
            }
            ast::Offset::DiscreteOffset(e) => match self.extract_literal(e) {
                ir::Constant::Int(i) => {
                    if i > 0 {
                        ir::Offset::FutureDiscreteOffset(i as u128)
                    } else {
                        ir::Offset::PastDiscreteOffset(i.abs() as u128)
                    }
                }
                _ => unreachable!("Eradicated in preceding step."),
            },
        }
    }

    /// Returns a duration representing the `lit` in combination with `unit`. The bool flag
    /// is true if the literal is strictly greater than 0.
    fn lower_time_spec(&self, e: &ast::Expression, unit: &ast::TimeUnit) -> (Duration, bool) {
        use crate::ast::TimeUnit;
        use crate::ir::Constant;

        let factor: u64 = match unit {
            TimeUnit::NanoSecond => 1u64,
            TimeUnit::MicroSecond => 10u64.pow(3),
            TimeUnit::MilliSecond => 10u64.pow(6),
            TimeUnit::Second => 10u64.pow(9),
            TimeUnit::Minute => 10u64.pow(9) * 60,
            TimeUnit::Hour => 10u64.pow(9) * 60 * 60,
            TimeUnit::Day => 10u64.pow(9) * 60 * 60 * 24,
            TimeUnit::Week => 10u64.pow(9) * 60 * 24 * 24 * 7,
            TimeUnit::Year => 10u64.pow(9) * 60 * 24 * 24 * 7 * 365, // fits in u57
        };
        match self.extract_literal(e) {
            Constant::Int(i) => {
                // TODO: Improve: Robust against overflows.
                let value = i as u128 * u128::from(factor); // Multiplication might fail.
                let secs = (value / 10u128.pow(9)) as u64; // Cast might fail.
                let nanos = (value % 10u128.pow(9)) as u32; // Perfectly safe cast to u32.
                (std::time::Duration::new(secs, nanos), i > 0)
            }
            Constant::Float(f) => {
                // TODO: Improve: Robust against overflows and inaccuracies.
                let value = f * factor as f64;
                let secs = (value / 1_000_000_000f64) as u64;
                let nanos = (value % 1_000_000_000f64) as u32;
                (std::time::Duration::new(secs, nanos), f > 0.0)
            }
            _ => unreachable!("Eradicated in preceding step."),
        }
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
            LitKind::Int(i) => ir::Constant::Int(*i),
            LitKind::Float(f) => ir::Constant::Float(*f),
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

    fn check_time_driven(
        &mut self,
        output: &ast::Output,
        reference: StreamReference,
    ) -> Option<TimeDrivenStream> {
        match &self.tt.get_stream_type(output._id).timing {
            TimingInfo::RealTime(f) => Some(TimeDrivenStream {
                reference,
                extend_rate: f.d,
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

    fn create_ref_lookup(
        inputs: &Vec<ast::Input>,
        outputs: &Vec<ast::Output>,
    ) -> HashMap<NodeId, StreamReference> {
        let ins = inputs
            .iter()
            .enumerate()
            .map(|(ix, i)| (*i.id(), StreamReference::InRef(ix)));
        let outs = outputs
            .iter()
            .enumerate()
            .map(|(ix, o)| (*o.id(), StreamReference::OutRef(ix))); // Re-start indexing @ 0.
        ins.chain(outs).collect()
    }

    fn order_to_table(eo: &EvaluationOrderResult) -> EvalTable {
        fn extr_id(step: &ComputeStep) -> NodeId {
            // TODO: Rework when parameters actually exist.
            use self::ComputeStep::*;
            match step {
                Evaluate(nid) | Extend(nid) | Invoke(nid) | Terminate(nid) => *nid,
            }
        }
        let o2t = |eo: &EvalOrder| {
            let mut res = Vec::new();
            for (ix, layer) in eo.iter().enumerate() {
                let vals = layer.iter().map(|s| (extr_id(s), ix as u32));
                res.extend(vals);
            }
            res.into_iter()
        };
        o2t(&eo.periodic_streams_order)
            .chain(o2t(&eo.event_based_streams_order))
            .collect()
    }

    fn get_decl(&self, nid: &NodeId) -> &Declaration {
        self.dt.get(nid).expect("Bug in DeclarationTable.")
    }

    fn get_layer(&self, nid: &NodeId) -> u32 {
        *self.et.get(nid).expect("Bug in EvaluationOrder.")
    }

    fn get_memory(&self, nid: &NodeId) -> StorageRequirement {
        *self.mt.get(nid).expect("Bug in MemoryTable.")
    }

    fn get_ref(&self, nid: &NodeId) -> StreamReference {
        *self.ref_lookup.get(nid).expect("Bug in ReferenceLookup.")
    }
}

#[derive(Debug)]
struct LoweringState {
    stmts: Vec<ir::Statement>,
    temps: HashMap<ir::Temporary, ir::Type>,
}

impl LoweringState {
    fn new(stmts: Vec<ir::Statement>, temps: HashMap<ir::Temporary, ir::Type>) -> LoweringState {
        LoweringState { stmts, temps }
    }

    fn get_target(&self) -> ir::Temporary {
        self.stmts
            .last()
            .expect("A expression needs to return a value.")
            .target
    }

    fn merge_with(mut self, other: LoweringState) -> LoweringState {
        let mut map = HashMap::new();
        for (key, ty) in self.temps.iter().chain(other.temps.iter()) {
            if let Some(conflict) = map.get(key) {
                panic!(
                    "Bug: Cannot assign first type {:?} and then {:?} to the same temporary {:?}.",
                    conflict, ty, key
                );
            } else {
                map.insert(*key, ty.clone());
            }
        }
        // Enter new values.
        self.temps = map;
        self.stmts.extend(other.stmts);
        self
    }

    fn merge_temps(self, other: HashMap<ir::Temporary, ir::Type>) -> LoweringState {
        self.merge_with(LoweringState::new(Vec::new(), other))
    }

    fn empty() -> LoweringState {
        LoweringState::new(Vec::new(), HashMap::new())
    }
}

/// Manages memory for evaluation of expressions.
struct MemoryManager {
    next_temp: u32,
}

impl MemoryManager {
    fn new() -> MemoryManager {
        MemoryManager { next_temp: 0u32 }
    }

    fn temp_for_type(&mut self, _t: &ir::Type) -> ir::Temporary {
        self.next_temp += 1;
        ir::Temporary(self.next_temp - 1)
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
    fn lower_one_output_event() {
        let ir = spec_to_ir("output a: Int32 := 34");
        check_stream_number(&ir, 0, 1, 0, 1, 0, 0, 0);
    }

    #[test]
    fn lower_one_output_time() {
        let ir = spec_to_ir("output a: Int32 { extend @1Hz } := 34");
        check_stream_number(&ir, 0, 1, 1, 0, 0, 0, 0);
    }

    #[test]
    fn lower_one_sliding() {
        let ir = spec_to_ir("input a: Int32 output b: Int64 { extend @1Hz } := a[3s, sum] ? 4");
        check_stream_number(&ir, 1, 1, 1, 0, 0, 1, 0);
    }

    #[test]
    #[ignore]
    fn lower_multiple_streams_with_windows() {
        let ir = spec_to_ir(
            "\
             input a: Int32 \n\
             input b: Bool \n\
             output c: Int32 := a \n\
             output d: Int64 { extend @1Hz } := a[3s, sum] ? 19 \n\
             output e: Bool := a > 4 && b \n\
             output f: Int64 { extend @1Hz } := if e[0] then c[0] else 0 \n\
             output g: Float64 { extend @0.1Hz } :=  f[10s, avg] \n\
             trigger g > 17.0 \
             ",
        );
        check_stream_number(&ir, 2, 5, 4, 1, 0, 2, 1);
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
        let ir = spec_to_ir("input a: Int32 output b: Int32 { extend @1Hz } := a[3s, sum] ? 3");
        let window = &ir.sliding_windows[0];
        assert_eq!(window, ir.get_window(window.reference));
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
}
