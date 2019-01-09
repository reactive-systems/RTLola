use crate::analysis::naming::DeclarationTable;
use crate::analysis::typing::TypeAnalysis;
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

struct Lowering<'a> {
    ast: &'a LolaSpec,
    ref_lookup: HashMap<NodeId, StreamReference>,
    dt: DeclarationTable<'a>,
    tt: TypeAnalysis<'a>,
    et: EvalTable,
    mt: MemoryTable,
    ir: LolaIR,
    tr: TrackingRequirements,
}

impl<'a> Lowering<'a> {
    pub(crate) fn new(
        ast: &'a LolaSpec,
        dt: DeclarationTable<'a>,
        tt: TypeAnalysis<'a>,
        eval_order: EvaluationOrderResult,
        mt: MemoryTable,
        tr: TrackingRequirements,
    ) -> Lowering<'a> {
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
        Lowering {
            ast,
            ref_lookup: Lowering::create_ref_lookup(&ast.inputs, &ast.outputs),
            dt,
            tt,
            et: Self::order_to_table(eval_order),
            mt,
            ir,
            tr,
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
            ty: self.lower_type(input._id),
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
            ty: self.lower_type(ast_output._id),
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

    /// Returns the flattened result of calling f on each node recursively in `pre_order` or post_order.
    fn collect_expression<T, F>(expr: &'a ast::Expression, f: &F, pre_order: bool) -> Vec<T>
    where
        F: Fn(&'a ast::Expression) -> Vec<T>,
    {
        let recursion = |e| Lowering::collect_expression(e, &f, pre_order);
        let pre = if pre_order {
            f(expr).into_iter()
        } else {
            Vec::new().into_iter()
        };
        let post = || {
            if pre_order {
                Vec::new().into_iter()
            } else {
                f(expr).into_iter()
            }
        };
        match &expr.kind {
            ExpressionKind::Lit(_) => pre.chain(post()).collect(),
            ExpressionKind::Ident(_) => pre.chain(post()).collect(),
            ExpressionKind::Default(e, dft) => pre
                .chain(recursion(e))
                .chain(recursion(dft))
                .chain(post())
                .collect(),
            ExpressionKind::Lookup(inst, _, _) => {
                let args = inst.arguments.iter().flat_map(|a| recursion(a));
                pre.chain(args).chain(post()).collect()
            }
            ExpressionKind::Binary(_, lhs, rhs) => pre
                .chain(recursion(lhs))
                .chain(recursion(rhs))
                .chain(post())
                .collect(),
            ExpressionKind::Unary(_, operand) => {
                pre.chain(recursion(operand)).chain(post()).collect()
            }
            ExpressionKind::Ite(cond, cons, alt) => pre
                .chain(recursion(cond))
                .chain(recursion(cons))
                .chain(recursion(alt))
                .chain(post())
                .collect(),
            ExpressionKind::ParenthesizedExpression(_, e, _) => pre
                .chain(Lowering::collect_expression(e, &f, pre_order))
                .chain(post())
                .collect(),
            ExpressionKind::MissingExpression() => panic!(), // TODO: Eradicate in preceding step.
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
    //
    //    fn find_windows(expr: &ast::Expression) -> Vec<&ast::Expression> {
    //        fn f(e: &ast::Expression) -> Vec<&ast::Expression> {
    //            match &e.kind {
    //                ExpressionKind::Lookup(_, _, Some(_)) => vec![e],
    //                _ => Vec::new()
    //            }
    //        }
    //        Lowering::collect_expression(expr, &f)
    //    }

    fn lower_window(&mut self, expr: &ast::Expression) -> WindowReference {
        if let ExpressionKind::Lookup(inst, offset, Some(op)) = &expr.kind {
            let duration = match offset {
                ast::Offset::DiscreteOffset(_) => panic!(), // TODO: Eradicate in preceding step.
                ast::Offset::RealTimeOffset(expr, unit) => unimplemented!(),
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
        if let ExpressionKind::Lookup(inst, offset, op) = lookup {
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
        if let Some(_temp_spec) = ast_output.template_spec.as_ref() {
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
        } else {
            assert!(ast_output.params.is_empty());
            None
        }
    }

    fn lower_param(&mut self, param: &ast::Parameter) -> ir::Parameter {
        ir::Parameter {
            name: param.name.name.clone(),
            ty: self.lower_type(param._id),
        }
    }

    fn lower_storage_req(&self, req: StorageRequirement) -> MemorizationBound {
        use self::StorageRequirement::*;
        match req {
            StorageRequirement::Finite(b) => MemorizationBound::Bounded(b as u16),
            StorageRequirement::FutureRef(_) | StorageRequirement::Unbounded => {
                MemorizationBound::Unbounded
            }
        }
    }

    fn lower_type(&mut self, id: NodeId) -> ir::Type {
        self.tt.get_type(id).into()
    }
    fn lower_expression(&mut self, expression: &ast::Expression) -> ir::Expression {
        use crate::ir::{Op, Statement, Temporary};
        let mut mm = MemoryManager::new();

        let state = self.lower_subexpression(expression, &mut mm, None);

        // TODO: HashMap<Temporary, ir::Type> -> Vec<ir::Type> with correspondences.
        let mut temps: Vec<(Temporary, ir::Type)> = state.temps.clone().into_iter().collect();
        temps.sort_unstable_by_key(|(temp, ty)| temp.0);
        // Make sure every temporary has its type.
        temps.iter().enumerate().for_each(|(i, (temp, ty))| {
            assert_eq!(
                i, temp.0 as usize,
                "Temporary {} does not have a type! Found type for Temporary {} instead.",
                i, temp.0
            )
        });
        let temporaries = temps.into_iter().map(|(temp, ty)| ty).collect();

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
        use crate::ir::{Op, Statement, Temporary};

        let result_type = self.lower_type(expr._id);

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
            ExpressionKind::Ident(i) => match self.get_decl(expr.id()) {
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
            ExpressionKind::Lookup(inst, _, _) => {
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
            ExpressionKind::Field(e, _) => unimplemented!(),
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
        let init: (LoweringState, Vec<ir::Temporary>) = (LoweringState::empty(), Vec::new());
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

            // Default is a no-op after the lookup, so only handle lookup.
            let target_instance = ir::StreamInstance {
                reference: target_ref,
                arguments: lookup_args,
            };
            let op = ir::Op::StreamLookup {
                instance: target_instance,
                offset: self.lower_offset(&offset),
            };

            let lookup_type = self.lower_type(lookup_expr._id);
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
        unimplemented!()
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
        match self
            .tt
            .get_stream_type(output._id)
            .expect("type information has to be present")
            .timing
        {
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

    fn order_to_table(eo: EvaluationOrderResult) -> EvalTable {
        fn extr_id(step: ComputeStep) -> NodeId {
            // TODO: Rework when parameters actually exist.
            use self::ComputeStep::*;
            match step {
                Evaluate(nid) | Extend(nid) | Invoke(nid) | Terminate(nid) => nid,
            }
        }
        let o2t = |eo: EvalOrder| {
            eo.into_iter()
                .enumerate()
                .flat_map(|(ix, layer)| layer.into_iter().map(move |s| (extr_id(s), ix as u32)))
        };
        o2t(eo.periodic_streams_order)
            .chain(o2t(eo.event_based_streams_order))
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
            .first()
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

    fn merge_temps(mut self, other: HashMap<ir::Temporary, ir::Type>) -> LoweringState {
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
