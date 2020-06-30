use crate::analysis::naming::DeclarationTable;
use crate::ty::check::TypeTable;
// Only import the unambiguous Nodes, use `ast::`/`ir::` prefix for disambiguation.
use crate::analysis::naming::Declaration;
use crate::ast;
use crate::ast::{ExpressionKind, RTLolaAst};
use crate::ir;
use crate::ir::{
    EventDrivenStream, MemorizationBound, RTLolaIR, StreamAccessKind, StreamReference, TimeDrivenStream,
    WindowReference,
};
use crate::parse::NodeId;
use crate::ty::StreamTy;
use std::collections::HashMap;
use std::convert::TryInto;
use std::{rc::Rc, time::Duration};

use crate::analysis::graph_based_analysis::evaluation_order::{EvalOrder, EvaluationOrderResult};
use crate::analysis::graph_based_analysis::space_requirements::{
    SpaceRequirements as MemoryTable, TrackingRequirements,
};
use crate::analysis::{
    graph_based_analysis::{ComputeStep, RequiredInputs, StorageRequirement, TrackingRequirement},
    Report,
};

use num::{traits::Inv, Signed, ToPrimitive};
use uom::si::frequency::hertz;
use uom::si::rational64::Time as UOM_Time;
use uom::si::time::{nanosecond, second};

type EvalTable = HashMap<NodeId, u32>;

pub(crate) struct Lowering<'a> {
    ast: &'a RTLolaAst,
    ref_lookup: HashMap<NodeId, StreamReference>,
    dt: &'a DeclarationTable,
    tt: &'a TypeTable,
    et: EvalTable,
    mt: &'a MemoryTable,
    tr: &'a TrackingRequirements,
    ir: RTLolaIR,
    ri: &'a RequiredInputs,
}

impl<'a> Lowering<'a> {
    pub(crate) fn new(ast: &'a RTLolaAst, analysis_result: &'a Report) -> Lowering<'a> {
        let mut ir = RTLolaIR {
            inputs: Vec::new(),
            outputs: Vec::new(),
            time_driven: Vec::new(),
            event_driven: Vec::new(),
            sliding_windows: Vec::new(),
            triggers: Vec::new(),
        };

        ir.inputs.reserve(ast.inputs.len());
        ir.outputs.reserve(ast.outputs.len());

        Lowering {
            ast,
            ref_lookup: Lowering::create_ref_lookup(&ast.inputs, &ast.outputs),
            dt: &analysis_result.declaration_table,
            tt: &analysis_result.type_table,
            et: Self::order_to_table(&analysis_result.graph_analysis_result.evaluation_order),
            mt: &analysis_result.graph_analysis_result.space_requirements,
            tr: &analysis_result.graph_analysis_result.tracking_requirements,
            ir,
            ri: &analysis_result.graph_analysis_result.input_dependencies,
        }
    }

    pub(crate) fn lower(mut self) -> RTLolaIR {
        self.lower_ast();
        self.ir
    }

    fn lower_ast(&mut self) {
        self.ast.inputs.iter().for_each(|i| self.lower_input(i));
        self.ast.outputs.iter().for_each(|o| self.lower_output_declaration(o));
        self.ast.outputs.iter().for_each(|o| self.lower_output_expression(o));
        self.link_windows();
        self.ast.trigger.iter().for_each(|t| self.lower_trigger(t));
    }

    /// Links streams to windows depending on them.
    /// Example:
    /// input in: Int8; output out Int8@5Hz := in.aggregate(5s, Σ)
    /// This function sets the connection from `in` to the window in `out`.
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

    /// Creates input stream and adds it to the IR. Does *not* link input streams to windows.
    fn lower_input(&mut self, input: &ast::Input) {
        let nid = input.id;
        let ast_req = self.get_memory(nid);
        let memory_bound = self.lower_storage_req(ast_req);
        let reference = self.get_ref_for_stream(nid);
        let layer = self.get_layer(nid);

        let trackings = self.collect_tracking_info(nid, None);

        let input = ir::InputStream {
            name: input.name.name.clone(),
            ty: self.lower_node_type(nid),
            dependent_streams: trackings,
            dependent_windows: Vec::new(),
            layer,
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
        let expr = self.lower_stream_expression(&trigger.expression, &ty);
        let reference = StreamReference::OutRef(self.ir.outputs.len());
        let mut outgoing_dependencies = Vec::new();
        self.find_dependencies(&trigger.expression, &mut outgoing_dependencies);
        let input_dependencies = self.gather_dependent_inputs(trigger.id);
        let ac = match self.check_time_driven(trigger.id, reference) {
            None => Some(self.tt.get_acti_cond(trigger.id).clone()),
            Some(_tds) => None,
        };
        let output = ir::OutputStream {
            name,
            ty,
            expr,
            dependent_streams: Vec::new(),
            dependent_windows: Vec::new(),
            memory_bound: MemorizationBound::Bounded(0),
            layer: self.get_layer(trigger.id),
            reference,
            outgoing_dependencies,
            input_dependencies,
            ac,
        };
        self.ir.outputs.push(output);
        let trig = ir::Trigger {
            message: trigger.message.clone().unwrap_or_else(|| format!("{}", trigger.expression)),
            reference,
            trigger_idx: self.ir.triggers.len(),
        };
        match self.check_time_driven(trigger.id, reference) {
            None => self.ir.event_driven.push(EventDrivenStream { reference }),
            Some(tds) => self.ir.time_driven.push(tds),
        }
        self.ir.triggers.push(trig);
    }

    fn collect_tracking_info(&self, nid: NodeId, time_driven: Option<&TimeDrivenStream>) -> Vec<ir::Tracking> {
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

    /// Creates outputs and adds them to the LolaIR. Does *not* lower expression nor link depending windows, yet.
    fn lower_output_declaration(&mut self, ast_output: &ast::Output) {
        let nid = ast_output.id;
        let ast_req = self.get_memory(nid);
        let memory_bound = self.lower_storage_req(ast_req);
        let layer = self.get_layer(nid);
        let reference = self.get_ref_for_stream(nid);
        let time_driven = self.check_time_driven(ast_output.id, reference);

        let trackings = self.collect_tracking_info(nid, time_driven.as_ref());

        let ac = match time_driven.as_ref() {
            None => Some(self.tt.get_acti_cond(ast_output.id).clone()),
            Some(_tds) => None,
        };
        let output = ir::OutputStream {
            name: ast_output.name.name.clone(),
            ty: ir::Type::Bool,
            expr: ir::Expression::new(
                ir::ExpressionKind::LoadConstant(ir::Constant::Str(String::from("not yet initialized"))),
                ir::Type::Bool,
            ),
            outgoing_dependencies: Vec::new(),
            dependent_streams: trackings,
            dependent_windows: Vec::new(),
            memory_bound,
            layer,
            reference,
            input_dependencies: Vec::new(),
            ac,
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
    }

    /// Lowers output expression and adds them to the LolaIR. Does *not* link depending windows, yet.
    fn lower_output_expression(&mut self, ast_output: &ast::Output) {
        let nid = ast_output.id;
        let reference = self.get_ref_for_stream(nid);

        let input_dependencies = self.gather_dependent_inputs(nid);
        let mut outgoing_dependencies = Vec::new();
        self.find_dependencies(&ast_output.expression, &mut outgoing_dependencies);
        let mut dep_map: HashMap<StreamReference, Vec<ir::Offset>> = HashMap::new();
        outgoing_dependencies.into_iter().for_each(|dep| {
            dep_map.entry(dep.stream).or_insert_with(Vec::new).extend_from_slice(dep.offsets.as_slice())
        });
        let outgoing_dependencies =
            dep_map.into_iter().map(|(sr, offsets)| ir::Dependency { stream: sr, offsets }).collect();

        let output_type = self.lower_node_type(nid);
        let expr = self.lower_stream_expression(&ast_output.expression, &output_type);
        let output = self.ir.get_out_mut(reference);

        output.ty = output_type;
        output.input_dependencies = input_dependencies;
        output.outgoing_dependencies = outgoing_dependencies;
        output.expr = expr;
    }

    /// Returns the flattened result of calling `map` on each node recursively in `pre_order` or post_order.
    /// Applies filter to each node before mapping. Children of filtered nodes will not be taken into account.
    #[allow(dead_code)]
    fn collect_expression<T, M, F>(expr: &'a ast::Expression, map: &M, filter: &F, pre_order: bool) -> Vec<T>
    where
        M: Fn(&'a ast::Expression) -> Vec<T>,
        F: Fn(&'a ast::Expression) -> bool,
    {
        let recursion = |e| Lowering::collect_expression(e, map, filter, pre_order);
        let pre = if pre_order { map(expr).into_iter() } else { Vec::new().into_iter() };
        let post = || {
            if pre_order {
                Vec::new().into_iter()
            } else {
                map(expr).into_iter()
            }
        };
        if filter(expr) {
            match &expr.kind {
                ExpressionKind::Lit(_) => pre.chain(post()).collect(),
                ExpressionKind::Ident(_) => pre.chain(post()).collect(),
                ExpressionKind::StreamAccess(e, _) => pre.chain(recursion(e)).chain(post()).collect(),
                ExpressionKind::Default(e, dft) => {
                    pre.chain(recursion(e)).chain(recursion(dft)).chain(post()).collect()
                }
                ExpressionKind::Offset(e, _) => pre.chain(recursion(e)).chain(post()).collect(),
                ExpressionKind::SlidingWindowAggregation { expr, duration, .. } => {
                    pre.chain(recursion(expr)).chain(recursion(duration)).chain(post()).collect()
                }
                ExpressionKind::Binary(_, lhs, rhs) => {
                    pre.chain(recursion(lhs)).chain(recursion(rhs)).chain(post()).collect()
                }
                ExpressionKind::Unary(_, operand) => pre.chain(recursion(operand)).chain(post()).collect(),
                ExpressionKind::Ite(cond, cons, alt) => {
                    pre.chain(recursion(cond)).chain(recursion(cons)).chain(recursion(alt)).chain(post()).collect()
                }
                ExpressionKind::ParenthesizedExpression(_, e, _) => {
                    pre.chain(Lowering::collect_expression(e, map, filter, pre_order)).chain(post()).collect()
                }
                ExpressionKind::MissingExpression => unreachable!(),
                ExpressionKind::Tuple(exprs) => {
                    let elems = exprs.iter().flat_map(|a| recursion(a));
                    pre.chain(elems).chain(post()).collect()
                }
                ExpressionKind::Function(_, _, args) => {
                    let args = args.iter().flat_map(|a| recursion(a));
                    pre.chain(args).chain(post()).collect()
                }
                ExpressionKind::Field(e, _) => pre.chain(recursion(e)).chain(post()).collect(),
                ExpressionKind::Method(_, _, _, _) => unimplemented!("Methods not supported, yet."),
            }
        } else {
            Vec::new()
        }
    }

    /// Finds all streams the expression accesses, excluding windows.
    fn find_dependencies(&self, expr: &ast::Expression, deps: &mut Vec<ir::Dependency>) {
        use ExpressionKind::*;
        match &expr.kind {
            Offset(inner, offset) => match &inner.kind {
                Ident(_ident) => {
                    let sr = self.get_ref_for_ident(inner.id);
                    let offset = self.lower_offset(sr, offset);
                    deps.push(ir::Dependency { stream: sr, offsets: vec![offset] })
                }
                _ => {
                    unreachable!("checked in AST verification");
                }
            },
            Lit(_) => {}
            Ident(_) => match self.get_decl(expr.id) {
                Declaration::In(inp) => {
                    let sr = self.get_ref_for_stream(inp.id);
                    deps.push(ir::Dependency { stream: sr, offsets: vec![ir::Offset::PastDiscreteOffset(0)] })
                }
                Declaration::Out(out) => {
                    let sr = self.get_ref_for_stream(out.id);
                    deps.push(ir::Dependency { stream: sr, offsets: vec![ir::Offset::PastDiscreteOffset(0)] })
                }
                _ => {}
            },
            StreamAccess(e, _) | Unary(_, e) | ParenthesizedExpression(_, e, _) | Field(e, _) => {
                self.find_dependencies(e, deps)
            }
            Default(left, right) | Binary(_, left, right) => {
                self.find_dependencies(left, deps);
                self.find_dependencies(right, deps);
            }
            SlidingWindowAggregation { .. } => {
                // ignore sliding windows
            }
            Ite(cond, cons, alt) => {
                self.find_dependencies(cond, deps);
                self.find_dependencies(cons, deps);
                self.find_dependencies(alt, deps);
            }
            MissingExpression => unreachable!("checked in AST verification"),
            Tuple(exprs) | Function(_, _, exprs) => {
                exprs.iter().for_each(|e| self.find_dependencies(e, deps));
            }
            Method(inner, _, _, params) => {
                self.find_dependencies(inner, deps);
                params.iter().for_each(|e| self.find_dependencies(e, deps));
            }
        }
    }

    fn lower_tracking_req(
        &self,
        tracker: Option<&TimeDrivenStream>,
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

    /// Creates a SlidingWindow, adds it to the IR, and returns a reference to it.
    fn lower_window(&mut self, win_expr: &ast::Expression) -> WindowReference {
        if let ExpressionKind::SlidingWindowAggregation { expr, duration, wait, aggregation } = &win_expr.kind {
            if let ExpressionKind::Ident(_) = &expr.kind {
                let target = self.get_ref_for_ident(expr.id);
                let duration = self.lower_duration(duration.as_ref());
                let op = *aggregation;
                let reference = WindowReference(self.ir.sliding_windows.len());
                let ty = self.lower_node_type(win_expr.id);
                let window = ir::SlidingWindow { target, duration, wait: *wait, op, reference, ty };
                self.ir.sliding_windows.push(window);
                reference
            } else {
                unreachable!("Verified in TypeChecker")
            }
        } else {
            unreachable!("Must not pass non-window expression to `Lowering::lower_window`")
        }
    }

    fn lower_duration(&self, duration: &ast::Expression) -> Duration {
        let exact_duration = duration.parse_duration().expect("Duration literal needs to be a duration specification.");
        Duration::from_nanos(
            exact_duration.get::<nanosecond>().to_integer().to_u64().expect("Period [ns] too large for u64!"),
        )
    }

    fn lower_storage_req(&self, req: StorageRequirement) -> MemorizationBound {
        match req {
            StorageRequirement::Finite(b) => MemorizationBound::Bounded(b),
            StorageRequirement::FutureRef(b) => MemorizationBound::Bounded(b),
            StorageRequirement::Unbounded => MemorizationBound::Unbounded,
        }
    }

    fn lower_node_type(&self, id: NodeId) -> ir::Type {
        self.tt.get_value_type(id).into()
    }

    fn lower_stream_expression(&mut self, expression: &ast::Expression, expected_type: &ir::Type) -> ir::Expression {
        let (result, ty) = self.lower_expression(expression);
        // TODO: Consider linearizing the tree to better exploit locality than when using Boxes.
        if &ty != expected_type {
            ir::Expression::new(
                ir::ExpressionKind::Convert { from: ty, to: expected_type.clone(), expr: Box::new(result) },
                expected_type.clone(),
            )
        } else {
            result
        }
    }

    fn lower_expression(&mut self, expr: &ast::Expression) -> (ir::Expression, ir::Type) {
        let result_type = self.lower_node_type(expr.id);

        let expr = match &expr.kind {
            ExpressionKind::Lit(l) => ir::Expression::new(
                ir::ExpressionKind::LoadConstant(self.lower_literal(l, expr.id)),
                result_type.clone(),
            ),
            ExpressionKind::Ident(_) => {
                let (src_ty, expr) = match self.get_decl(expr.id) {
                    Declaration::In(input) => (
                        self.lower_node_type(input.id),
                        ir::Expression::new(
                            ir::ExpressionKind::StreamAccess(self.get_ref_for_stream(input.id), StreamAccessKind::Sync),
                            self.lower_node_type(input.id),
                        ),
                    ),
                    Declaration::Out(output) => (
                        self.lower_node_type(output.id),
                        ir::Expression::new(
                            ir::ExpressionKind::StreamAccess(
                                self.get_ref_for_stream(output.id),
                                StreamAccessKind::Sync,
                            ),
                            self.lower_node_type(output.id),
                        ),
                    ),
                    Declaration::Const(constant) => {
                        let node_type = self.lower_node_type(constant.id);
                        (
                            node_type.clone(),
                            ir::Expression::new(
                                ir::ExpressionKind::LoadConstant(self.lower_literal(&constant.literal, constant.id)),
                                node_type,
                            ),
                        )
                    }
                    _ => unreachable!(),
                };
                if src_ty != result_type {
                    ir::Expression::new(
                        ir::ExpressionKind::Convert { from: src_ty, to: result_type.clone(), expr: expr.into() },
                        result_type.clone(),
                    )
                } else {
                    expr
                }
            }
            ExpressionKind::StreamAccess(expr, kind) => {
                let target_id = match &expr.kind {
                    ExpressionKind::Ident(_) => expr.id,
                    _ => unreachable!("checked by AST verifier"),
                };
                let target = self.get_ref_for_ident(target_id);
                ir::Expression::new(ir::ExpressionKind::StreamAccess(target, *kind), self.lower_node_type(expr.id))
            }
            ExpressionKind::Default(e, dft) => ir::Expression::new(
                ir::ExpressionKind::Default {
                    expr: Box::new(self.lower_expression(e).0),
                    default: Box::new(self.lower_expression(dft).0),
                },
                result_type.clone(),
            ),
            ExpressionKind::Offset(stream, offset) => {
                let target = self.get_ref_for_ident(stream.id);
                let offset = self.lower_offset(target, offset);
                ir::Expression::new(ir::ExpressionKind::OffsetLookup { target, offset }, result_type.clone())
            }
            ExpressionKind::SlidingWindowAggregation { .. } => {
                let win_ref = self.lower_window(expr);
                ir::Expression::new(ir::ExpressionKind::WindowLookup(win_ref), result_type.clone())
            }
            ExpressionKind::Binary(ast_op, lhs, rhs) => {
                let ir_op = Lowering::lower_bin_op(*ast_op);

                self.lower_arith_log(expr.id, ir_op, &[lhs, rhs], result_type.clone(), |resolved_poly_types| {
                    use crate::ast::BinOp::*;
                    match ast_op {
                        Add | Sub | Mul | Div | Rem | Pow | Eq | Lt | Le | Ne | Ge | Gt | BitAnd | BitOr | BitXor => {
                            assert_eq!(resolved_poly_types.len(), 1);
                            let arg_ty = resolved_poly_types[0].clone();
                            vec![arg_ty.clone(), arg_ty]
                        }
                        And | Or => vec![ir::Type::Bool, ir::Type::Bool],
                        Shl | Shr => {
                            assert_eq!(resolved_poly_types.len(), 2);
                            let lhs_ty = resolved_poly_types[0].clone();
                            let rhs_ty = resolved_poly_types[1].clone();
                            vec![lhs_ty, rhs_ty]
                        }
                    }
                })
            }
            ExpressionKind::Unary(ast_op, operand) => {
                let ir_op = Lowering::lower_un_op(*ast_op);

                self.lower_arith_log(expr.id, ir_op, &[operand], result_type.clone(), |resolved_poly_types| {
                    vec![match ast_op {
                        ast::UnOp::Neg | ast::UnOp::BitNot => {
                            assert_eq!(resolved_poly_types.len(), 1);
                            resolved_poly_types[0].clone()
                        }
                        ast::UnOp::Not => ir::Type::Bool,
                    }]
                })
            }
            ExpressionKind::Ite(cond, cons, alt) => {
                let (cond_expr, _) = self.lower_expression(cond);
                let mut args = self.handle_func_args(&[result_type.clone(), result_type.clone()], &[cons, alt]);
                // We remove the elements to avoid having to clone them when moving into the expression.
                ir::Expression::new(
                    ir::ExpressionKind::Ite {
                        condition: Box::new(cond_expr),
                        consequence: Box::new(args.remove(0)),
                        alternative: Box::new(args.remove(0)),
                    },
                    result_type.clone(),
                )
            }
            ExpressionKind::ParenthesizedExpression(_, e, _) => self.lower_expression(e).0,
            ExpressionKind::MissingExpression => unreachable!(),
            ExpressionKind::Tuple(exprs) => {
                let exprs = exprs.iter().map(|e| self.lower_expression(e).0).collect();
                ir::Expression::new(ir::ExpressionKind::Tuple(exprs), result_type.clone())
            }
            ExpressionKind::Function(name, _, args) => {
                let args: Vec<&ast::Expression> = args.iter().map(Box::as_ref).collect();

                let generics = self.tt.get_func_arg_types(expr.id);
                let (arg_types, ret_type) = if let Declaration::Func(fd) = self.get_decl(expr.id) {
                    fd.get_types_for_args_and_ret(generics)
                } else {
                    unreachable!("Function not declared as such.")
                };
                let arg_types: Vec<ir::Type> = arg_types.into_iter().map(|ty| (&ty).into()).collect();
                let args = self.handle_func_args(&arg_types, &args[..]);

                let (func_expr, ret_type) = if name.name.name == "cast" {
                    // cast is no actual function
                    assert!(!args.is_empty());
                    assert!(!arg_types.is_empty());
                    (args[0].clone(), arg_types[0].clone())
                } else {
                    let ret_type: ir::Type = (&ret_type).into();
                    let fun_ty = ir::Type::Function(arg_types, Box::new(ret_type.clone()));
                    (
                        ir::Expression::new(
                            ir::ExpressionKind::Function(name.name.name.clone(), args, fun_ty),
                            ret_type.clone(),
                        ),
                        ret_type,
                    )
                };
                if ret_type != result_type {
                    ir::Expression::new(
                        ir::ExpressionKind::Convert { from: ret_type, to: result_type.clone(), expr: func_expr.into() },
                        result_type.clone(),
                    )
                } else {
                    func_expr
                }
            }
            ExpressionKind::Method(inner, name, _, args) => {
                let args: Vec<&ast::Expression> = std::iter::once(inner).chain(args).map(Box::as_ref).collect();

                let generics = self.tt.get_func_arg_types(expr.id);
                let (arg_types, ret_type) = if let Declaration::Func(fd) = self.get_decl(expr.id) {
                    fd.get_types_for_args_and_ret(generics)
                } else {
                    unreachable!("Function not declared as such.")
                };
                let arg_types: Vec<ir::Type> = arg_types.into_iter().map(|ty| (&ty).into()).collect();
                let ret_type: ir::Type = (&ret_type).into();

                let args = self.handle_func_args(&arg_types, &args[..]);
                let fun_ty = ir::Type::Function(arg_types, Box::new(ret_type.clone()));

                let func_expr = ir::Expression::new(
                    ir::ExpressionKind::Function(name.name.name.clone(), args, fun_ty),
                    ret_type.clone(),
                );
                if ret_type != result_type {
                    ir::Expression::new(
                        ir::ExpressionKind::Convert { from: ret_type, to: result_type.clone(), expr: func_expr.into() },
                        result_type.clone(),
                    )
                } else {
                    func_expr
                }
            }
            ExpressionKind::Field(expr, ident) => {
                let num: usize = ident.name.parse::<usize>().expect("checked in AST verifier");
                ir::Expression::new(
                    ir::ExpressionKind::TupleAccess(self.lower_expression(expr).0.into(), num),
                    result_type.clone(),
                )
            }
        };
        (expr, result_type)
    }

    /// Handles arithmetic-logic operations.
    /// `nid` is the node id of the expression.
    /// `op` is the operation.
    /// `args` contains all expressions yielding the arguments.
    /// `result_type` is the return type of the operation.
    /// `f` transforms a list of resolved polymorphic types into a full list of argument types.
    fn lower_arith_log<F>(
        &mut self,
        nid: NodeId,
        op: ir::ArithLogOp,
        args: &[&ast::Expression],
        result_type: ir::Type,
        f: F,
    ) -> ir::Expression
    where
        F: FnOnce(Vec<ir::Type>) -> Vec<ir::Type>,
    {
        // resolved_poly_types is the vector of resolved polymorphic components.
        // e.g. for `+<T: Numeric>(T, T) -> T`, it can be `vec![Int32]`.
        let resolved_poly_types = self.tt.get_func_arg_types(nid).iter().map(|t| t.into()).collect();
        let arg_types = f(resolved_poly_types);
        let args = self.handle_func_args(&arg_types, args);
        let fun_ty = ir::Type::Function(arg_types, Box::new(result_type.clone()));
        ir::Expression::new(ir::ExpressionKind::ArithLog(op, args, fun_ty), result_type)
    }

    fn handle_func_args(&mut self, types: &[ir::Type], args: &[&ast::Expression]) -> Vec<ir::Expression> {
        assert_eq!(types.len(), args.len());
        types
            .iter()
            .zip(args.iter())
            .map(|(req_ty, a)| {
                let (arg, actual_ty) = self.lower_expression(a);
                if req_ty != &actual_ty {
                    ir::Expression::new(
                        ir::ExpressionKind::Convert { from: actual_ty, to: req_ty.clone(), expr: Box::new(arg) },
                        req_ty.clone(),
                    )
                } else {
                    arg
                }
            })
            .collect()
    }

    fn lower_offset(&self, target: StreamReference, offset: &ast::Offset) -> ir::Offset {
        match offset {
            &ast::Offset::Discrete(val) if val < 0 => {
                assert!(val < 0); // Should be checked by type checker, though.
                ir::Offset::PastDiscreteOffset(
                    val.abs().try_into().expect("conversion from i16.abs() => u32 cannot fail"),
                )
            }
            &ast::Offset::Discrete(val) => {
                // val >= 0
                ir::Offset::FutureDiscreteOffset(
                    val.try_into().expect("conversion from i16 to u32 guarded by `if !(val < 0)`"),
                )
            }
            ast::Offset::RealTime(_, _) => {
                let uom_offset = offset.to_uom_time().expect("ast::Offset::RealTime should return uom_time");
                let period = self
                    .ir
                    .time_driven
                    .iter()
                    .find(|td| td.reference == target)
                    .expect("target should exist in ir.time_driven")
                    .period;
                let offset = uom_offset.get::<second>() / period.get::<second>();
                debug_assert!(offset.is_integer(), "offset={:#?}, period={:#?}", uom_offset, period); // should be checked already
                debug_assert!(offset.is_negative());
                let offset = offset.abs().to_integer().to_u32().expect("offset to big for u32");
                ir::Offset::PastDiscreteOffset(offset)
            }
        }
    }

    fn lower_literal(&self, lit: &ast::Literal, nid: NodeId) -> ir::Constant {
        use crate::ast::LitKind;
        let expected_type = self.lower_node_type(nid);
        match &lit.kind {
            LitKind::Str(s) | LitKind::RawStr(s) => ir::Constant::Str(s.clone()),
            LitKind::Numeric(_, unit) => {
                assert!(unit.is_none());
                match expected_type {
                    ir::Type::Float(_) => {
                        ir::Constant::Float(lit.parse_numeric::<f64>().expect("checked by type checker"))
                    }
                    ir::Type::UInt(_) => {
                        ir::Constant::UInt(lit.parse_numeric::<u64>().expect("checked by type checker"))
                    }
                    ir::Type::Int(_) => ir::Constant::Int(lit.parse_numeric::<i64>().expect("checked by type checker")),
                    _ => unreachable!("checked by type checker {}", expected_type),
                }
            }
            LitKind::Bool(b) => ir::Constant::Bool(*b),
        }
    }

    fn lower_un_op(ast_op: ast::UnOp) -> ir::ArithLogOp {
        match ast_op {
            ast::UnOp::Neg => ir::ArithLogOp::Neg,
            ast::UnOp::Not => ir::ArithLogOp::Not,
            ast::UnOp::BitNot => ir::ArithLogOp::BitNot,
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
            BitAnd => ir::ArithLogOp::BitAnd,
            BitOr => ir::ArithLogOp::BitOr,
            BitXor => ir::ArithLogOp::BitXor,
            Shl => ir::ArithLogOp::Shl,
            Shr => ir::ArithLogOp::Shr,
        }
    }

    fn check_time_driven(&mut self, stream_id: NodeId, reference: StreamReference) -> Option<TimeDrivenStream> {
        match &self.tt.get_stream_type(stream_id) {
            StreamTy::RealTime(f) => {
                let period = UOM_Time::new::<second>(f.freq.get::<hertz>().inv());
                Some(TimeDrivenStream {
                    reference,
                    frequency: f.freq,
                    extend_rate: Duration::from_nanos(
                        period.get::<nanosecond>().to_integer().to_u64().expect("Period [ns] too large for u64!"),
                    ),
                    period,
                })
            }
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

    fn create_ref_lookup(inputs: &[Rc<ast::Input>], outputs: &[Rc<ast::Output>]) -> HashMap<NodeId, StreamReference> {
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
        o2t(&eo.periodic_streams_order).chain(o2t(&eo.event_based_streams_order)).collect()
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
            Declaration::Type(_) | Declaration::Func(_) => unreachable!("Types and functions are not streams."),
            Declaration::ParamOut(_) => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::*;
    use crate::FrontendConfig;

    fn spec_to_ir(spec: &str) -> RTLolaIR {
        crate::parse("stdin", spec, FrontendConfig::default()).expect("spec was invalid")
    }

    fn check_stream_number(
        ir: &RTLolaIR,
        inputs: usize,
        outputs: usize,
        time: usize,
        event: usize,
        sliding: usize,
        triggers: usize,
    ) {
        assert_eq!(inputs, ir.inputs.len());
        assert_eq!(outputs, ir.outputs.len());
        assert_eq!(time, ir.time_driven.len());
        assert_eq!(event, ir.event_driven.len());
        assert_eq!(sliding, ir.sliding_windows.len());
        assert_eq!(triggers, ir.triggers.len());
    }

    #[test]
    fn lower_one_input() {
        let ir = spec_to_ir("input a: Int32");
        check_stream_number(&ir, 1, 0, 0, 0, 0, 0);
    }

    #[test]
    fn lower_triggers() {
        let ir = spec_to_ir("input a: Int32\ntrigger a > 50\ntrigger a < 30 \"So low...\"");
        // Note: Each trigger needs to be accounted for as an output stream.
        check_stream_number(&ir, 1, 2, 0, 2, 0, 2);
    }

    #[test]
    fn lower_one_output_event() {
        let ir = spec_to_ir("output a: Int32 := 34");
        check_stream_number(&ir, 0, 1, 0, 1, 0, 0);
    }

    #[test]
    fn lower_one_output_event_float() {
        let ir = spec_to_ir("output a: Float64 := 34.");
        check_stream_number(&ir, 0, 1, 0, 1, 0, 0);
    }

    #[test]
    fn lower_one_output_event_float16() {
        let ir = spec_to_ir("output a: Float16 := 34.");
        check_stream_number(&ir, 0, 1, 0, 1, 0, 0);
    }

    #[test]
    fn lower_one_output_time() {
        let ir = spec_to_ir("output a: Int32 @1Hz := 34");
        check_stream_number(&ir, 0, 1, 1, 0, 0, 0);
    }

    #[test]
    fn lower_one_sliding() {
        let ir = spec_to_ir("input a: Int32 output b: Int64 @1Hz := a.aggregate(over: 3s, using: sum)");
        check_stream_number(&ir, 1, 1, 1, 0, 1, 0);
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
        check_stream_number(&ir, 2, 6, 5, 1, 2, 1);
    }

    #[test]
    fn lower_constant_expression() {
        let ir = spec_to_ir("output a: Int32 := 3+4*7");
        let stream: &OutputStream = &ir.outputs[0];

        let ty = Type::Int(crate::ty::IntTy::I32);

        assert_eq!(stream.ty, ty);

        let tar = &ir.outputs[0];
        assert_eq!("+(3,*(4,7) : [(Int32,Int32) -> Int32]) : [(Int32,Int32) -> Int32]", format!("{}", tar.expr))
    }

    #[test]
    fn lower_expr_with_widening() {
        let ir = spec_to_ir("input a: UInt8 output b: UInt16 := a");
        let stream = &ir.outputs[0];

        let expr = &stream.expr;
        assert_eq!("cast<UInt8,UInt16>(In(0))", format!("{}", expr))
    }

    #[test]
    fn lower_function_expression() {
        let ir = spec_to_ir("import math input a: Float32 output v: Float64 := sqrt(a)");
        let stream: &OutputStream = &ir.outputs[0];

        let ty = Type::Float(crate::ty::FloatTy::F64);

        assert_eq!(stream.ty, ty);

        let expr = &stream.expr;
        assert_eq!("cast<Float32,Float64>(sqrt(In(0): Float32) -> Float32)", format!("{}", expr))
    }

    #[test]
    fn lower_cast_expression() {
        let ir = spec_to_ir("input a: Float64 output v: Float32 := cast(a)");
        let stream: &OutputStream = &ir.outputs[0];

        let ty = Type::Float(crate::ty::FloatTy::F32);

        assert_eq!(stream.ty, ty);

        let expr = &stream.expr;
        assert_eq!("cast<Float64,Float32>(In(0))", format!("{}", expr))
    }

    #[ignore] // Needs to be adapted to new lowering.
    #[test]
    fn lower_function_expression_regex() {
        //        let ir = spec_to_ir("import regex\ninput a: String output v: Bool := matches_regex(a, r\"a*b\")");
        //
        //        let stream: &OutputStream = &ir.outputs[0];
        //
        //        let ty = Type::Bool;
        //
        //        assert_eq!(stream.ty, ty);
        //
        //        let expr = &stream.expr;
        //        assert_eq!(expr.stmts.len(), 3);
        //
        //        let load = &expr.stmts[0];
        //
        //        match &load.op {
        //            Op::SyncStreamLookup(StreamInstance { reference, arguments }) => {
        //                assert!(arguments.is_empty(), "Lookup does not have arguments.");
        //                match reference {
        //                    StreamReference::InRef(0) => {}
        //                    _ => unreachable!("Incorrect StreamReference"),
        //                }
        //            }
        //            _ => unreachable!("Need to load the constant first."),
        //        };
        //
        //        let constant = &expr.stmts[1];
        //        match &constant.op {
        //            Op::LoadConstant(Constant::Str(s)) => assert_eq!(s, "a*b"),
        //            c => unreachable!("expected constant, found {:?}", c),
        //        }
        //
        //        let regex_match = &expr.stmts[2];
        //
        //        match &regex_match.op {
        //            Op::Function(s) => assert_eq!(s, "matches_regex"),
        //            _ => unreachable!("Need to apply the function!"),
        //        }
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
        let ir = spec_to_ir("input a: Int32 output b: Int32 @1Hz := a.aggregate(over: 3s, using: sum)");
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

    #[test]
    fn dependency_test() {
        let ir = spec_to_ir(
            "input a: Int32\ninput b: Int32\ninput c: Int32\noutput d: Int32 := a + b + b[-1].defaults(to: 0) + a[-2].defaults(to: 0) + c",
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
        let a_dep = out_dep.iter().find(|&x| x.stream == in_refs[0]).expect("a dependencies not found");
        let b_dep = out_dep.iter().find(|&x| x.stream == in_refs[1]).expect("b dependencies not found");
        let c_dep = out_dep.iter().find(|&x| x.stream == in_refs[2]).expect("c dependencies not found");
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
