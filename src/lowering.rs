use crate::analysis::naming::DeclarationTable;
use crate::analysis::typing::TypeAnalysis;
// Only import the unambiguous Nodes, use `ast::`/`ir::` prefix for disambiguation.
use crate::ast;
use crate::ast::{ExpressionKind, LolaSpec};
use crate::ir;
use crate::ir::{LolaIR, MemorizationBound, ParametrizedStream, StreamReference, TimeDrivenStream};
use crate::ty::TimingInfo;
use ast_node::{AstNode, NodeId};
use std::collections::HashMap;
type MemoryTable = HashMap<NodeId, StorageRequirement>;
type EvalOrder = Vec<Vec<ComputeStep>>;

struct EvaluationOrder {
    aligned: EvalOrder,
    event_driven: EvalOrder,
    time_driven: EvalOrder,
}

// TODO: Replace StorageRequirement with crate::analysis::dependency::StorageRequirement or similar.
// TODO: Same for ComputeStep, MemoryTable, Eval[uation]Order.
#[derive(Debug, Copy, Clone)]
pub(crate) enum StorageRequirement {
    Finite(usize),
    FutureRef(usize),
    Shifted,
}
#[derive(Debug, Copy, Clone)]
pub(crate) enum ComputeStep {
    Invoke(ast_node::NodeId),
    Extend(ast_node::NodeId),
    Evaluate(ast_node::NodeId),
    Terminate(ast_node::NodeId),
}

type EvalTable = HashMap<NodeId, u32>;

struct Lowering<'a> {
    dt: DeclarationTable<'a>,
    tt: TypeAnalysis<'a>,
    et: EvalTable,
    mt: MemoryTable,
    ir: LolaIR,
}

impl<'a> Lowering<'a> {
    pub(crate) fn new(
        dt: DeclarationTable<'a>,
        tt: TypeAnalysis<'a>,
        eval_order: EvaluationOrder,
        mt: MemoryTable,
    ) -> Lowering<'a> {
        let ir = LolaIR {
            inputs: Vec::new(),
            outputs: Vec::new(),
            time_driven: Vec::new(),
            event_driven: Vec::new(),
            parametrized: Vec::new(),
            sliding_windows: Vec::new(),
            triggers: Vec::new(),
            feature_flags: Vec::new(),
        };
        Lowering {
            dt,
            tt,
            et: Self::order_to_table(eval_order),
            mt,
            ir,
        }
    }

    pub(crate) fn lower(mut self, ast: LolaSpec) -> LolaIR {
        self.lower_ast(ast);
        self.ir
    }

    fn lower_ast(&mut self, ast: LolaSpec) -> LolaIR {
        ast.inputs.into_iter().for_each(|i| self.lower_input(i));
        ast.outputs.into_iter().for_each(|o| self.lower_output(o));
        unimplemented!()
    }

    fn lower_input(&mut self, input: ast::Input) {
        unimplemented!()
    }

    fn lower_output(&mut self, ast_output: ast::Output) {
        let nid = ast_output.id();
        let mem_bound = self.lower_storage_req(*self.mt.get(nid).unwrap()); // TODO: Handle bug in MT.
        let layer = *self.et.get(nid).unwrap(); // TODO: Handle bug in MT.
                                                // Crucial: The very next thing we add to the IR output vector *must* be the output stream!
        let reference = StreamReference::OutRef(self.ir.outputs.len());

        let output = ir::OutputStream::new(
            ast_output.name.name.clone(),
            self.lower_type(ast_output._id), // TODO: Exclude these cases in preceding analysis steps.
            self.lower_expr(&ast_output.expression),
            mem_bound,
            layer,
            reference,
        );
        let debug_clone = output.clone();
        self.ir.outputs.push(output);

        assert_eq!(self.ir.get_out(reference), &debug_clone, "Bug in implementation: Output vector in IR changed between creation of reference and insertion of stream.");

        if let Some(td_ref) = self.check_time_driven(&ast_output, reference) {
            self.ir.time_driven.push(td_ref)
        }

        if let Some(param_ref) = self.check_parametrized(&ast_output, reference) {
            self.ir.parametrized.push(param_ref);
        }

        let windows: Vec<ir::SlidingWindow> = self
            .find_windows(&ast_output.expression)
            .iter()
            .map(|e| self.lower_window(&e, reference))
            .collect();
        self.ir.sliding_windows.extend(windows);
    }

    fn find_windows(&self, expr: &'a ast::Expression) -> Vec<&'a ast::Expression> {
        match &expr.kind {
            ExpressionKind::Lit(_) => Vec::new(),
            ExpressionKind::Ident(_) => Vec::new(),
            ExpressionKind::Default(expr, dft) => self
                .find_windows(expr)
                .into_iter()
                .chain(self.find_windows(dft))
                .collect(),
            ExpressionKind::Lookup(inst, _, op) => inst
                .arguments
                .iter()
                .flat_map(|a| self.find_windows(a))
                .chain(op.map(|_| expr).into_iter())
                .collect(),
            ExpressionKind::Binary(_, lhs, rhs) => self
                .find_windows(lhs)
                .into_iter()
                .chain(self.find_windows(rhs).into_iter())
                .collect(),
            ExpressionKind::Unary(_, operand) => self.find_windows(operand),
            ExpressionKind::Ite(cond, cons, alt) => self
                .find_windows(cond)
                .into_iter()
                .chain(self.find_windows(cons).into_iter())
                .chain(self.find_windows(alt).into_iter())
                .collect(),
            ExpressionKind::ParenthesizedExpression(_, expr, _) => self.find_windows(expr),
            ExpressionKind::MissingExpression() => panic!(), // TODO: Eradicate in preceding step.
            ExpressionKind::Tuple(exprs) => {
                exprs.iter().flat_map(|a| self.find_windows(a)).collect()
            }
            ExpressionKind::Function(_, _, args) => {
                args.iter().flat_map(|a| self.find_windows(a)).collect()
            }
            ExpressionKind::Field(expr, _) => self.find_windows(expr),
            ExpressionKind::Method(_, _, _, _) => unimplemented!(),
        }
    }

    fn lower_window(
        &self,
        expr: &ast::Expression,
        reference: StreamReference,
    ) -> ir::SlidingWindow {
        if let ExpressionKind::Lookup(inst, offset, op) = &expr.kind {
            let duration = match offset {
                ast::Offset::DiscreteOffset(_) => panic!(), // TODO: Eradicate in preceding step.
                ast::Offset::RealTimeOffset(expr, unit) => unimplemented!(),
            };
            ir::SlidingWindow {
                target: reference,
                duration,
                op: self.lower_window_op(op.unwrap()),
            }
        } else {
            unreachable!()
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
            Finite(b) => MemorizationBound::Bounded(b as u16),
            FutureRef(_) | Shifted => MemorizationBound::Unbounded,
        }
    }

    fn lower_type(&mut self, ty: NodeId) -> ir::Type {
        self.tt.get_type(ty).into()
    }

    fn lower_expr(&mut self, expr: &ast::Expression) -> ir::Expression {
        unimplemented!()
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

    fn order_to_table(eo: EvaluationOrder) -> EvalTable {
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
        o2t(eo.aligned)
            .chain(o2t(eo.time_driven))
            .chain(o2t(eo.event_driven))
            .collect()
    }
}
