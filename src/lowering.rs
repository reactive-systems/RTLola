use crate::analysis::naming::DeclarationTable;
use crate::analysis::typing::TypeAnalysis;
// Only import the unambiguous Nodes, use `ast::`/`ir::` prefix for disambiguation.
use crate::ast;
use crate::ast::LolaSpec;
use crate::ir;
use crate::ir::{LolaIR, MemorizationBound, StreamReference, TimeDrivenStream};
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

        // TODO: Check params and windows.
        unimplemented!()
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

    fn resolve_constant_expression(expr: &ast::Expression) -> ast::LitKind {
        unimplemented!()
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
