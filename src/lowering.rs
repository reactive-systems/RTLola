use crate::analysis::naming::DeclarationTable;
use crate::ast::*;
use crate::ir::*;
use ast_node::NodeId;
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

struct Lowering<'a> {
    dt: DeclarationTable<'a>,
    tt: TypeTable,
    eval_order: EvaluationOrder,
    mt: MemoryTable,
    ir: LolaIR,
}

impl<'a> Lowering<'a> {
    pub(crate) fn new(
        dt: DeclarationTable<'a>,
        tt: TypeTable,
        eval_order: EvaluationOrder,
        mt: MemoryTable,
    ) -> Lowering {
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
            eval_order,
            mt,
            ir,
        }
    }

    pub(crate) fn lower(mut self, ast: &LolaSpec) -> LolaIR {
        self.lower_ast(ast)
    }

    fn lower_ast(&mut self, ast: &LolaSpec) -> LolaIR {
        unimplemented!()
    }
}
