use crate::analysis::naming::DeclarationTable;
use crate::ast::*;
use crate::ir::*;
use ast_node::NodeId;
use std::collections::HashMap;
type MemoryTable = HashMap<NodeId, StorageRequirement>;
type EvalOrder = Vec<Vec<ComputeStep>>;

struct EvaluationOrder {
    aligned: EvalOrder,
    event_based: EvalOrder,
    time_based: EvalOrder,
}

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
}

impl<'a> Lowering<'a> {
    pub(crate) fn new(
        dt: DeclarationTable<'a>,
        tt: TypeTable,
        eval_order: EvaluationOrder,
        mt: MemoryTable,
    ) -> Lowering {
        Lowering {
            dt,
            tt,
            eval_order,
            mt,
        }
    }

    pub(crate) fn lower(self, ast: &LolaSpec) -> LolaIR {
        self.lower_ast(ast)
    }

    fn lower_ast(&self, ast: &LolaSpec) -> LolaIR {
        unimplemented!()
    }
}
