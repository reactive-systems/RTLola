pub mod dependency_graph;
pub mod evaluation_order;
pub mod future_dependency;

use crate::ast::TimeUnit;
use ast_node::NodeId;
use ast_node::Span;
use petgraph::Directed;
use petgraph::Graph;

#[derive(Debug, Copy, Clone)]
pub(crate) enum StreamNode {
    ClassicInput(ast_node::NodeId),
    ClassicOutput(ast_node::NodeId),
    ParameterizedInput(ast_node::NodeId),
    ParameterizedOutput(ast_node::NodeId),
    RTOutput(ast_node::NodeId),
    Trigger(ast_node::NodeId),
    RTTrigger(ast_node::NodeId),
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum ComputeStep {
    Invoke(ast_node::NodeId),
    Extend(ast_node::NodeId),
    Evaluate(ast_node::NodeId),
    Terminate(ast_node::NodeId),
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum Offset {
    Discrete(i32),
    Time(f64, TimeUnit),
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum Location {
    Invoke,
    Extend,
    Terminate,
    Expression,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum StreamDependency {
    Access(Location, Offset, Span),
    InvokeByName(Span),
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum StorageRequirement {
    Finite(usize),
    FutureRef(usize),
}

pub(crate) type DependencyGraph = Graph<StreamNode, StreamDependency, Directed>;
pub(crate) type ComputationGraph = Graph<ComputeStep, (), Directed>;
//pub(crate) type NId = <DependencyGraph as petgraph::visit::GraphBase>::NodeId;
pub(crate) type NIx = petgraph::prelude::NodeIndex;
type EIx = petgraph::prelude::EdgeIndex;
//type EId = petgraph::prelude::EdgeIndex;

fn get_ast_id(dependent_node: StreamNode) -> NodeId {
    match dependent_node {
        StreamNode::ClassicOutput(id)
        | StreamNode::ParameterizedOutput(id)
        | StreamNode::RTOutput(id)
        | StreamNode::ClassicInput(id)
        | StreamNode::ParameterizedInput(id)
        | StreamNode::Trigger(id)
        | StreamNode::RTTrigger(id) => id,
    }
}
