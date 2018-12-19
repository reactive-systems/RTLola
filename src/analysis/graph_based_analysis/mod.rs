pub mod dependency_graph;
pub mod evaluation_order;
pub mod future_dependency;
pub mod space_requirements;

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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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

/// For every stream we need to store
/// - the last k+1 values if there is an access with discrete offset -k
/// - everything if its auxiliary streams depend on the future
///
/// We also have do differentiate between
/// - streams that do not (transitively) depend on the future and can therefore directly be computed
/// - streams that (transitively) depend on the future and therefore positions may not already be evaluated while in the buffer
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum StorageRequirement {
    Finite(u16),
    FutureRef(u16),
    Unbounded,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum TrackingRequirement {
    Finite(u16),
    Unbounded,
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
