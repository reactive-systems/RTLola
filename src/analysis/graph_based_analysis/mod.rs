pub mod dependency_graph;
pub mod evaluation_order;
pub mod future_dependency;
mod memory_analysis;
pub mod space_requirements;

use super::lola_version::LolaVersionTable;
use super::typing::TypeTable;
use super::DeclarationTable;
use crate::ast::LolaSpec;
use crate::parse::{NodeId, Span};
use crate::reporting::Handler;
use petgraph::Directed;
use petgraph::Graph;
use std::time::Duration;

pub(crate) use self::evaluation_order::EvaluationOrderResult;
pub(crate) use self::future_dependency::FutureDependentStreams;
pub(crate) use self::space_requirements::SpaceRequirements;
use self::space_requirements::TrackingRequirements;
use crate::ty::{FloatTy, IntTy, UIntTy, ValueTy};
use num::BigRational;

#[derive(Debug, Copy, Clone)]
pub(crate) enum MemoryBound {
    Bounded(u128),
    Unbounded,
    Unknown,
}

pub(crate) struct GraphAnalysisResult {
    pub(crate) evaluation_order: EvaluationOrderResult,
    pub(crate) future_dependent_streams: FutureDependentStreams,
    pub(crate) space_requirements: SpaceRequirements,
    pub(crate) tracking_requirements: TrackingRequirements,
}

pub(crate) fn analyze<'a>(
    spec: &'a LolaSpec,
    version_analysis: &LolaVersionTable,
    declaration_table: &DeclarationTable<'a>,
    type_table: &TypeTable,
    handler: &Handler,
) -> Option<GraphAnalysisResult> {
    let dependency_analysis =
        dependency_graph::analyse_dependencies(spec, version_analysis, declaration_table, &handler);

    if handler.contains_error() {
        handler.error("aborting due to previous error");
        return None;
    }

    let (evaluation_order_result, pruned_graph) =
        evaluation_order::determine_evaluation_order(dependency_analysis.dependency_graph);

    let future_dependent_streams = future_dependency::future_dependent_stream(&pruned_graph);

    let space_requirements =
        space_requirements::determine_buffer_size(&pruned_graph, &future_dependent_streams);

    let tracking_requirements = space_requirements::determine_tracking_size(
        &pruned_graph,
        type_table,
        &future_dependent_streams,
    );

    let memory_requirement = memory_analysis::determine_worst_case_memory_consumption(
        spec,
        &space_requirements,
        &tracking_requirements,
        type_table,
        declaration_table,
    );
    match memory_requirement {
        MemoryBound::Unbounded => {
            println!("The specification has no bound on the memory consumption.")
        }
        MemoryBound::Bounded(bytes) => println!("The specification uses at most {} bytes.", bytes),
        MemoryBound::Unknown => {
            println!("Incomplete specification: we cannot determine the memory consumption.")
        }
    };

    Some(GraphAnalysisResult {
        evaluation_order: evaluation_order_result,
        future_dependent_streams,
        space_requirements,
        tracking_requirements,
    })
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum StreamNode {
    ClassicInput(NodeId),
    ClassicOutput(NodeId),
    ParameterizedInput(NodeId),
    ParameterizedOutput(NodeId),
    RTOutput(NodeId),
    Trigger(NodeId),
    RTTrigger(NodeId),
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum ComputeStep {
    Invoke(NodeId),
    Extend(NodeId),
    Evaluate(NodeId),
    Terminate(NodeId),
}

#[derive(Debug, Clone)]
pub(crate) enum TimeOffset {
    UpToNow(Duration, BigRational),
    Future(Duration, BigRational),
}

#[derive(Debug, Clone)]
pub(crate) enum Offset {
    Discrete(i32),
    Time(TimeOffset),
    SlidingWindow,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum Location {
    Invoke,
    Extend,
    Terminate,
    Expression,
}

#[derive(Debug, Clone)]
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
    // TODO What about accessing a future dependent stream?
    Future,
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

fn get_byte_size(value_ty: &ValueTy) -> MemoryBound {
    match value_ty {
        ValueTy::Bool => MemoryBound::Bounded(1),
        ValueTy::Int(int_ty) => MemoryBound::Bounded(match int_ty {
            IntTy::I8 => 1,
            IntTy::I16 => 2,
            IntTy::I32 => 4,
            IntTy::I64 => 8,
        }),
        ValueTy::UInt(uint_ty) => MemoryBound::Bounded(match uint_ty {
            UIntTy::U8 => 1,
            UIntTy::U16 => 2,
            UIntTy::U32 => 4,
            UIntTy::U64 => 8,
        }),
        ValueTy::Float(float_ty) => MemoryBound::Bounded(match float_ty {
            FloatTy::F32 => 4,
            FloatTy::F64 => 8,
        }),
        // an abstract data type, e.g., structs, enums, etc.
        //ValueTy::Adt(AdtDef),
        ValueTy::String => MemoryBound::Unbounded,
        ValueTy::Tuple(elements) => {
            let mut accu = 0_u128;
            for element in elements.iter().map(get_byte_size) {
                match element {
                    MemoryBound::Bounded(i) => accu += i,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                    MemoryBound::Unknown => return MemoryBound::Unknown,
                };
            }
            MemoryBound::Bounded(accu)
        }
        // an optional value type, e.g., resulting from accessing a stream with offset -1
        ValueTy::Option(inner) => get_byte_size(inner),
        // Used during type inference
        ValueTy::Infer(_value_var) => unreachable!(),
        ValueTy::Constr(_type_constraint) => unreachable!(),
        // A reference to a generic parameter in a function declaration, e.g. `T` in `a<T>(x:T) -> T`
        ValueTy::Param(_, _) => MemoryBound::Bounded(0),
        ValueTy::Error => MemoryBound::Unknown,
    }
}
