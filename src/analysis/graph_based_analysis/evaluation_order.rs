use crate::analysis::graph_based_analysis::get_ast_id;
use crate::analysis::graph_based_analysis::ComputationGraph;
use crate::analysis::graph_based_analysis::ComputeStep;
use crate::analysis::graph_based_analysis::DependencyGraph;
use crate::analysis::graph_based_analysis::Location;
use crate::analysis::graph_based_analysis::NIx;
use crate::analysis::graph_based_analysis::Offset;
use crate::analysis::graph_based_analysis::StreamDependency;
use crate::analysis::graph_based_analysis::StreamNode::ClassicInput;
use crate::analysis::graph_based_analysis::StreamNode::ClassicOutput;
use crate::analysis::graph_based_analysis::StreamNode::ParameterizedInput;
use crate::analysis::graph_based_analysis::StreamNode::ParameterizedOutput;
use crate::analysis::graph_based_analysis::StreamNode::RTOutput;
use crate::analysis::graph_based_analysis::StreamNode::RTTrigger;
use crate::analysis::graph_based_analysis::StreamNode::Trigger;
use crate::analysis::graph_based_analysis::TimeOffset;
use crate::parse::NodeId;
use std::collections::HashMap;

pub(crate) type EvalOrder = Vec<Vec<ComputeStep>>;

#[derive(Debug)]
pub(crate) struct EvaluationOrderResult {
    pub event_based_streams_order: EvalOrder,
    pub periodic_streams_order: EvalOrder,
}

pub(crate) fn determine_evaluation_order(
    mut dependency_graph: DependencyGraph,
) -> (EvaluationOrderResult, DependencyGraph) {
    prune_graph(&mut dependency_graph);
    let pruned_dependency_graph = dependency_graph.clone();

    let (compute_graph_event_based, compute_graph_periodic) =
        build_compute_graphs(dependency_graph);

    let event_based_streams_order = get_compute_order(compute_graph_event_based);
    let periodic_streams_order = get_compute_order(compute_graph_periodic);

    (
        EvaluationOrderResult {
            event_based_streams_order,
            periodic_streams_order,
        },
        pruned_dependency_graph,
    )
}

fn prune_graph(dependency_graph: &mut DependencyGraph) {
    // TODO In the future we will do pruning.
}

fn build_compute_graphs(dependency_graph: DependencyGraph) -> (ComputationGraph, ComputationGraph) {
    let mut normal_time_graph = dependency_graph.clone();
    normal_time_graph.retain_nodes(|g, node_index| match g.node_weight(node_index).unwrap() {
        RTTrigger(_) | RTOutput(_) => false,
        ClassicInput(_)
        | ClassicOutput(_)
        | ParameterizedInput(_)
        | ParameterizedOutput(_)
        | Trigger(_) => true,
    });
    let mut real_time_graph = dependency_graph.clone();
    real_time_graph.retain_nodes(|g, node_index| match g.node_weight(node_index).unwrap() {
        RTTrigger(_) | RTOutput(_) => true,
        ClassicInput(_)
        | ClassicOutput(_)
        | ParameterizedInput(_)
        | ParameterizedOutput(_)
        | Trigger(_) => false,
    });

    (
        build_compute_graph(normal_time_graph),
        build_compute_graph(real_time_graph),
    )
}

struct StepEntry {
    invoke: NIx,
    extend: NIx,
    evaluate: NIx,
    terminate: NIx,
}

fn build_compute_graph(mut dependency_graph: DependencyGraph) -> ComputationGraph {
    let expected_capacity = dependency_graph.node_count() * 4;
    let mut mapping: HashMap<NodeId, StepEntry> = HashMap::with_capacity(expected_capacity);
    let mut computation_graph =
        ComputationGraph::with_capacity(expected_capacity, expected_capacity);
    for node in dependency_graph.node_weights_mut() {
        // TODO ignore real time streams
        let id = get_ast_id(*node);
        let invoke: NIx = computation_graph.add_node(ComputeStep::Invoke(id));
        let extend: NIx = computation_graph.add_node(ComputeStep::Extend(id));
        let evaluate: NIx = computation_graph.add_node(ComputeStep::Evaluate(id));
        let terminate: NIx = computation_graph.add_node(ComputeStep::Terminate(id));
        computation_graph.add_edge(extend, invoke, ());
        computation_graph.add_edge(terminate, invoke, ());
        computation_graph.add_edge(evaluate, extend, ());
        mapping.insert(
            id,
            StepEntry {
                invoke,
                extend,
                evaluate,
                terminate,
            },
        );
    }

    for edge_idx in dependency_graph.edge_indices() {
        let edge_weight = dependency_graph
            .edge_weight(edge_idx)
            .expect("We iterate over all existing EdgeIndices");
        let (source, target) = dependency_graph
            .edge_endpoints(edge_idx)
            .expect("We iterate over all existing EdgeIndices");
        let source_id = get_ast_id(
            *dependency_graph
                .node_weight(source)
                .expect("We just git this NodeIndex from the graph"),
        );
        let target_id = get_ast_id(
            *dependency_graph
                .node_weight(target)
                .expect("We just git this NodeIndex from the graph"),
        );

        match edge_weight {
            StreamDependency::InvokeByName(_) => {
                // we need to know the values so we need evaluate
                computation_graph.add_edge(
                    mapping[&source_id].invoke,
                    mapping[&target_id].evaluate,
                    (),
                );
            }
            StreamDependency::Access(location, offset, _) => {
                match offset {
                    Offset::Discrete(value) => {
                        if *value < 0 {
                            // we only need whether there was a new value
                            match location {
                                Location::Invoke => computation_graph.add_edge(
                                    mapping[&source_id].invoke,
                                    mapping[&target_id].extend,
                                    (),
                                ),
                                Location::Extend => computation_graph.add_edge(
                                    mapping[&source_id].extend,
                                    mapping[&target_id].extend,
                                    (),
                                ),
                                Location::Expression => computation_graph.add_edge(
                                    mapping[&source_id].evaluate,
                                    mapping[&target_id].extend,
                                    (),
                                ),
                                Location::Terminate => computation_graph.add_edge(
                                    mapping[&source_id].terminate,
                                    mapping[&target_id].extend,
                                    (),
                                ),
                            };
                        } else if *value == 0 {
                            // we need the current value
                            match location {
                                Location::Invoke => computation_graph.add_edge(
                                    mapping[&source_id].invoke,
                                    mapping[&target_id].evaluate,
                                    (),
                                ),
                                Location::Extend => computation_graph.add_edge(
                                    mapping[&source_id].extend,
                                    mapping[&target_id].evaluate,
                                    (),
                                ),
                                Location::Expression => computation_graph.add_edge(
                                    mapping[&source_id].evaluate,
                                    mapping[&target_id].evaluate,
                                    (),
                                ),
                                Location::Terminate => computation_graph.add_edge(
                                    mapping[&source_id].terminate,
                                    mapping[&target_id].evaluate,
                                    (),
                                ),
                            };
                        }
                        // no need to add dependency for positive edges
                    }
                    Offset::Time(offset) => {
                        if let TimeOffset::UpToNow(_, _) = offset {
                            computation_graph.add_edge(
                                mapping[&source_id].evaluate,
                                mapping[&target_id].evaluate,
                                (),
                            );
                        }
                    }
                    Offset::SlidingWindow => {
                        if source_id != target_id {
                            computation_graph.add_edge(
                                mapping[&source_id].evaluate,
                                mapping[&target_id].evaluate,
                                (),
                            );
                        }
                    }
                }
            }
        }
    }
    computation_graph
}

fn get_compute_order(mut compute_graph: ComputationGraph) -> Vec<Vec<ComputeStep>> {
    let mut compute_step_order: Vec<Vec<ComputeStep>> = Vec::new();

    while compute_graph.node_count() > 0 {
        compute_step_order.push(Vec::new());

        let mut pruned_nodes = Vec::new();
        for index in compute_graph.node_indices() {
            if compute_graph.neighbors(index).next().is_none() {
                // all dependencies are already in the order
                pruned_nodes.push(index); // mark this node for pruning
                compute_step_order
                    .last_mut()
                    .expect("We always push a new vector at the beginning of the iteration.")
                    .push(
                        *compute_graph
                            .node_weight(index)
                            .expect("The indices are provided by the graph."),
                    );
            }
        }

        // remove the current front
        for index in pruned_nodes.iter().rev() {
            compute_graph.remove_node(*index);
        }
    }

    // now prune empty steps ones from the back
    while !compute_step_order.is_empty()
        && compute_step_order
            .last()
            .expect("We just checked that the vector is not empty.")
            .is_empty()
    {
        compute_step_order.pop();
    }
    compute_step_order
}
