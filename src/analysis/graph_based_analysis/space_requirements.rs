use crate::analysis::graph_based_analysis::get_ast_id;
use crate::analysis::graph_based_analysis::DependencyGraph;
use crate::analysis::graph_based_analysis::Location;
use crate::analysis::graph_based_analysis::Offset::Discrete;
use crate::analysis::graph_based_analysis::StorageRequirement;
use crate::analysis::graph_based_analysis::StreamDependency::Access;
use crate::analysis::graph_based_analysis::StreamDependency::InvokeByName;
use crate::analysis::graph_based_analysis::StreamNode::*;
use ast_node::NodeId;
use petgraph::Direction;
use std::cmp::max;
use std::collections::HashMap;
use std::collections::HashSet;

pub(crate) fn determine_space_requirements(
    dependency_graph: &DependencyGraph,
    future_dependent_streams: &HashSet<NodeId>,
) -> HashMap<NodeId, StorageRequirement> {
    let mut store_all_inputs = false;
    let mut storage_requirements: HashMap<NodeId, StorageRequirement> = HashMap::new();
    for node_index in dependency_graph.node_indices() {
        let this_stream_is_future_dependent = future_dependent_streams.contains(&get_ast_id(
            *dependency_graph.node_weight(node_index).unwrap(),
        ));
        match dependency_graph.node_weight(node_index).unwrap() {
            ClassicInput(id)
            | ParameterizedInput(id)
            | ClassicOutput(id)
            | ParameterizedOutput(id)
            | Trigger(id) => {
                // normal event based stream
                let mut storage_required = 0_u16;
                for edge in dependency_graph.edges_directed(node_index, Direction::Incoming) {
                    match edge.weight() {
                        Access(location, offset, _) => {
                            if let Discrete(offset) = offset {
                                if *offset < 0 {
                                    storage_required =
                                        max(storage_required, 1 + (*offset).abs() as u16);
                                }
                                if *offset > 0
                                    && this_stream_is_future_dependent
                                    && !store_all_inputs
                                    && *location != Location::Expression
                                {
                                    // future dependency in auxiliary streams is really bad
                                    store_all_inputs = true;
                                }
                            }
                        }
                        InvokeByName(_) => storage_required = max(storage_required, 1_u16),
                    }
                }

                if this_stream_is_future_dependent {
                    storage_requirements
                        .insert(*id, StorageRequirement::FutureRef(storage_required));
                } else {
                    storage_requirements.insert(*id, StorageRequirement::Finite(storage_required));
                }
            }
            RTOutput(_id) | RTTrigger(_id) => unimplemented!(),
        }
    }

    // we have a future reference in
    if store_all_inputs {
        for node_index in dependency_graph.node_indices() {
            let node_weight = dependency_graph.node_weight(node_index).unwrap();
            match node_weight {
                ClassicInput(id) | ParameterizedInput(id) => {
                    storage_requirements.insert(*id, StorageRequirement::Unbounded);
                }
                _ => {}
            }
        }
    }

    storage_requirements
}
