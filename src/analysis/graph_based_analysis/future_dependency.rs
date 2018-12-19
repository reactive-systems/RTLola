use crate::analysis::graph_based_analysis::get_ast_id;
use crate::analysis::graph_based_analysis::DependencyGraph;
use crate::analysis::graph_based_analysis::NIx;
use crate::analysis::graph_based_analysis::Offset;
use crate::analysis::graph_based_analysis::StreamDependency::Access;
use crate::analysis::graph_based_analysis::StreamNode::*;
use ast_node::NodeId;
use petgraph::Direction;
use std::collections::HashSet;

pub(crate) fn future_dependent_stream(dependency_graph: &DependencyGraph) -> HashSet<NodeId> {
    let mut future_dependent_streams: HashSet<NodeId> = HashSet::new();

    for node_index in dependency_graph.node_indices() {
        // check if this node is already in the future_dependent_set
        if future_dependent_streams.contains(&get_ast_id(
            *(dependency_graph.node_weight(node_index).unwrap()),
        )) {
            continue;
        }

        // check if this node has an edge with future dependence

        let directly_future_dependent = dependency_graph
            .edges_directed(node_index, Direction::Outgoing)
            .any(|edge| match edge.weight() {
                Access(_, offset, _) => match offset {
                    Offset::Discrete(offset) => *offset > 0,
                    Offset::Time(offset, _) => *offset > 0.0,
                },
                _ => false,
            });

        if directly_future_dependent {
            propagate_future_dependence(
                &mut future_dependent_streams,
                &dependency_graph,
                node_index,
            );
        }
    }

    future_dependent_streams
}

fn propagate_future_dependence(
    future_dependent_streams: &mut HashSet<NodeId>,
    dependency_graph: &DependencyGraph,
    node_index: NIx,
) {
    future_dependent_streams.insert(get_ast_id(
        *(dependency_graph.node_weight(node_index).unwrap()),
    ));

    for neighbor in dependency_graph.neighbors_directed(node_index, Direction::Incoming) {
        if !future_dependent_streams.contains(&get_ast_id(
            *(dependency_graph.node_weight(neighbor).unwrap()),
        )) {
            propagate_future_dependence(future_dependent_streams, dependency_graph, neighbor);
        }
    }
}
