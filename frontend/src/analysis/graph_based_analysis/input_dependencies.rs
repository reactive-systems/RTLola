use crate::analysis::graph_based_analysis::{get_ast_id, DependencyGraph, NIx, StreamNode};
use crate::parse::NodeId;
use petgraph::visit::IntoNodeIdentifiers;
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

pub(crate) type RequiredInputs = HashMap<NodeId, Vec<NodeId>>;

pub(crate) fn determine_required_inputs(dependency_graph: &DependencyGraph) -> RequiredInputs {
    let mut input_dependencies: RequiredInputs = HashMap::new();
    for node in dependency_graph.node_identifiers() {
        let node_info = dependency_graph.node_weight(node).expect("we iterate over the NIx");
        input_dependencies.insert(get_ast_id(node_info), Vec::new());
    }

    for node in dependency_graph.node_identifiers() {
        let node_info = dependency_graph.node_weight(node).expect("we iterate over the NIx");
        let mut visited: HashSet<NIx> = HashSet::new();
        let mut stack: Vec<NIx> = Vec::new();
        if let StreamNode::ClassicInput(node_id) = node_info {
            visited.clear();
            stack.push(node);
            visited.insert(node);
            input_dependencies.get_mut(node_id).unwrap().push(*node_id);
            bfs_color_reachable(&mut visited, &mut stack, *node_id, &mut input_dependencies, dependency_graph);
        }
    }
    input_dependencies
}

fn bfs_color_reachable(
    visited: &mut HashSet<NIx>,
    stack: &mut Vec<NIx>,
    node_id: NodeId,
    input_dependencies: &mut RequiredInputs,
    dependency_graph: &DependencyGraph,
) {
    while !stack.is_empty() {
        let top = stack.pop().expect("We checked for empty stack");
        for dependent_stream in dependency_graph.neighbors_directed(top, Direction::Incoming) {
            if visited.insert(dependent_stream) {
                let id = get_ast_id(dependency_graph.node_weight(dependent_stream).expect("We just got this NIx"));
                input_dependencies.get_mut(&id).unwrap().push(node_id);
                stack.push(dependent_stream)
            }
        }
    }
}
