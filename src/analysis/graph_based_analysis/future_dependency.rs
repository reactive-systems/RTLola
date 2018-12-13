use crate::analysis::graph_based_analysis::get_ast_id;
use crate::analysis::graph_based_analysis::DependencyGraph;
use crate::analysis::graph_based_analysis::NIx;
use crate::analysis::graph_based_analysis::Offset;
use crate::analysis::graph_based_analysis::StreamDependency::Access;
use crate::analysis::graph_based_analysis::StreamNode::*;
use ast_node::NodeId;
use petgraph::Direction;
use std::collections::HashSet;

/// Computes the set of streams and triggers (represented by their NodeId)
/// that (transitively) depend on future values.
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

/// Recursively mark all dependent streams and triggers as future dependent.
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

#[cfg(test)]
mod tests {
    use crate::analysis::graph_based_analysis::dependency_graph::analyse_dependencies;
    use crate::analysis::graph_based_analysis::evaluation_order::determine_evaluation_order;
    use crate::analysis::graph_based_analysis::future_dependency::future_dependent_stream;
    use crate::analysis::id_assignment;
    use crate::analysis::lola_version::LolaVersionAnalysis;
    use crate::analysis::naming::NamingAnalysis;
    use crate::parse::parse;
    use crate::parse::SourceMapper;
    use crate::reporting::Handler;
    use std::path::PathBuf;

    fn check_future_dependence(
        content: &str,
        expected_errors: usize,
        expected_warning: usize,
        expected_number_future_dependent: usize,
        // index of output stream with future dependence
        expected_future_dependent: Vec<usize>,
    ) {
        let mut spec = parse(content).unwrap_or_else(|e| panic!("{}", e));
        id_assignment::assign_ids(&mut spec);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let mut naming_analyzer = NamingAnalysis::new(&handler);
        naming_analyzer.check(&spec);
        let mut version_analyzer = LolaVersionAnalysis::new(&handler);
        let version = version_analyzer.analyse(&spec);

        let dependency_analysis = analyse_dependencies(
            &spec,
            &version_analyzer.result,
            &naming_analyzer.result,
            &handler,
        );

        let evaluation_order_result =
            determine_evaluation_order(dependency_analysis.dependency_graph);

        let future_dependent_stream =
            future_dependent_stream(&evaluation_order_result.pruned_dependency_graph);

        assert_eq!(expected_errors, handler.emitted_errors());
        assert_eq!(expected_warning, handler.emitted_warnings());
        assert_eq!(
            expected_number_future_dependent,
            future_dependent_stream.len()
        );
        for index in expected_future_dependent {
            let output = &spec.outputs[index];
            assert!(future_dependent_stream.contains(&output._id));
        }
    }

    #[test]
    fn a_simple_future_dependence() {
        check_future_dependence("output a := b[+1]?0 input b: Int8", 0, 0, 1, vec![0])
    }
    #[test]
    fn a_simple_same_time_dependence_is_not_a_future_dependence() {
        check_future_dependence("output a := b[0]?0 input b: Int8", 0, 0, 0, vec![])
    }
    #[test]
    fn a_simple_past_dependence_is_not_a_future_dependence() {
        check_future_dependence("output a := b[-1]?0 input b: Int8", 0, 0, 0, vec![])
    }

}
