use crate::analysis::graph_based_analysis::get_ast_id;
use crate::analysis::graph_based_analysis::DependencyGraph;
use crate::analysis::graph_based_analysis::Location;
use crate::analysis::graph_based_analysis::NIx;
use crate::analysis::graph_based_analysis::Offset::Discrete;
use crate::analysis::graph_based_analysis::Offset::Time;
use crate::analysis::graph_based_analysis::StorageRequirement;
use crate::analysis::graph_based_analysis::StreamDependency::Access;
use crate::analysis::graph_based_analysis::StreamDependency::InvokeByName;
use crate::analysis::graph_based_analysis::StreamNode::*;
use crate::analysis::graph_based_analysis::TrackingRequirement;
use ast_node::NodeId;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::cmp::max;
use std::collections::HashMap;
use std::collections::HashSet;

pub(crate) type SpaceRequirements = HashMap<NodeId, StorageRequirement>;

pub(crate) fn determine_buffer_size(
    dependency_graph: &DependencyGraph,
    future_dependent_streams: &HashSet<NodeId>,
) -> SpaceRequirements {
    let mut store_all_inputs = false;
    let mut storage_requirements: HashMap<NodeId, StorageRequirement> = HashMap::new();
    for node_index in dependency_graph.node_indices() {
        let id = get_ast_id(*dependency_graph.node_weight(node_index).unwrap());
        let this_stream_is_future_dependent = future_dependent_streams.contains(&id);

        // normal event based stream
        let mut storage_required = 0_u16;
        for edge in dependency_graph.edges_directed(node_index, Direction::Incoming) {
            match edge.weight() {
                Access(location, offset, _) => {
                    if let Discrete(offset) = offset {
                        if *offset <= 0 {
                            storage_required = max(storage_required, 1 + (*offset).abs() as u16);
                        }

                        if *offset > 0 {
                            storage_required = max(storage_required, 1_u16);
                            if this_stream_is_future_dependent
                                && !store_all_inputs
                                && *location != Location::Expression
                            {
                                // future dependency in auxiliary streams is really bad
                                store_all_inputs = true;
                            }
                        }
                    }
                }
                InvokeByName(_) => storage_required = max(storage_required, 1_u16),
            }
        }

        if this_stream_is_future_dependent {
            storage_requirements.insert(id, StorageRequirement::FutureRef(storage_required));
        } else {
            storage_requirements.insert(id, StorageRequirement::Finite(storage_required));
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

pub(crate) type TrackingRequirements = HashMap<NodeId, Vec<(NodeId, TrackingRequirement)>>;

pub(crate) fn determine_tracking_size(
    dependency_graph: &DependencyGraph,
    // TODO add typing
) -> TrackingRequirements {
    let mut tracking: HashMap<NodeId, Vec<(NodeId, TrackingRequirement)>> = HashMap::new();
    for node_index in dependency_graph.node_indices() {
        let id = get_ast_id(*dependency_graph.node_weight(node_index).unwrap());
        let tracking_requirement: Vec<(NodeId, TrackingRequirement)> = Vec::new();

        let this_is_time_based = is_it_time_based(dependency_graph, node_index);

        for edge in dependency_graph.edges_directed(node_index, Direction::Outgoing) {
            match edge.weight() {
                Access(location, offset, _) => {
                    if let Time(offset, unit) = offset {
                        let target_id =
                            get_ast_id(*dependency_graph.node_weight(edge.target()).unwrap());

                        unimplemented!()
                    }
                }
                InvokeByName(_) => {}
            }
        }

        tracking.insert(id, tracking_requirement);
    }
    tracking
}

fn is_it_time_based(
    dependency_graph: &DependencyGraph,
    node_index: NIx,
    // TODO add typing information
) -> bool {
    let id = get_ast_id(*dependency_graph.node_weight(node_index).unwrap());

    unimplemented!()
}

#[cfg(test)]
mod tests {
    use crate::analysis::graph_based_analysis::dependency_graph::analyse_dependencies;
    use crate::analysis::graph_based_analysis::evaluation_order::determine_evaluation_order;
    use crate::analysis::graph_based_analysis::future_dependency::future_dependent_stream;
    use crate::analysis::graph_based_analysis::space_requirements::determine_buffer_size;
    use crate::analysis::graph_based_analysis::StorageRequirement;
    use crate::analysis::id_assignment;
    use crate::analysis::lola_version::LolaVersionAnalysis;
    use crate::analysis::naming::NamingAnalysis;
    use crate::parse::parse;
    use crate::parse::SourceMapper;
    use crate::reporting::Handler;
    use std::path::PathBuf;

    #[derive(Debug, Clone, Copy)]
    enum StreamIndex {
        Out(usize),
        In(usize),
    }

    fn check_buffer_size(
        content: &str,
        expected_errors: usize,
        expected_warning: usize,
        expected_size: usize,
        expected_future_dependent: Vec<(StreamIndex, StorageRequirement)>,
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

        let space_requirements = determine_buffer_size(
            &evaluation_order_result.pruned_dependency_graph,
            &future_dependent_stream,
        );

        assert_eq!(expected_errors, handler.emitted_errors());
        assert_eq!(expected_warning, handler.emitted_warnings());
        assert_eq!(expected_size, space_requirements.len());
        for (index, expected_buffer_size) in expected_future_dependent {
            let node_id = match index {
                StreamIndex::Out(i) => spec.outputs[i]._id,
                StreamIndex::In(i) => spec.inputs[i]._id,
            };
            let actual_buffer = space_requirements.get(&node_id).unwrap_or_else(|| {
                panic!("There is no buffer size for this NodeId in the result",)
            });
            assert_eq!(
                expected_buffer_size, *actual_buffer,
                "The expected buffer size and the actual buffer size do not match."
            );
        }
    }

    #[test]
    fn a_simple_past_dependence() {
        check_buffer_size(
            "output a := b[-1]?0 input b: Int8",
            0,
            0,
            2,
            vec![
                (StreamIndex::Out(0), StorageRequirement::Finite(0)),
                (StreamIndex::In(0), StorageRequirement::Finite(2)),
            ],
        )
    }

    #[test]
    fn a_simple_current_dependence() {
        check_buffer_size(
            "output a := b[0]?0 input b: Int8",
            0,
            0,
            2,
            vec![
                (StreamIndex::Out(0), StorageRequirement::Finite(0)),
                (StreamIndex::In(0), StorageRequirement::Finite(1)),
            ],
        )
    }

    #[test]
    fn a_simple_future_dependence() {
        check_buffer_size(
            "output a := b[+1]?0 input b: Int8",
            0,
            0,
            2,
            vec![
                (StreamIndex::Out(0), StorageRequirement::FutureRef(0)),
                (StreamIndex::In(0), StorageRequirement::Finite(1)),
            ],
        )
    }
}
