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
use crate::analysis::graph_based_analysis::TimeOffset;
use crate::analysis::graph_based_analysis::TrackingRequirement;
use crate::analysis::typing::TypeTable;
use crate::parse::NodeId;
use crate::ty::TimingInfo;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::cmp::max;
use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Duration;

const NANOS_PER_SEC: u32 = 1_000_000_000;

fn dur_as_nanos(dur: Duration) -> u128 {
    u128::from(dur.as_secs()) * u128::from(NANOS_PER_SEC) + u128::from(dur.subsec_nanos())
}

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
    type_table: &TypeTable,
    future_dependent_stream: &HashSet<NodeId>,
) -> TrackingRequirements {
    let mut tracking: HashMap<NodeId, Vec<(NodeId, TrackingRequirement)>> = HashMap::new();
    for node_index in dependency_graph.node_indices() {
        let id = get_ast_id(*dependency_graph.node_weight(node_index).unwrap());
        let mut tracking_requirements: Vec<(NodeId, TrackingRequirement)> = Vec::new();

        let this_is_time_based = is_it_time_based(dependency_graph, node_index, type_table);

        for edge in dependency_graph.edges_directed(node_index, Direction::Outgoing) {
            // TODO What about the invoke expression (if we allow one)?
            if let Access(Location::Expression, Time(offset), _) = edge.weight() {
                let src_node = *dependency_graph.node_weight(edge.target()).unwrap();
                let src_id = get_ast_id(src_node);
                assert!(
                    !future_dependent_stream.contains(&src_id),
                    "time based access of future dependent streams is not implemented"
                ); // TODO time based access of future dependent streams is not implemented
                if let TimeOffset::UpToNow(offset_duration) = offset {
                    let src_timing = &type_table.get_stream_type(src_id).timing;
                    if this_is_time_based {
                        let out_timing = &type_table.get_stream_type(id).timing;
                        if let TimingInfo::RealTime(freq) = out_timing {
                            let complete_periods =
                                (dur_as_nanos(*offset_duration) / dur_as_nanos(freq.d)) as u16;
                            let needed_space =
                                if dur_as_nanos(*offset_duration) % dur_as_nanos(freq.d) == 0 {
                                    complete_periods
                                } else {
                                    complete_periods + 1
                                };
                            tracking_requirements
                                .push((src_id, TrackingRequirement::Finite(needed_space)));
                        // TODO We might be able to use the max(src_duration, out_duration)
                        } else {
                            unreachable!()
                        }
                    } else {
                        match src_timing {
                            TimingInfo::Event => {
                                tracking_requirements
                                    .push((src_id, TrackingRequirement::Unbounded));
                            }
                            TimingInfo::RealTime(freq) => {
                                let complete_periods =
                                    (dur_as_nanos(*offset_duration) / dur_as_nanos(freq.d)) as u16;
                                let needed_space =
                                    if dur_as_nanos(*offset_duration) % dur_as_nanos(freq.d) == 0 {
                                        complete_periods
                                    } else {
                                        complete_periods + 1
                                    };
                                tracking_requirements
                                    .push((src_id, TrackingRequirement::Finite(needed_space)));
                            }
                        }
                    }
                } else {
                    tracking_requirements.push((src_id, TrackingRequirement::Future));
                }
            }
        }

        tracking.insert(id, tracking_requirements);
    }
    tracking
}

fn is_it_time_based(
    dependency_graph: &DependencyGraph,
    node_index: NIx,
    type_table: &TypeTable,
) -> bool {
    let id = get_ast_id(*dependency_graph.node_weight(node_index).unwrap());
    match type_table.get_stream_type(id).timing {
        TimingInfo::Event => false,
        TimingInfo::RealTime(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use crate::analysis::graph_based_analysis::dependency_graph::analyse_dependencies;
    use crate::analysis::graph_based_analysis::evaluation_order::determine_evaluation_order;
    use crate::analysis::graph_based_analysis::future_dependency::future_dependent_stream;
    use crate::analysis::graph_based_analysis::space_requirements::determine_buffer_size;
    use crate::analysis::graph_based_analysis::space_requirements::determine_tracking_size;
    use crate::analysis::graph_based_analysis::StorageRequirement;
    use crate::analysis::graph_based_analysis::TrackingRequirement;
    use crate::analysis::lola_version::LolaVersionAnalysis;
    use crate::analysis::naming::NamingAnalysis;
    use crate::analysis::typing::TypeAnalysis;
    use crate::parse::parse;
    use crate::parse::SourceMapper;
    use crate::reporting::Handler;
    use std::path::PathBuf;

    #[derive(Debug, Clone, Copy)]
    enum StreamIndex {
        Out(usize),
        In(usize),
        Trig(usize),
    }

    fn check_buffer_size(
        content: &str,
        expected_errors: usize,
        expected_warning: usize,
        expected_number_of_entries: usize,
        expected_buffer_size: Vec<(StreamIndex, StorageRequirement)>,
    ) {
        let spec = parse(content).unwrap_or_else(|e| panic!("{}", e));
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let mut naming_analyzer = NamingAnalysis::new(&handler);
        let decl_table = naming_analyzer.check(&spec);
        let mut version_analyzer = LolaVersionAnalysis::new(&handler);
        let _version = version_analyzer.analyse(&spec);

        let dependency_analysis =
            analyse_dependencies(&spec, &version_analyzer.result, &decl_table, &handler);

        let (_, pruned_graph) = determine_evaluation_order(dependency_analysis.dependency_graph);

        let future_dependent_stream = future_dependent_stream(&pruned_graph);

        let space_requirements = determine_buffer_size(&pruned_graph, &future_dependent_stream);

        assert_eq!(expected_errors, handler.emitted_errors());
        assert_eq!(expected_warning, handler.emitted_warnings());
        assert_eq!(expected_number_of_entries, space_requirements.len());
        for (index, expected_buffer_size) in expected_buffer_size {
            let node_id = match index {
                StreamIndex::Out(i) => spec.outputs[i].id,
                StreamIndex::In(i) => spec.inputs[i].id,
                StreamIndex::Trig(i) => spec.trigger[i].id,
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

    fn check_tracking_size(
        content: &str,
        expected_errors: usize,
        expected_warning: usize,
        expected_number_of_entries: usize,
        expected_tracking_requirements: Vec<(StreamIndex, Vec<(StreamIndex, TrackingRequirement)>)>,
    ) {
        let spec = parse(content).unwrap_or_else(|e| panic!("{}", e));
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let mut naming_analyzer = NamingAnalysis::new(&handler);
        let decl_table = naming_analyzer.check(&spec);
        let mut version_analyzer = LolaVersionAnalysis::new(&handler);
        let _version = version_analyzer.analyse(&spec);

        let dependency_analysis =
            analyse_dependencies(&spec, &version_analyzer.result, &decl_table, &handler);
        let mut type_analysis = TypeAnalysis::new(&handler, &decl_table);
        let type_table = type_analysis.check(&spec);
        let (_, pruned_graph) = determine_evaluation_order(dependency_analysis.dependency_graph);

        let future_dependent_stream = future_dependent_stream(&pruned_graph);

        let tracking_requirements = determine_tracking_size(
            &pruned_graph,
            &type_table.unwrap(),
            &future_dependent_stream,
        );

        assert_eq!(expected_errors, handler.emitted_errors());
        assert_eq!(expected_warning, handler.emitted_warnings());
        assert_eq!(expected_number_of_entries, tracking_requirements.len());
        for (index, expected_tracking_info) in expected_tracking_requirements {
            let node_id = match index {
                StreamIndex::In(i) => spec.inputs[i].id,
                StreamIndex::Out(i) => spec.outputs[i].id,
                StreamIndex::Trig(i) => spec.trigger[i].id,
            };
            let actual_tracking_info = tracking_requirements.get(&node_id).unwrap_or_else(|| {
                panic!("There is no tracking info for this NodeId in the result",)
            });

            assert_eq!(
                expected_tracking_info.len(),
                actual_tracking_info.len(),
                "The number of expected entries is {} but {} are present",
                expected_tracking_info.len(),
                actual_tracking_info.len()
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

    #[test]
    #[ignore]
    fn tracking_the_future() {
        check_tracking_size(
            "output a :Int8 := b[+1s]?0 input b: Int8",
            0,
            0,
            2,
            vec![
                (
                    StreamIndex::Out(0),
                    vec![(StreamIndex::In(0), TrackingRequirement::Future)],
                ),
                (StreamIndex::In(0), vec![]),
            ],
        )
    }

    #[test]
    #[ignore]
    fn tracking_the_event_based_past() {
        check_tracking_size(
            "output a :Int8 := b[-1s]?0 input b: Int8",
            0,
            0,
            2,
            vec![
                (
                    StreamIndex::Out(0),
                    vec![(StreamIndex::In(0), TrackingRequirement::Unbounded)],
                ),
                (StreamIndex::In(0), vec![]),
            ],
        )
    }

    #[test]
    #[ignore]
    fn tracking_the_time_based_past() {
        check_tracking_size(
            "output a :Int8 := c[-1s]?0 input b: Int8 output c: Int8 {extend @ 1 s} := b[0]?0",
            0,
            0,
            3,
            vec![
                (
                    StreamIndex::Out(0),
                    vec![(StreamIndex::Out(1), TrackingRequirement::Finite(1))],
                ),
                (StreamIndex::In(0), vec![]),
                (StreamIndex::Out(1), vec![]),
            ],
        )
    }

    #[test]
    #[ignore]
    fn time_based_tracking_the_past() {
        check_tracking_size(
            "output a :Int8 {extend @ 1s} := b[-2.5s]?0 input b: Int8",
            0,
            0,
            2,
            vec![
                (
                    StreamIndex::Out(0),
                    vec![(StreamIndex::In(0), TrackingRequirement::Finite(3))],
                ),
                (StreamIndex::In(0), vec![]),
            ],
        )
    }
}
