use crate::analysis::graph_based_analysis::Offset::Discrete;
use crate::analysis::graph_based_analysis::Offset::Time;
use crate::analysis::graph_based_analysis::StreamDependency::{Access, InvokeByName};
use crate::analysis::graph_based_analysis::{
    get_ast_id, DependencyGraph, Location, NIx, StorageRequirement, StreamNode, TimeOffset, TrackingRequirement,
};
use crate::parse::NodeId;
use crate::ty::check::TypeTable;
use crate::ty::StreamTy;
use num::rational::Rational64 as Rational;
use num::ToPrimitive;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::cmp::max;
use std::collections::HashMap;
use std::collections::HashSet;
use uom::si::frequency::hertz;
use uom::si::time::second;

pub(crate) type SpaceRequirements = HashMap<NodeId, StorageRequirement>;

pub(crate) fn determine_buffer_size(
    dependency_graph: &DependencyGraph,
    future_dependent_streams: &HashSet<NodeId>,
) -> SpaceRequirements {
    let mut store_all_inputs = false;
    let mut storage_requirements: HashMap<NodeId, StorageRequirement> = HashMap::new();
    for node_index in dependency_graph.node_indices() {
        let id = get_ast_id(dependency_graph.node_weight(node_index).expect("We iterate over all node indices"));
        let this_stream_is_future_dependent = future_dependent_streams.contains(&id);

        // normal event based stream
        let mut storage_required = 1_u16;
        for edge in dependency_graph.edges_directed(node_index, Direction::Incoming) {
            match edge.weight() {
                Access(location, offset, _) => {
                    if let Discrete(offset) = offset {
                        if *offset <= 0 {
                            storage_required = max(storage_required, 1 + (*offset).abs() as u16);
                        }

                        if *offset > 0 {
                            storage_required = max(storage_required, 1_u16);
                            if this_stream_is_future_dependent && !store_all_inputs && *location != Location::Expression
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
            let node_weight = dependency_graph.node_weight(node_index).expect("We iterate over all node indices");
            if let StreamNode::ClassicInput(id) = node_weight {
                storage_requirements.insert(*id, StorageRequirement::Unbounded);
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
) -> Result<TrackingRequirements, String> {
    let mut tracking: HashMap<NodeId, Vec<(NodeId, TrackingRequirement)>> = HashMap::new();
    for node_index in dependency_graph.node_indices() {
        let id = get_ast_id(dependency_graph.node_weight(node_index).expect("We iterate over all node indices"));
        let mut tracking_requirements: Vec<(NodeId, TrackingRequirement)> = Vec::new();

        let this_is_time_based = is_it_time_based(dependency_graph, node_index, type_table);

        for edge in dependency_graph.edges_directed(node_index, Direction::Outgoing) {
            // TODO What about the invoke expression (if we allow one)?
            if let Access(Location::Expression, Time(offset), _) = edge.weight() {
                let src_node = dependency_graph
                    .node_weight(edge.target())
                    .expect("We iterate over edges so their target should exist");
                let src_id = get_ast_id(src_node);
                assert!(
                    !future_dependent_stream.contains(&src_id),
                    "time based access of future dependent streams is not implemented"
                ); // TODO time based access of future dependent streams is not implemented
                if let TimeOffset::UpToNow(uom_time) = offset {
                    let src_timing = &type_table.get_stream_type(src_id);
                    if this_is_time_based {
                        let out_timing = &type_table.get_stream_type(id);
                        if let StreamTy::RealTime(freq) = out_timing {
                            let result: Rational = uom_time.get::<second>() / freq.freq.get::<hertz>();
                            let needed_space = match result.ceil().to_integer().to_u16() {
                                Some(u) => u,
                                _ => return Err("buffer size does not fit in u16".to_string()),
                            };
                            tracking_requirements.push((src_id, TrackingRequirement::Finite(needed_space)));
                        // TODO We might be able to use the max(src_duration, out_duration)
                        } else {
                            unreachable!()
                        }
                    } else {
                        match src_timing {
                            StreamTy::Event(_) => {
                                tracking_requirements.push((src_id, TrackingRequirement::Unbounded));
                            }
                            StreamTy::RealTime(freq) => {
                                let result: Rational = uom_time.get::<second>() / freq.freq.get::<hertz>();
                                let needed_space = match result.ceil().to_integer().to_u16() {
                                    Some(u) => u,
                                    _ => return Err("buffer size does not fit in u16".to_string()),
                                };
                                tracking_requirements.push((src_id, TrackingRequirement::Finite(needed_space)));
                            }
                            _ => unreachable!(),
                        }
                    }
                } else {
                    tracking_requirements.push((src_id, TrackingRequirement::Future));
                }
            }
        }

        tracking.insert(id, tracking_requirements);
    }
    Ok(tracking)
}

fn is_it_time_based(dependency_graph: &DependencyGraph, node_index: NIx, type_table: &TypeTable) -> bool {
    let id = get_ast_id(
        dependency_graph
            .node_weight(node_index)
            .expect("We assume that the type-table has information about every stream and trigger"),
    );
    match type_table.get_stream_type(id) {
        StreamTy::Event(_) => false,
        StreamTy::RealTime(_) => true,
        StreamTy::Infer(_) => unreachable!(),
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
    use crate::analysis::naming::NamingAnalysis;
    use crate::parse::parse;
    use crate::parse::SourceMapper;
    use crate::reporting::Handler;
    use crate::ty::check::TypeAnalysis;
    use crate::FrontendConfig;
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
        expected_number_of_entries: usize,
        expected_buffer_size: Vec<(StreamIndex, StorageRequirement)>,
    ) {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let spec = parse(content, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        let mut naming_analyzer = NamingAnalysis::new(&handler, FrontendConfig::default());
        let mut decl_table = naming_analyzer.check(&spec);
        let mut type_analysis = TypeAnalysis::new(&handler, &mut decl_table);
        let type_table = type_analysis.check(&spec);
        let type_table = type_table.as_ref().expect("We expect that the version analysis found no error");

        let dependency_analysis = analyse_dependencies(&spec, &decl_table, &handler, &type_table);

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
            };
            let actual_buffer = space_requirements
                .get(&node_id)
                .unwrap_or_else(|| panic!("There is no buffer size for this NodeId in the result",));
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
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let spec = parse(content, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        let mut naming_analyzer = NamingAnalysis::new(&handler, FrontendConfig::default());
        let mut decl_table = naming_analyzer.check(&spec);
        let mut type_analysis = TypeAnalysis::new(&handler, &mut decl_table);
        let type_table = type_analysis.check(&spec);
        let type_table = type_table.as_ref().expect("We expect in these tests that the type analysis checks out.");

        let dependency_analysis = analyse_dependencies(&spec, &decl_table, &handler, &type_table);
        let _ = TypeAnalysis::new(&handler, &mut decl_table);
        let (_, pruned_graph) = determine_evaluation_order(dependency_analysis.dependency_graph);

        let future_dependent_stream = future_dependent_stream(&pruned_graph);

        let tracking_requirements = determine_tracking_size(&pruned_graph, &type_table, &future_dependent_stream)
            .expect("determining tracking size failed");

        assert_eq!(expected_errors, handler.emitted_errors());
        assert_eq!(expected_warning, handler.emitted_warnings());
        assert_eq!(expected_number_of_entries, tracking_requirements.len());
        for (index, expected_tracking_info) in expected_tracking_requirements {
            let node_id = match index {
                StreamIndex::In(i) => spec.inputs[i].id,
                StreamIndex::Out(i) => spec.outputs[i].id,
            };
            let actual_tracking_info = tracking_requirements
                .get(&node_id)
                .unwrap_or_else(|| panic!("There is no tracking info for this NodeId in the result",));

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
            "output a := b[-1].defaults(to:0) input b: Int8",
            0,
            0,
            2,
            vec![
                (StreamIndex::Out(0), StorageRequirement::Finite(1)),
                (StreamIndex::In(0), StorageRequirement::Finite(2)),
            ],
        )
    }

    #[test]
    fn a_simple_current_dependence() {
        check_buffer_size(
            "output a := b[0] input b: Int8",
            0,
            0,
            2,
            vec![
                (StreamIndex::Out(0), StorageRequirement::Finite(1)),
                (StreamIndex::In(0), StorageRequirement::Finite(1)),
            ],
        )
    }

    #[test]
    fn a_simple_future_dependence() {
        check_buffer_size(
            "output a := b[+1] input b: Int8",
            0,
            0,
            2,
            vec![
                (StreamIndex::Out(0), StorageRequirement::FutureRef(1)),
                (StreamIndex::In(0), StorageRequirement::Finite(1)),
            ],
        )
    }

    #[test]
    #[ignore] // fix spec
    fn tracking_the_future() {
        check_tracking_size(
            "output a :Int8 := b[+1s] input b: Int8",
            0,
            0,
            2,
            vec![
                (StreamIndex::Out(0), vec![(StreamIndex::In(0), TrackingRequirement::Future)]),
                (StreamIndex::In(0), vec![]),
            ],
        )
    }

    #[test]
    #[ignore] // fix spec
    fn tracking_the_event_based_past() {
        check_tracking_size(
            "output a :Int8 := b[-1s].defaults(to:0) input b: Int8",
            0,
            0,
            2,
            vec![
                (StreamIndex::Out(0), vec![(StreamIndex::In(0), TrackingRequirement::Unbounded)]),
                (StreamIndex::In(0), vec![]),
            ],
        )
    }

    #[test]
    #[ignore] // fix spec
    fn tracking_the_time_based_past() {
        check_tracking_size(
            "output a :Int8 := c[-1s].defaults(to:0) input b: Int8 output c: Int8 @ 1 s := b[0].defaults(to:0)",
            0,
            0,
            3,
            vec![
                (StreamIndex::Out(0), vec![(StreamIndex::Out(1), TrackingRequirement::Finite(1))]),
                (StreamIndex::In(0), vec![]),
                (StreamIndex::Out(1), vec![]),
            ],
        )
    }

    #[test]
    #[ignore] // fix spec
    fn time_based_tracking_the_past() {
        check_tracking_size(
            "output a :Int8 @ 1s := b[-2.5s].defaults(to:0) input b: Int8",
            0,
            0,
            2,
            vec![
                (StreamIndex::Out(0), vec![(StreamIndex::In(0), TrackingRequirement::Finite(3))]),
                (StreamIndex::In(0), vec![]),
            ],
        )
    }
}
