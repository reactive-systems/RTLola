use crate::analysis::graph_based_analysis::{StreamNode::*, *};
use crate::parse::NodeId;
use crate::ty::StreamTy;
use std::collections::HashMap;

pub(crate) type EvalOrder = Vec<Vec<ComputeStep>>;

#[derive(Debug)]
pub(crate) struct EvaluationOrderResult {
    pub(crate) event_based_streams_order: EvalOrder,
    pub(crate) periodic_streams_order: EvalOrder,
}

pub(crate) fn determine_evaluation_order(
    dependency_graph: DependencyGraph,
) -> (EvaluationOrderResult, DependencyGraph) {
    // TODO prune the graph

    let (compute_graph_unperiodic, compute_graph_periodic) = build_compute_graphs(dependency_graph.clone());
    //    println!("{:?}", petgraph::dot::Dot::new(&compute_graph_unperiodic));
    //    println!("{:?}", petgraph::dot::Dot::new(&compute_graph_periodic));
    let event_based_streams_order = get_compute_order(compute_graph_unperiodic);
    let periodic_streams_order = get_compute_order(compute_graph_periodic);

    (EvaluationOrderResult { event_based_streams_order, periodic_streams_order }, dependency_graph)
}

fn build_compute_graphs(dependency_graph: DependencyGraph) -> (ComputationGraph, ComputationGraph) {
    let mut unperiodic_streams_graph = dependency_graph.clone();
    unperiodic_streams_graph.retain_nodes(|g, node_index| {
        match g.node_weight(node_index).expect("Existence guaranteed by the library") {
            RTTrigger(_, StreamTy::RealTime(_)) | RTOutput(_, StreamTy::RealTime(_)) => false,
            ClassicInput(_)
            | RTTrigger(_, StreamTy::Event(_))
            | RTOutput(_, StreamTy::Event(_))
            | RTTrigger(_, StreamTy::Infer(_))
            | RTOutput(_, StreamTy::Infer(_)) => true,
        }
    });
    let mut periodic_streams_graph = dependency_graph;
    periodic_streams_graph.retain_nodes(|g, node_index| {
        match g.node_weight(node_index).expect("Existence guaranteed by the library") {
            RTTrigger(_, StreamTy::RealTime(_))
            | RTOutput(_, StreamTy::RealTime(_))
            | RTTrigger(_, StreamTy::Infer(_))
            | RTOutput(_, StreamTy::Infer(_)) => true,
            RTTrigger(_, StreamTy::Event(_)) | RTOutput(_, StreamTy::Event(_)) | ClassicInput(_) => false,
        }
    });

    (build_compute_graph(unperiodic_streams_graph), build_compute_graph(periodic_streams_graph))
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
    let mut computation_graph = ComputationGraph::with_capacity(expected_capacity, expected_capacity);
    for node in dependency_graph.node_weights_mut() {
        // TODO ignore real time streams
        let id = get_ast_id(node);
        let invoke: NIx = computation_graph.add_node(ComputeStep::Invoke(id));
        let extend: NIx = computation_graph.add_node(ComputeStep::Extend(id));
        let evaluate: NIx = computation_graph.add_node(ComputeStep::Evaluate(id));
        let terminate: NIx = computation_graph.add_node(ComputeStep::Terminate(id));
        computation_graph.add_edge(extend, invoke, ());
        computation_graph.add_edge(terminate, invoke, ());
        computation_graph.add_edge(evaluate, extend, ());
        mapping.insert(id, StepEntry { invoke, extend, evaluate, terminate });
    }

    for edge_idx in dependency_graph.edge_indices() {
        let edge_weight = dependency_graph.edge_weight(edge_idx).expect("We iterate over all existing EdgeIndices");
        let (source, target) =
            dependency_graph.edge_endpoints(edge_idx).expect("We iterate over all existing EdgeIndices");
        let source_id =
            get_ast_id(dependency_graph.node_weight(source).expect("We just git this NodeIndex from the graph"));
        let target_id =
            get_ast_id(dependency_graph.node_weight(target).expect("We just git this NodeIndex from the graph"));

        match edge_weight {
            StreamDependency::InvokeByName(_) => {
                // we need to know the values so we need evaluate
                computation_graph.add_edge(mapping[&source_id].invoke, mapping[&target_id].evaluate, ());
            }
            StreamDependency::Access(location, offset, _) => {
                match offset {
                    Offset::Discrete(value) => {
                        use std::cmp::Ordering;
                        match value.cmp(&0) {
                            Ordering::Equal => {
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
                            Ordering::Less => {
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
                            }
                            Ordering::Greater => {} // no need to add dependency for positive edges
                        }
                    }
                    Offset::Time(offset) => {
                        if let TimeOffset::UpToNow(_) = offset {
                            computation_graph.add_edge(mapping[&source_id].evaluate, mapping[&target_id].evaluate, ());
                        }
                    }
                    Offset::SlidingWindow => {
                        if source_id != target_id {
                            computation_graph.add_edge(mapping[&source_id].evaluate, mapping[&target_id].evaluate, ());
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
                    .push(*compute_graph.node_weight(index).expect("The indices are provided by the graph."));
            }
        }

        // remove the current front
        for index in pruned_nodes.iter().rev() {
            compute_graph.remove_node(*index);
        }
    }

    // now prune empty steps ones from the back
    while !compute_step_order.is_empty()
        && compute_step_order.last().expect("We just checked that the vector is not empty.").is_empty()
    {
        compute_step_order.pop();
    }
    compute_step_order
}

#[cfg(test)]
mod tests {
    use crate::analysis::graph_based_analysis::dependency_graph::analyse_dependencies;
    use crate::analysis::graph_based_analysis::evaluation_order::determine_evaluation_order;
    use crate::analysis::graph_based_analysis::ComputeStep;
    use crate::analysis::naming::NamingAnalysis;
    use crate::parse::parse;
    use crate::parse::NodeId;
    use crate::parse::SourceMapper;
    use crate::reporting::Handler;
    use crate::ty::check::TypeAnalysis;
    use crate::FrontendConfig;
    use std::path::PathBuf;
    use StreamIndex::{In, Out};

    #[derive(Debug, Clone, Copy)]
    enum StreamIndex {
        Out(usize),
        In(usize),
    }

    fn check_eval_order(
        content: &str,
        expected_errors: usize,
        expected_warning: usize,
        expected_event_based_streams_order: Vec<Vec<(StreamIndex, ComputeStep)>>,
    ) {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let ast = parse(content, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        let mut naming_analyzer = NamingAnalysis::new(&handler, FrontendConfig::default());
        let mut decl_table = naming_analyzer.check(&ast);
        let mut type_analysis = TypeAnalysis::new(&handler, &mut decl_table);
        let type_table = type_analysis.check(&ast);
        let type_table = type_table.as_ref().expect("We expect in these tests that the type analysis checks out.");
        let dependency_analysis = analyse_dependencies(&ast, &decl_table, &handler, &type_table);

        let (order, _) = determine_evaluation_order(dependency_analysis.dependency_graph);
        assert_eq!(expected_errors, handler.emitted_errors());
        assert_eq!(expected_warning, handler.emitted_warnings());
        assert_eq!(expected_event_based_streams_order.len(), order.event_based_streams_order.len());
        let mut level_iter = order.event_based_streams_order.iter();
        for level in expected_event_based_streams_order {
            let result_level = level_iter.next().expect("we checked the length");
            assert_eq!(level.len(), result_level.len());
            for (index, step) in level {
                let node_id = match index {
                    Out(i) => ast.outputs[i].id,
                    In(i) => ast.inputs[i].id,
                };
                let expected_step = match step {
                    ComputeStep::Evaluate(_) => ComputeStep::Evaluate(node_id),
                    ComputeStep::Invoke(_) => ComputeStep::Invoke(node_id),
                    ComputeStep::Terminate(_) => ComputeStep::Terminate(node_id),
                    ComputeStep::Extend(_) => ComputeStep::Extend(node_id),
                };
                assert!(result_level.contains(&expected_step));
            }
        }
    }

    #[test]
    fn simple_spec() {
        check_eval_order(
            "input a: UInt8
output b: UInt8 := a",
            0,
            0,
            vec![
                vec![(In(0), ComputeStep::Invoke(NodeId::new(0))), (Out(0), ComputeStep::Invoke(NodeId::new(0)))],
                vec![
                    (In(0), ComputeStep::Extend(NodeId::new(0))),
                    (In(0), ComputeStep::Terminate(NodeId::new(0))),
                    (Out(0), ComputeStep::Extend(NodeId::new(0))),
                    (Out(0), ComputeStep::Terminate(NodeId::new(0))),
                ],
                vec![(In(0), ComputeStep::Evaluate(NodeId::new(0)))],
                vec![(Out(0), ComputeStep::Evaluate(NodeId::new(0)))],
            ],
        )
    }
}
