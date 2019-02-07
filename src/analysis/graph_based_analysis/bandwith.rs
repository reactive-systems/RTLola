use crate::analysis::graph_based_analysis::get_ast_id;
use crate::analysis::graph_based_analysis::get_byte_size;
use crate::analysis::graph_based_analysis::DependencyGraph;
use crate::analysis::graph_based_analysis::MemoryBound;
use crate::analysis::graph_based_analysis::NIx;
use crate::analysis::graph_based_analysis::StreamNode::*;
use crate::analysis::TypeTable;
use crate::ast::LolaSpec;
use crate::parse::NodeId;
use crate::ty::TimingInfo;
use num::{BigInt, BigRational, FromPrimitive, Integer, Zero};
use petgraph::visit::IntoNodeReferences;

#[derive(Clone, Debug)]
pub(crate) struct Cut(Vec<NodeId>);

impl Cut {
    pub(crate) fn print(&self, spec: &LolaSpec) {
        self.0.iter().for_each(|id| {
            let displayed_name = spec
                .outputs
                .iter()
                .find(|output| output.id == *id)
                .map(|output| output.name.name.clone())
                .unwrap_or_else(|| {
                    spec.trigger
                        .iter()
                        .find(|trigger| trigger.id == *id)
                        .map(|trigger| {
                            trigger
                                .name
                                .clone()
                                .map(|ident| {
                                    assert!(!ident.name.is_empty());
                                    ident.name
                                })
                                .unwrap_or_else(|| {
                                    trigger
                                        .message
                                        .clone()
                                        .map(|message| format!("trigger with message: {}", message))
                                        .unwrap_or_else(|| {
                                            String::from("unnamed trigger without message")
                                        })
                                })
                        })
                        .expect("We only have outputs and triggers in the cut!")
                });
            println!("  {}", displayed_name);
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) enum BytesPerSecond {
    Bounded(BigRational),
    Unbounded,
}

pub(crate) type Bandwith = Vec<(Cut, BytesPerSecond)>;

pub(crate) fn determine_bandwith_requirements(
    type_table: &TypeTable,
    dependency_graph: &DependencyGraph,
) -> Bandwith {
    let mut result: Bandwith = Vec::new();

    let non_input_stream_ids: Vec<NodeId> = dependency_graph
        .node_references()
        .filter_map(|(_node_index, node)| match node {
            ClassicInput(_id) => None,
            ClassicOutput(id) => Some(*id),
            ParameterizedInput(_id) => None,
            ParameterizedOutput(_id) => unimplemented!(),
            RTOutput(id) => Some(*id),
            Trigger(id) => Some(*id),
            RTTrigger(id) => Some(*id),
        })
        .collect();

    let mut cut: Cut = Cut(Vec::new());

    bandwith_helper(
        &mut result,
        &mut cut,
        0,
        &non_input_stream_ids,
        dependency_graph,
        type_table,
    );

    result
}

fn bandwith_helper(
    result: &mut Bandwith,
    cut: &mut Cut,
    current_index: usize,
    non_input_stream_ids: &[NodeId],
    dependency_graph: &DependencyGraph,
    type_table: &TypeTable,
) {
    if current_index < non_input_stream_ids.len() {
        // recurse
        cut.0.push(non_input_stream_ids[current_index]);
        bandwith_helper(
            result,
            cut,
            current_index + 1,
            non_input_stream_ids,
            dependency_graph,
            type_table,
        );
        cut.0.pop();
        bandwith_helper(
            result,
            cut,
            current_index + 1,
            non_input_stream_ids,
            dependency_graph,
            type_table,
        );
    } else {
        if cut.0.is_empty() {
            return;
        }
        // check
        let valid_cut = cut.0.iter().all(|id| {
            // get node index
            let (index, _node) = dependency_graph
                .node_references()
                .find(|(_node_index, node)| *id == get_ast_id(**node))
                .unwrap();
            // check that all dependent streams are also in the cut
            dependency_graph.neighbors(index).all(|other| {
                let other_id = get_ast_id(*dependency_graph.node_weight(other).unwrap());
                cut.0.iter().any(|entry| *entry == other_id)
            })
        });

        if valid_cut {
            let streams_on_device: Vec<(NIx, NodeId)> = dependency_graph
                .node_references()
                .filter_map(|(node_index, node)| match node {
                    ClassicInput(id) => Some((node_index, *id)),
                    ClassicOutput(id) => Some((node_index, *id)),
                    ParameterizedInput(id) => Some((node_index, *id)),
                    ParameterizedOutput(_id) => unimplemented!(),
                    RTOutput(id) => Some((node_index, *id)),
                    Trigger(id) => Some((node_index, *id)),
                    RTTrigger(id) => Some((node_index, *id)),
                })
                .filter(|(_, node_id)| !cut.0.contains(node_id))
                .collect();
            let zero_bandwith: BytesPerSecond = BytesPerSecond::Bounded(BigRational::zero());

            let required_bandwith: BytesPerSecond = streams_on_device
                .iter()
                .map(|(node_index, node_id)| {
                    let type_size = get_byte_size(type_table.get_value_type(*node_id));
                    let type_size = match type_size {
                        MemoryBound::Unbounded => return BytesPerSecond::Unbounded,
                        MemoryBound::Bounded(s) => s,
                    };
                    match type_table.get_stream_type(*node_id).timing {
                        TimingInfo::Event => {
                            // check that none of my neighbors is in the cut
                            let some_dependent_streans_is_in_the_cut =
                                dependency_graph.neighbors(*node_index).any(|other_index| {
                                    let other_id = get_ast_id(
                                        *dependency_graph.node_weight(other_index).unwrap(),
                                    );
                                    cut.0.contains(&other_id)
                                });
                            if some_dependent_streans_is_in_the_cut {
                                // unknown amount of events
                                BytesPerSecond::Unbounded
                            } else {
                                // we do not need to send this
                                BytesPerSecond::Bounded(BigRational::zero())
                            }
                        }
                        TimingInfo::RealTime(_) => {
                            let own_frequency: BigRational =
                                match &type_table.get_stream_type(*node_id).timing {
                                    TimingInfo::Event => unreachable!(),
                                    TimingInfo::RealTime(freq) => {
                                        freq.ns.clone().recip()
                                            * BigRational::from_integer(
                                                BigInt::from_u64(1_000_000_000).unwrap(),
                                            )
                                    }
                                };
                            let mut frequencies: Vec<BigRational> = Vec::new();
                            dependency_graph
                                .neighbors(*node_index)
                                .for_each(|other_index| {
                                    let other_id = get_ast_id(
                                        *dependency_graph.node_weight(other_index).unwrap(),
                                    );
                                    match &type_table.get_stream_type(other_id).timing {
                                        TimingInfo::Event => {
                                            frequencies.push(own_frequency.clone())
                                        } // we need to push all values
                                        TimingInfo::RealTime(freq) => {
                                            let other_freq = freq.ns.clone().recip()
                                                * BigRational::from_integer(
                                                    BigInt::from_u64(1_000_000_000).unwrap(),
                                                );
                                            if own_frequency < other_freq {
                                                frequencies.push(own_frequency.clone())
                                            } else {
                                                frequencies.push(other_freq) // we can push less
                                            }
                                        }
                                    }
                                });
                            // LCM of fractions = LCM of numerators/ gcd  of denominators
                            // min(own,lcm(frequencies))
                            BytesPerSecond::Bounded(
                                if frequencies.is_empty() {
                                    own_frequency
                                } else {
                                    let lcm_numer = frequencies
                                        .iter()
                                        .map(|frac| frac.numer().clone())
                                        .fold(frequencies[0].numer().clone(), |left, right| {
                                            left.lcm(&right)
                                        });
                                    let gcd_denom = frequencies
                                        .iter()
                                        .map(|frac| frac.denom().clone())
                                        .fold(frequencies[0].denom().clone(), |left, right| {
                                            left.gcd(&right)
                                        });
                                    let lcm_frequencies = BigRational::from((lcm_numer, gcd_denom));
                                    if own_frequency < lcm_frequencies {
                                        own_frequency
                                    } else {
                                        lcm_frequencies
                                    }
                                } * BigRational::from_integer(
                                    BigInt::from_u128(type_size).unwrap(),
                                ),
                            )
                        }
                    }
                })
                .fold(zero_bandwith, |left, right| match (left, right) {
                    (BytesPerSecond::Bounded(l), BytesPerSecond::Bounded(r)) => {
                        BytesPerSecond::Bounded(l + r)
                    }
                    _ => BytesPerSecond::Unbounded,
                });
            result.push((cut.clone(), required_bandwith))
        }
    }
}
