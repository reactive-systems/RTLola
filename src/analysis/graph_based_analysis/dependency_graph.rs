use crate::analysis::graph_based_analysis::DependencyGraph;
use crate::analysis::graph_based_analysis::EIx;
use crate::analysis::graph_based_analysis::Location;
use crate::analysis::graph_based_analysis::NIx;
use crate::analysis::graph_based_analysis::Offset;
use crate::analysis::graph_based_analysis::StreamDependency;
use crate::analysis::graph_based_analysis::StreamNode;
use crate::analysis::graph_based_analysis::StreamNode::*;
use crate::analysis::lola_version::LolaVersionTable;
use crate::analysis::naming::Declaration;
use crate::analysis::naming::DeclarationTable;
use crate::ast::ExpressionKind;
use crate::ast::LanguageSpec;
use crate::ast::LitKind;
use crate::ast::LolaSpec;
use crate::ast::Output;
use crate::ast::TemplateSpec;
use crate::ast::UnOp;
use crate::reporting::DiagnosticBuilder;
use crate::reporting::Handler;
use crate::reporting::LabeledSpan;
use crate::reporting::Level;
use ast_node::AstNode;
use ast_node::NodeId;
use ast_node::Span;
use petgraph::algo::tarjan_scc;
use petgraph::graph::edge_index;
use petgraph::graph::node_index;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::visit::NodeIndexable;
use std::collections::HashMap;

#[derive(Debug, Copy, Clone)]
pub(crate) struct StreamMappingInfo {
    normal_time_index: NIx,
}

pub(crate) type StreamMapping = HashMap<ast_node::NodeId, StreamMappingInfo>;

#[derive(Copy, Clone)]
enum CycleWeight {
    Positive,
    Zero,
    Negative,
}

struct CycleTracker {
    pub positive: Vec<usize>,
    pub negative: Vec<usize>,
}

impl Default for CycleTracker {
    fn default() -> Self {
        CycleTracker {
            positive: Vec::new(),
            negative: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct DependencyAnalyser<'a> {
    dependency_graph: DependencyGraph,
    spec: &'a LolaSpec,
    version_table: &'a LolaVersionTable,
    naming_table: &'a DeclarationTable<'a>,
    handler: &'a Handler,
    stream_names: HashMap<NodeId, String>,
}

// TODO add fields
#[derive(Debug)]
pub(crate) struct DependencyAnalysis {
    pub dependency_graph: DependencyGraph,
    pub nodes_with_positive_cycle: Vec<NodeId>,
}

struct CycleFinder<'a> {
    dependency_graph: &'a DependencyGraph,
    blocked: Vec<bool>,
    b_lists: Vec<Vec<NIx>>,
    stack: Vec<NIx>,
    cycles: Vec<Vec<NIx>>,
}

impl<'a> CycleFinder<'a> {
    fn new(dependency_graph: &'a DependencyGraph) -> CycleFinder<'a> {
        let number_of_nodes = dependency_graph.node_count();
        let mut blocked: Vec<bool> = Vec::with_capacity(number_of_nodes);
        blocked.resize(number_of_nodes, false);
        let mut b_lists: Vec<Vec<NIx>> = Vec::with_capacity(number_of_nodes);
        b_lists.resize(number_of_nodes, Vec::new());
        let stack: Vec<NIx> = Vec::new();
        let cycles: Vec<Vec<NIx>> = Vec::new();
        CycleFinder {
            dependency_graph,
            blocked,
            b_lists,
            stack,
            cycles,
        }
    }

    fn get_elementary_cycles(mut self) -> Vec<Vec<NIx>> {
        let strongly_connected_components: Vec<Vec<NIx>> = tarjan_scc(self.dependency_graph)
            .iter()
            .map(|component| {
                component
                    .iter()
                    .map(|id| node_index(self.dependency_graph.to_index(*id)))
                    .collect()
            })
            .collect();
        for component in strongly_connected_components.iter() {
            let starting_node: NIx = *component.iter().min().unwrap();
            self.get_elementary_circuits_for_component(starting_node, starting_node, component);
        }
        // check for self-loops
        for node in self.dependency_graph.node_indices() {
            for edge in self.dependency_graph.edges(node) {
                if edge.target() == node {
                    let self_loop = vec![node];
                    self.cycles.push(self_loop);
                    break;
                }
            }
        }
        self.cycles
    }

    fn unblock(&mut self, node: NIx) {
        self.blocked[node.index()] = false;
        while !self.b_lists[node.index()].is_empty() {
            let w = self.b_lists[node.index()].pop().unwrap();
            if self.blocked[w.index()] {
                self.unblock(w);
            }
        }
    }

    fn get_elementary_circuits_for_component(
        &mut self,
        node: NIx,
        starting_node: NIx,
        nodes_in_scc: &[NIx],
    ) -> bool {
        let mut found = false;
        self.stack.push(node);
        self.blocked[node.index()] = true;
        let mut neighbors: Vec<NodeIndex> = self.dependency_graph.neighbors(node).collect();
        neighbors.sort();
        neighbors.dedup();

        for w in neighbors.iter() {
            //check for self-loops -- we will add them later.
            if *w != node && nodes_in_scc.contains(w) {
                if *w == starting_node {
                    // we found a cycle
                    self.cycles.push(self.stack.clone());
                    found = true;
                } else if !self.blocked[w.index()]
                    && self.get_elementary_circuits_for_component(*w, starting_node, nodes_in_scc)
                {
                    found = true;
                }
            }
        }

        if found {
            self.unblock(node);
        } else {
            for w in neighbors {
                if w != node
                    && nodes_in_scc.contains(&w)
                    && !self.b_lists[w.index()].contains(&node)
                {
                    self.b_lists[w.index()].push(node);
                }
            }
        }

        self.stack.pop();
        found
    }
}

fn find_elementary_cycles(graph: &DependencyGraph) -> Vec<Vec<NodeIndex<u32>>> {
    let cycle_finder = CycleFinder::new(graph);
    cycle_finder.get_elementary_cycles()
}

fn enumerate_paths(
    dependency_graph: &DependencyGraph,
    cycle: &[NodeIndex],
    remaining_elements: usize,
    paths: &mut Vec<Vec<petgraph::prelude::EdgeIndex>>,
    stack: &mut Vec<petgraph::prelude::EdgeIndex>,
    start_node: NodeIndex,
) {
    if remaining_elements == 0 {
        let mut new_path_reversed = stack.clone();
        new_path_reversed.reverse();
        paths.push(new_path_reversed);
        return;
    }
    let index = remaining_elements - 1;

    let target = if cycle.len() == remaining_elements {
        start_node
    } else {
        cycle[remaining_elements]
    };
    let node_index = cycle[index];
    for edge in dependency_graph.edges(node_index) {
        if edge.target() == target {
            stack.push(edge_index(edge.id().index()));
            enumerate_paths(dependency_graph, cycle, index, paths, stack, start_node);
            stack.pop();
        }
    }
}

impl<'a> DependencyAnalyser<'a> {
    fn add_trigger_nodes(&mut self, mapping: &mut HashMap<ast_node::NodeId, StreamMappingInfo>) {
        // for each trigger add one node
        for trigger in &self.spec.trigger {
            let id = *trigger.id();
            match self.version_table[&id] {
                LanguageSpec::RTLola => {
                    let normal_time_index = self.dependency_graph.add_node(RTTrigger(id));
                    mapping.insert(id, StreamMappingInfo { normal_time_index });
                }
                LanguageSpec::Classic | LanguageSpec::Lola2 => {
                    let normal_time_index = self.dependency_graph.add_node(Trigger(id));
                    mapping.insert(id, StreamMappingInfo { normal_time_index });
                }
            }
        }
    }

    fn add_output_nodes(&mut self, mapping: &mut HashMap<ast_node::NodeId, StreamMappingInfo>) {
        // for each output add a node
        for output in &self.spec.outputs {
            let id = *output.id();
            self.stream_names.insert(id, output.name.name.clone());
            match self.version_table[&id] {
                LanguageSpec::Classic => {
                    let normal_time_index = self.dependency_graph.add_node(ClassicOutput(id));
                    mapping.insert(id, StreamMappingInfo { normal_time_index });
                }
                LanguageSpec::Lola2 => {
                    let normal_time_index = self.dependency_graph.add_node(ParameterizedOutput(id));
                    mapping.insert(id, StreamMappingInfo { normal_time_index });
                }
                LanguageSpec::RTLola => {
                    let normal_time_index = self.dependency_graph.add_node(RTOutput(id));
                    mapping.insert(id, StreamMappingInfo { normal_time_index });
                }
            }
        }
    }

    fn add_input_nodes(&mut self, mapping: &mut HashMap<ast_node::NodeId, StreamMappingInfo>) {
        // for each input add a node
        for input in &self.spec.inputs {
            let id = *input.id();
            self.stream_names.insert(id, input.name.name.clone());
            match self.version_table[&id] {
                LanguageSpec::Classic => {
                    let normal_time_index = self.dependency_graph.add_node(ClassicInput(id));
                    mapping.insert(id, StreamMappingInfo { normal_time_index });
                }
                LanguageSpec::Lola2 => {
                    let normal_time_index = self.dependency_graph.add_node(ParameterizedInput(id));
                    mapping.insert(id, StreamMappingInfo { normal_time_index });
                }
                LanguageSpec::RTLola => unreachable!(),
            }
        }
    }

    fn evaluate_discrete(
        &self,
        expr: &crate::ast::Expression,
        naming_table: &'a DeclarationTable,
    ) -> i32 {
        // TODO need typing information
        match &expr.kind {
            ExpressionKind::Lit(lit) => match &lit.kind {
                LitKind::Int(int) => *int as i32,
                _ => unimplemented!(),
            },
            ExpressionKind::Ident(_) => match &naming_table[&expr.id()] {
                _ => unimplemented!(),
            },
            ExpressionKind::Default(_, _) => unreachable!(),
            ExpressionKind::Lookup(_, _, _) => unreachable!(),
            ExpressionKind::Binary(_, _, _) => unimplemented!(),
            ExpressionKind::Unary(op, expr) => {
                assert_eq!(UnOp::Neg, *op);
                -self.evaluate_discrete(expr, naming_table)
            }
            ExpressionKind::Ite(_, _, _) => unreachable!(),
            ExpressionKind::ParenthesizedExpression(_, expr, _) => {
                self.evaluate_discrete(expr, naming_table)
            }
            ExpressionKind::MissingExpression() => unreachable!(),
            ExpressionKind::Tuple(_) => unimplemented!(),
            ExpressionKind::Function(_, _, _) => unimplemented!(),
            ExpressionKind::Field(_, _) => unimplemented!(),
            ExpressionKind::Field(_, _) => unimplemented!(),
            ExpressionKind::Method(_, _, _, _) => unimplemented!(),
        }
    }

    fn evaluate_float(
        &self,
        expr: &crate::ast::Expression,
        naming_table: &'a DeclarationTable,
    ) -> f64 {
        // TODO need typing information
        match &expr.kind {
            ExpressionKind::Lit(lit) => match &lit.kind {
                LitKind::Int(int) => *int as f64,
                LitKind::Float(float) => *float as f64,
                _ => unimplemented!(),
            },
            ExpressionKind::Ident(_) => match &naming_table[&expr.id()] {
                _ => unimplemented!(),
            },
            ExpressionKind::Default(_, _) => unreachable!(),
            ExpressionKind::Lookup(_, _, _) => unreachable!(),
            ExpressionKind::Binary(_, _, _) => unimplemented!(),
            ExpressionKind::Unary(op, expr) => {
                assert_eq!(UnOp::Neg, *op);
                -self.evaluate_float(expr, naming_table)
            }
            ExpressionKind::Ite(_, _, _) => unreachable!(),
            ExpressionKind::ParenthesizedExpression(_, expr, _) => {
                self.evaluate_float(expr, naming_table)
            }
            ExpressionKind::MissingExpression() => unreachable!(),
            ExpressionKind::Tuple(_) => unimplemented!(),
            ExpressionKind::Function(_, _, _) => unimplemented!(),
            ExpressionKind::Field(_, _) => unimplemented!(),
            ExpressionKind::Field(_, _) => unimplemented!(),
            ExpressionKind::Method(_, _, _, _) => unimplemented!(),
        }
    }

    fn translate_offset(
        &self,
        offset: &crate::ast::Offset,
        naming_table: &'a DeclarationTable,
    ) -> Offset {
        match offset {
            crate::ast::Offset::DiscreteOffset(expr) => {
                let offset = self.evaluate_discrete(expr, naming_table);
                Offset::Discrete(offset)
            }
            crate::ast::Offset::RealTimeOffset(expr, unit) => {
                let offset = self.evaluate_float(expr, naming_table);
                Offset::Time(offset, *unit)
            }
        }
    }

    fn add_edges_for_expression(
        &mut self,
        current_node: NodeIndex<u32>,
        expr: &crate::ast::Expression,
        location: Location,
        mapping: &mut StreamMapping,
    ) {
        match &expr.kind {
            ExpressionKind::Tuple(elements) | ExpressionKind::Function(_, _, elements) => {
                elements.iter().for_each(|element| {
                    self.add_edges_for_expression(current_node, &*element, location, mapping)
                });
            }
            ExpressionKind::MissingExpression() | ExpressionKind::Lit(_) => {}
            ExpressionKind::Default(stream, default) => {
                self.add_edges_for_expression(current_node, &*stream, location, mapping);
                self.add_edges_for_expression(current_node, &*default, location, mapping);
            }
            ExpressionKind::ParenthesizedExpression(_, expr, _)
            | ExpressionKind::Unary(_, expr) => {
                self.add_edges_for_expression(current_node, &*expr, location, mapping);
            }
            ExpressionKind::Ite(cond, if_case, else_case) => {
                self.add_edges_for_expression(current_node, &*cond, location, mapping);
                self.add_edges_for_expression(current_node, &*if_case, location, mapping);
                self.add_edges_for_expression(current_node, &*else_case, location, mapping);
            }
            ExpressionKind::Binary(_, left, right) => {
                self.add_edges_for_expression(current_node, &*left, location, mapping);
                self.add_edges_for_expression(current_node, &*right, location, mapping);
            }
            ExpressionKind::Ident(_) => match &self.naming_table[&expr.id()] {
                Declaration::Type(_)
                | Declaration::Func(_)
                | Declaration::Param(_)
                | Declaration::Const(_) => {}
                Declaration::In(input) => {
                    let target_stream_id = input.id();
                    let target_stream_entry = mapping[&target_stream_id];
                    let target_stream_index = target_stream_entry.normal_time_index;
                    self.dependency_graph.add_edge(
                        current_node,
                        target_stream_index,
                        StreamDependency::Access(location, Offset::Discrete(0), *expr.span()),
                    );
                }
                Declaration::Out(output) => {
                    let target_stream_id = output.id();
                    let target_stream_entry = mapping[&target_stream_id];
                    let target_stream_index = target_stream_entry.normal_time_index;
                    self.dependency_graph.add_edge(
                        current_node,
                        target_stream_index,
                        StreamDependency::Access(location, Offset::Discrete(0), *expr.span()),
                    );
                }
            },
            ExpressionKind::Lookup(instance, offset, _) => {
                let target_stream_id = match &self.naming_table[instance.id()] {
                    Declaration::Out(output) => output.id(),
                    Declaration::In(input) => input.id(),
                    _ => unreachable!(),
                };
                let target_stream_entry = mapping[&target_stream_id];
                let target_stream_index = target_stream_entry.normal_time_index;
                let offset = self.translate_offset(offset, self.naming_table);
                self.dependency_graph.add_edge(
                    current_node,
                    target_stream_index,
                    StreamDependency::Access(location, offset, *expr.span()),
                );
            }
            ExpressionKind::Field(_, _) => unimplemented!(),
            ExpressionKind::Method(_, _, _, _) => unimplemented!(),
        }
    }

    fn get_cycle_weight(&self, cycle_path: &[EIx]) -> Option<CycleWeight> {
        let any_rt =
            cycle_path.iter().any(
                |&edge| match self.dependency_graph.edge_weight(edge).unwrap() {
                    StreamDependency::Access(_, offset, _) => match offset {
                        Offset::Time(_, ..) => true,
                        Offset::Discrete(_) => false,
                    },
                    _ => false,
                },
            );

        if any_rt {
            unimplemented!()
        }

        let mut total_weight = 0;

        for edge_index in cycle_path {
            let edge_info = self.dependency_graph.edge_weight(*edge_index).unwrap();
            match edge_info {
                StreamDependency::InvokeByName(_) => {}
                StreamDependency::Access(_, offset, _) => match offset {
                    Offset::Time(_, ..) => unreachable!("This is a cycle without realtime"),
                    Offset::Discrete(offset) => total_weight += offset,
                },
            }
        }

        match total_weight {
            num if num < 0 => Some(CycleWeight::Negative),
            0 => Some(CycleWeight::Zero),
            num if num > 0 => Some(CycleWeight::Positive),
            _ => unreachable!(),
        }
    }

    fn find_all_cyclic_paths(
        &mut self,
        cycles: &[Vec<NodeIndex>],
    ) -> Vec<Vec<petgraph::prelude::EdgeIndex>> {
        let mut paths: Vec<Vec<petgraph::prelude::EdgeIndex>> = Vec::with_capacity(cycles.len());
        let mut path: Vec<petgraph::prelude::EdgeIndex> = Vec::new();
        for cycle in cycles.iter() {
            println!("Transforming cycle {:?}", cycle);
            enumerate_paths(
                &self.dependency_graph,
                cycle,
                cycle.len(),
                &mut paths,
                &mut path,
                cycle[0],
            );
        }
        paths
    }

    fn add_output_dependencies(
        &mut self,
        mut mapping: &mut HashMap<ast_node::NodeId, StreamMappingInfo>,
    ) {
        for output in &self.spec.outputs {
            let id = *output.id();
            let current_node = mapping[&id].normal_time_index;
            self.handle_expression(&mut mapping, &output, id, current_node);

            match self.version_table[&id] {
                LanguageSpec::Classic => {}
                LanguageSpec::Lola2 | LanguageSpec::RTLola => {
                    if let Some(ref template_spec) = output.template_spec {
                        self.handle_invoke(&mut mapping, current_node, template_spec);
                        self.handle_extend(&mut mapping, current_node, template_spec);
                        self.handle_terminate(&mut mapping, current_node, template_spec);
                    }
                }
            }
        }
    }

    fn handle_invoke(
        &mut self,
        mut mapping: &mut HashMap<ast_node::NodeId, StreamMappingInfo>,
        current_node: NodeIndex<u32>,
        template_spec: &TemplateSpec,
    ) {
        if let Some(ref invoke_spec) = template_spec.inv {
            let mut by_name = true;
            // try by_name
            if let ExpressionKind::Ident(_) = invoke_spec.target.kind {
                let target_id = match &self.naming_table[invoke_spec.target.id()] {
                    Declaration::In(input) => *input.id(),
                    Declaration::Out(output) => *output.id(),
                    _ => {
                        by_name = false;
                        NodeId::new(0)
                    }
                };
                if by_name {
                    let accessed_node = mapping[&target_id].normal_time_index;
                    self.dependency_graph.add_edge(
                        current_node,
                        accessed_node,
                        StreamDependency::InvokeByName(*invoke_spec.target.span()),
                    );
                }
            }
            // otherwise it is an expression
            if !by_name {
                self.add_edges_for_expression(
                    current_node,
                    &invoke_spec.target,
                    Location::Invoke,
                    &mut mapping,
                );
            }

            // check the unless/if
            if let Some(ref cond) = invoke_spec.condition {
                self.add_edges_for_expression(current_node, &cond, Location::Invoke, &mut mapping);
            }
        }
    }

    fn handle_extend(
        &mut self,
        mut mapping: &mut HashMap<ast_node::NodeId, StreamMappingInfo>,
        current_node: NodeIndex<u32>,
        template_spec: &TemplateSpec,
    ) {
        if let Some(ref extend_spec) = template_spec.ext {
            if let Some(target) = &extend_spec.target {
                self.add_edges_for_expression(
                    current_node,
                    &target,
                    Location::Extend,
                    &mut mapping,
                );
            }
        }
    }

    fn handle_terminate(
        &mut self,
        mut mapping: &mut HashMap<ast_node::NodeId, StreamMappingInfo>,
        current_node: NodeIndex<u32>,
        template_spec: &TemplateSpec,
    ) {
        if let Some(ref terminate_spec) = template_spec.ter {
            self.add_edges_for_expression(
                current_node,
                &terminate_spec.target,
                Location::Terminate,
                &mut mapping,
            );
        }
    }

    fn handle_expression(
        &mut self,
        mut mapping: &mut HashMap<ast_node::NodeId, StreamMappingInfo>,
        output: &Output,
        id: ast_node::NodeId,
        current_node: NodeIndex<u32>,
    ) {
        self.add_edges_for_expression(
            current_node,
            &output.expression,
            Location::Expression,
            &mut mapping,
        );
    }

    fn add_trigger_dependencies(
        &mut self,
        mut mapping: &mut HashMap<ast_node::NodeId, StreamMappingInfo>,
    ) {
        for trigger in &self.spec.trigger {
            let id = *trigger.id();
            let current_node = mapping[&id].normal_time_index;
            self.add_edges_for_expression(
                current_node,
                &trigger.expression,
                Location::Expression,
                &mut mapping,
            );
        }
    }

    fn analyse_dependencies(mut self) -> DependencyAnalysis {
        let mut mapping: StreamMapping = StreamMapping::new();

        self.add_input_nodes(&mut mapping);
        self.add_output_nodes(&mut mapping);
        self.add_trigger_nodes(&mut mapping);

        // At this point we have all nodes.
        // Input streams are therefore already completely processed
        // Triggers still need their expression checked
        // TODO in the future we may have to add information about the extension
        self.add_trigger_dependencies(&mut mapping);

        // Outputs need their expressions and their template_spec/stream_pattern checked checked
        self.add_output_dependencies(&mut mapping);

        let nodes_with_positive_cycle = self.check_cycles();

        DependencyAnalysis {
            dependency_graph: self.dependency_graph,
            nodes_with_positive_cycle,
        }
    }

    fn check_cycles(&mut self) -> Vec<NodeId> {
        // find all cycles as a list of nodes
        let cycles: Vec<Vec<NodeIndex>> = find_elementary_cycles(&self.dependency_graph);
        // construct all paths -- this takes parallel edges into account
        let paths = self.find_all_cyclic_paths(&cycles);
        // TODO check each path individually
        // TODO check +/0/-
        // Map stream.index  -> (Vec<Vec<EIx>>,,)
        let mut cycles_per_stream: HashMap<NodeId, CycleTracker> =
            HashMap::with_capacity(self.spec.outputs.len());
        // add entry for each output
        for output in &self.spec.outputs {
            cycles_per_stream.insert(*output.id(), CycleTracker::default());
        }
        for (path_index, cyclic_path) in paths.iter().enumerate() {
            let weight = self.get_cycle_weight(&cyclic_path);
            if let Some(weight) = weight {
                match weight {
                    CycleWeight::Negative => self.add_path_index_for_all_nodes(
                        &mut cycles_per_stream,
                        path_index,
                        cyclic_path,
                        weight,
                    ),
                    CycleWeight::Zero => self.build_zero_weight_cycle_error(cyclic_path),
                    CycleWeight::Positive => self.add_path_index_for_all_nodes(
                        &mut cycles_per_stream,
                        path_index,
                        cyclic_path,
                        weight,
                    ),
                }
            }
        }
        let mut nodes_with_positive_cycle = Vec::new();
        for (id, cycle_tracker) in cycles_per_stream {
            if !cycle_tracker.negative.is_empty() && !cycle_tracker.positive.is_empty() {
                unimplemented!("We still need to give an error")
            }
            if !cycle_tracker.positive.is_empty() {
                nodes_with_positive_cycle.push(id);
                for positive_cycle_index in cycle_tracker.positive {
                    let positive_cycle = &paths[positive_cycle_index];
                    self.build_positive_weight_cycle_warning(positive_cycle)
                }
            }
        }
        nodes_with_positive_cycle
    }

    fn get_stream_name(&self, node_index: NIx) -> &String {
        let node = self.dependency_graph.node_weight(node_index).unwrap();
        let stream_id = match node {
            StreamNode::ClassicOutput(id)
            | StreamNode::ParameterizedOutput(id)
            | StreamNode::RTOutput(id) => id,
            StreamNode::ClassicInput(_)
            | StreamNode::ParameterizedInput(_)
            | StreamNode::Trigger(_)
            | StreamNode::RTTrigger(_) => {
                unreachable!("Inputs and triggers must never appear in a cycle.")
            }
        };

        &self.stream_names[stream_id]
    }

    fn build_zero_weight_cycle_error(&self, cyclic_path: &[EIx]) {
        let mut builder: Option<DiagnosticBuilder> = None;
        for edge_index in cyclic_path {
            let edge_weight = self.dependency_graph.edge_weight(*edge_index).unwrap();
            let (start_node, end_node) = self.dependency_graph.edge_endpoints(*edge_index).unwrap();
            let (span, label): (Span, String) = match edge_weight {
                StreamDependency::InvokeByName(span) => (
                    *span,
                    format!(
                        "The stream {} is invoked by stream {}",
                        self.get_stream_name(start_node),
                        self.get_stream_name(end_node)
                    ),
                ),
                StreamDependency::Access(location, _, span) => match location {
                    Location::Expression => (
                        *span,
                        format!(
                            "The stream {} accesses the stream {}",
                            self.get_stream_name(start_node),
                            self.get_stream_name(end_node)
                        ),
                    ),
                    Location::Invoke => (
                        *span,
                        format!(
                            "The stream {} accesses the stream {} in the invokation specification",
                            self.get_stream_name(start_node),
                            self.get_stream_name(end_node)
                        ),
                    ),
                    Location::Extend => (
                        *span,
                        format!(
                            "The stream {} accesses the stream {} in the extension expression",
                            self.get_stream_name(start_node),
                            self.get_stream_name(end_node)
                        ),
                    ),
                    Location::Terminate => (
                        *span,
                        format!(
                            "The stream {} accesses the stream {} in the termination expression",
                            self.get_stream_name(start_node),
                            self.get_stream_name(end_node)
                        ),
                    ),
                },
            };
            if let Some(ref mut builder) = builder {
                builder.add_span_with_label(span, label.as_str(), false);
            } else {
                let mut diagnostic_builder = self.handler.build_error_with_span(
                    &"There is a 0 weight cycle.".to_string(),
                    LabeledSpan::new(span, label.as_str(), false),
                );
                diagnostic_builder.prevent_sorting();
                builder = Some(diagnostic_builder);
            }
        }
        builder.unwrap().emit();
    }

    fn build_positive_weight_cycle_warning(&self, cyclic_path: &[EIx]) {
        let mut builder: Option<DiagnosticBuilder> = None;
        for edge_index in cyclic_path {
            let edge_weight = self.dependency_graph.edge_weight(*edge_index).unwrap();
            let (start_node, end_node) = self.dependency_graph.edge_endpoints(*edge_index).unwrap();
            let (span, label): (Span, String) = match edge_weight {
                StreamDependency::InvokeByName(span) => (
                    *span,
                    format!(
                        "The stream {} is invoked by stream {}",
                        self.get_stream_name(start_node),
                        self.get_stream_name(end_node)
                    ),
                ),
                StreamDependency::Access(location, _, span) => match location {
                    Location::Expression => (
                        *span,
                        format!(
                            "The stream {} accesses the stream {}",
                            self.get_stream_name(start_node),
                            self.get_stream_name(end_node)
                        ),
                    ),
                    Location::Invoke => (
                        *span,
                        format!(
                            "The stream {} accesses the stream {} in the invokation specification",
                            self.get_stream_name(start_node),
                            self.get_stream_name(end_node)
                        ),
                    ),
                    Location::Extend => (
                        *span,
                        format!(
                            "The stream {} accesses the stream {} in the extension expression",
                            self.get_stream_name(start_node),
                            self.get_stream_name(end_node)
                        ),
                    ),
                    Location::Terminate => (
                        *span,
                        format!(
                            "The stream {} accesses the stream {} in the termination expression",
                            self.get_stream_name(start_node),
                            self.get_stream_name(end_node)
                        ),
                    ),
                },
            };
            if let Some(ref mut builder) = builder {
                builder.add_span_with_label(span, label.as_str(), false);
            } else {
                let mut diagnostic_builder = self.handler.build_diagnostic(
                    "There is a positive weight cycle. This is a problem for monitoring.",
                    Level::Warning,
                );
                diagnostic_builder.add_labeled_span(LabeledSpan::new(span, label.as_str(), false));
                diagnostic_builder.prevent_sorting();
                builder = Some(diagnostic_builder);
            }
        }
        builder.unwrap().emit();
    }

    fn add_path_index_for_all_nodes(
        &mut self,
        cycles_per_stream: &mut HashMap<NodeId, CycleTracker>,
        path_index: usize,
        cyclic_path: &[EIx],
        weight: CycleWeight,
    ) {
        if let CycleWeight::Positive = weight {
            for edge_index in cyclic_path {
                let stream_id = self.get_stream_id_from_start_node(*edge_index);
                let node_entry = cycles_per_stream.get_mut(stream_id).unwrap();
                node_entry.positive.push(path_index);
            }
        } else {
            for edge_index in cyclic_path {
                let stream_id = self.get_stream_id_from_start_node(*edge_index);
                let node_entry = cycles_per_stream.get_mut(stream_id).unwrap();
                node_entry.negative.push(path_index);
            }
        }
    }

    fn get_stream_id_from_start_node(&mut self, edge_index: EIx) -> &NodeId {
        let edge_start_index = self.dependency_graph.edge_endpoints(edge_index).unwrap().0;
        let start_node_info = self.dependency_graph.node_weight(edge_start_index).unwrap();
        match start_node_info {
            StreamNode::ClassicOutput(id)
            | StreamNode::ParameterizedOutput(id)
            | StreamNode::RTOutput(id) => id,
            StreamNode::ClassicInput(_)
            | StreamNode::ParameterizedInput(_)
            | StreamNode::Trigger(_)
            | StreamNode::RTTrigger(_) => {
                unreachable!("Inputs and triggers must never appear in a cycle.")
            }
        }
    }
}

pub(crate) fn analyse_dependencies<'a>(
    spec: &'a LolaSpec,
    version_table: &'a LolaVersionTable,
    naming_table: &'a DeclarationTable,
    handler: &'a Handler,
) -> DependencyAnalysis {
    let analyser = DependencyAnalyser {
        dependency_graph: DependencyGraph::default(),
        spec,
        version_table,
        naming_table,
        handler,
        stream_names: HashMap::new(),
    };
    analyser.analyse_dependencies()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::id_assignment;
    use crate::analysis::lola_version::LolaVersionAnalysis;
    use crate::analysis::naming::NamingAnalysis;
    use crate::parse::parse;
    use crate::parse::SourceMapper;
    use crate::reporting::Handler;
    use std::path::PathBuf;

    /// Parses the content, runs naming analysis, and check expected number of errors and version
    fn check_graph(content: &str, num_errors: usize, num_warnings: usize) {
        let mut ast = parse(content).unwrap_or_else(|e| panic!("{}", e));
        id_assignment::assign_ids(&mut ast);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let mut naming_analyzer = NamingAnalysis::new(&handler);
        naming_analyzer.check(&ast);
        let mut version_analyzer = LolaVersionAnalysis::new(&handler);
        let version = version_analyzer.analyse(&ast);
        assert!(
            !version.is_none(),
            "We only analyze dependencies for specifications that so far seem to be ok."
        );
        let _dependency_analysis = analyse_dependencies(
            &ast,
            &version_analyzer.result,
            &naming_analyzer.result,
            &handler,
        );
        assert_eq!(num_errors, handler.emitted_errors());
        assert_eq!(num_warnings, handler.emitted_warnings());
    }

    #[test]
    fn simple_cycle() {
        check_graph(
            "input a: Int8\noutput b: Int8 := a+d\noutput c: Int8 := b\noutput d: Int8 := c",
            1,
            0,
        )
    }

    #[test]
    fn linear_should_be_no_problem() {
        check_graph(
            "input a: Int8\noutput b: Int8 := a\noutput c: Int8 := b\noutput d: Int8 := c",
            0,
            0,
        )
    }

    #[test]
    fn negative_cycle_should_be_no_problem() {
        check_graph("output a: Int8 := a[-1]?0", 0, 0)
    }

    #[test]
    fn positive_cycle_should_cause_a_warning() {
        check_graph("output a: Int8 := a[1]?0", 0, 1)
    }

    #[test]
    fn self_loop() {
        check_graph("input a: Int8\noutput b: Int8 := a\noutput c: Int8 := b\noutput d: Int8 := c\noutput e: Int8 := e",1,0)
    }

    #[test]
    fn parallel_edges_in_a_cycle() {
        check_graph(
            "input a: Int8\noutput b: Int8 := a+d+d\noutput c: Int8 := b\noutput d: Int8 := c",
            2,
            0,
        )
    }
}