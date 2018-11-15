use super::super::ast::*;
extern crate ast_node;
use crate::analysis::AnalysisError;
use ast_node::{AstNode, NodeId};
use std::collections::HashMap;

pub(crate) type LolaVersionTable = HashMap<NodeId, LanguageSpec>;

struct VersionTracker {
    pub can_be_classic: bool,
    pub can_be_lola2: bool,
}

impl VersionTracker {
    fn new() -> Self {
        VersionTracker {
            can_be_classic: true,
            can_be_lola2: true,
        }
    }
    fn from_stream(is_not_parameterized: bool) -> Self {
        VersionTracker {
            can_be_classic: is_not_parameterized,
            can_be_lola2: true,
        }
    }
}

fn analyse_expression(
    version_tracker: &mut VersionTracker,
    expr: &Expression,
    toplevel_in_trigger: bool,
) -> () {
    match &expr.kind {
        ExpressionKind::Lit(_) => {}
        ExpressionKind::Ident(_) => {}
        ExpressionKind::Default(target, default) => {
            analyse_expression(version_tracker, &*target, false);
            analyse_expression(version_tracker, &*default, false);
        }
        ExpressionKind::Lookup(_, offset, _) => match offset {
            Offset::DiscreteOffset(expr) => {
                analyse_expression(version_tracker, &*expr, false);
            }
            Offset::RealTimeOffset(expr, _) => {
                analyse_expression(version_tracker, &*expr, false);
                version_tracker.can_be_lola2 = false;
                version_tracker.can_be_classic = false;
            }
        },
        ExpressionKind::Binary(_, left, right) => {
            analyse_expression(version_tracker, &*left, false);
            analyse_expression(version_tracker, &*right, false);
        }
        ExpressionKind::Unary(_, nested) => {
            analyse_expression(version_tracker, &*nested, false);
        }
        ExpressionKind::Ite(condition, if_case, else_case) => {
            analyse_expression(version_tracker, &*condition, false);
            analyse_expression(version_tracker, &*if_case, false);
            analyse_expression(version_tracker, &*else_case, false);
        }
        ExpressionKind::ParenthesizedExpression(_, nested, _) => {
            analyse_expression(version_tracker, &*nested, false);
        }
        ExpressionKind::MissingExpression() => {}
        ExpressionKind::Tuple(nested_exprs) => {
            nested_exprs.iter().for_each(|nested| {
                analyse_expression(version_tracker, &*nested, false);
            });
        }
        ExpressionKind::Function(_, arguments) => {
            arguments.iter().for_each(|arg| {
                analyse_expression(version_tracker, &*arg, false);
            });
        }
    }
}

pub(crate) struct LolaVersionAnalysis {
    pub result: LolaVersionTable,
}

impl<'a> LolaVersionAnalysis {
    pub(crate) fn new() -> Self {
        LolaVersionAnalysis {
            result: HashMap::new(),
        }
    }

    fn analyse_input(&mut self, input: &'a Input) {
        if input.params.is_empty() {
            self.result.insert(*input.id(), LanguageSpec::Classic);
        } else {
            self.result.insert(*input.id(), LanguageSpec::Lola2);
        }
    }

    fn analyse_output(&mut self, output: &'a Output) {
        let mut version_tracker = VersionTracker::from_stream(output.params.is_empty());
        analyse_expression(&mut version_tracker, &output.expression, false);

        // TODO check parameters for InvocationType

        if version_tracker.can_be_classic {
            self.result.insert(*output.id(), LanguageSpec::Classic);
            return;
        }
        if version_tracker.can_be_lola2 {
            self.result.insert(*output.id(), LanguageSpec::Lola2);
            return;
        }
        self.result.insert(*output.id(), LanguageSpec::RTLola);
    }

    fn analyse_trigger(&mut self, trigger: &'a Trigger) {
        let mut version_tracker = VersionTracker::new();
        analyse_expression(&mut version_tracker, &trigger.expression, true);

        if version_tracker.can_be_classic {
            self.result.insert(*trigger.id(), LanguageSpec::Classic);
            return;
        }
        if version_tracker.can_be_lola2 {
            self.result.insert(*trigger.id(), LanguageSpec::Lola2);
            return;
        }
        self.result.insert(*trigger.id(), LanguageSpec::RTLola);
    }

    pub(crate) fn analyse(&mut self, spec: &'a LolaSpec) -> LanguageSpec {
        // analyse each stream/trigger to find out their minimal Lola version
        for input in &spec.inputs {
            self.analyse_input(&input);
        }
        for output in &spec.outputs {
            self.analyse_output(&output);
        }
        for trigger in &spec.trigger {
            self.analyse_trigger(&trigger);
        }

        // each stream/trigger can be attributed to some (minimal) Lola version but the different versions might be incompatible.
        // Therefore iterate again over all streams and triggers and record reasons against the various versions.
        let mut reason_against_classic_lola: Option<String> = None;
        let mut reason_against_lola2: Option<String> = None;

        self.rule_out_versions_based_on_inputs(&spec, &mut reason_against_classic_lola);

        self.rule_out_versions_based_on_outputs(
            &spec,
            &mut reason_against_classic_lola,
            &mut reason_against_lola2,
        );
        self.rule_out_versions_based_on_triggers(
            &spec,
            &mut reason_against_classic_lola,
            &mut reason_against_lola2,
        );

        // Try to use the minimal Lola version or give an error containing the reasons why none of the versions is possible.
        if reason_against_classic_lola.is_none() {
            return LanguageSpec::Classic;
        }
        if reason_against_lola2.is_none() {
            return LanguageSpec::Lola2;
        }
        LanguageSpec::RTLola
    }

    fn rule_out_versions_based_on_triggers(
        &mut self,
        spec: &LolaSpec,
        reason_against_classic_lola: &mut Option<String>,
        reason_against_lola2: &mut Option<String>,
    ) {
        for trigger in &spec.trigger {
            let name = match trigger.name {
                None => "an unnamed trigger",
                Some(ref trigger_name) => trigger_name.name.as_str(),
            };
            match &self.result[trigger.id()] {
                LanguageSpec::Classic => {}
                LanguageSpec::Lola2 => {
                    if reason_against_classic_lola.is_none() {
                        *reason_against_classic_lola = Some(format!(
                            "Classic Lola is not possible due to {} being a Lola2 trigger.",
                            name
                        ))
                    }
                }
                LanguageSpec::RTLola => {
                    if reason_against_classic_lola.is_none() {
                        *reason_against_classic_lola = Some(format!(
                            "Classic Lola is not possible due to {} being a RTLola trigger.",
                            name
                        ))
                    }
                    if reason_against_lola2.is_none() {
                        *reason_against_lola2 = Some(format!(
                            "Lola2 is not possible due to {} being a RTLola stream.",
                            name
                        ))
                    }
                }
            }
        }
    }

    fn rule_out_versions_based_on_outputs(
        &mut self,
        spec: &LolaSpec,
        reason_against_classic_lola: &mut Option<String>,
        reason_against_lola2: &mut Option<String>,
    ) {
        for output in &spec.outputs {
            match &self.result[output.id()] {
                LanguageSpec::Classic => {}
                LanguageSpec::Lola2 => {
                    if reason_against_classic_lola.is_none() {
                        *reason_against_classic_lola = Some(format!(
                            "Classic Lola is not possible due to {} being a Lola2 stream.",
                            output.name.name
                        ));
                    }
                }
                LanguageSpec::RTLola => {
                    if reason_against_classic_lola.is_none() {
                        *reason_against_classic_lola = Some(format!(
                            "Classic Lola is not possible due to {} being a RTLola stream.",
                            output.name.name
                        ))
                    }
                    if reason_against_lola2.is_none() {
                        *reason_against_lola2 = Some(format!(
                            "Lola2 is not possible due to {} being a RTLola stream.",
                            output.name.name
                        ))
                    }
                }
            }
        }
    }

    fn rule_out_versions_based_on_inputs(
        &mut self,
        spec: &LolaSpec,
        reason_against_classic_lola: &mut Option<String>,
    ) {
        for input in &spec.inputs {
            match &self.result[input.id()] {
                LanguageSpec::Classic => {}
                LanguageSpec::Lola2 => {
                    if reason_against_classic_lola.is_none() {
                        *reason_against_classic_lola = Some(format!(
                            "Classic Lola is not possible due to {} being parameterized.",
                            input.name.name
                        ))
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::id_assignment;
    use crate::parse::parse;

    // TODO: implement test cases
    #[test]
    fn parameterized_output_stream_causes_lola2() -> Result<(), String> {
        let spec = "output test<ab: Int8, c: Int8>: Int8 := 3";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut version_analyzer = LolaVersionAnalysis::new();
        let version = version_analyzer.analyse(&ast);
        assert_eq!(version, LanguageSpec::Lola2);
        Ok(())
    }

    #[test]
    fn time_offset_causes_rtlola() -> Result<(), String> {
        let spec = "output test: Int8 := stream[3 s]";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut version_analyzer = LolaVersionAnalysis::new();
        let version = version_analyzer.analyse(&ast);
        assert_eq!(version, LanguageSpec::RTLola);
        Ok(())
    }

    #[test]
    fn simple_trigger_causes_lola() -> Result<(), String> {
        let spec = "trigger test := false";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut version_analyzer = LolaVersionAnalysis::new();
        let version = version_analyzer.analyse(&ast);
        assert_eq!(version, LanguageSpec::Classic);
        Ok(())
    }

    #[test]
    fn time_offset_in_trigger_causes_rtlola() -> Result<(), String> {
        let spec = "trigger test := stream[3 s]";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut version_analyzer = LolaVersionAnalysis::new();
        let version = version_analyzer.analyse(&ast);
        assert_eq!(version, LanguageSpec::RTLola);
        Ok(())
    }

    #[test]
    fn parameterized_input_stream_causes_lola2() -> Result<(), String> {
        let spec = "input test<ab: Int8, c: Int8> : Int8";
        let throw = |e| panic!("{}", e);
        let mut ast = parse(spec).unwrap_or_else(throw);
        id_assignment::assign_ids(&mut ast);
        let mut version_analyzer = LolaVersionAnalysis::new();
        let version = version_analyzer.analyse(&ast);
        assert_eq!(version, LanguageSpec::Lola2);
        Ok(())
    }
}
