use super::*;
use crate::reporting::{Handler, LabeledSpan};
use num::Signed;
use uom::si::time::second;

/// The grammar is an over-approximation of syntactical valid specifications
/// The verifier checks for those over-approximations and reports errors.
pub(crate) struct Verifier<'a, 'b> {
    spec: &'a RTLolaAst,
    handler: &'b Handler,
}

impl<'a, 'b> Verifier<'a, 'b> {
    pub(crate) fn new(spec: &'a RTLolaAst, handler: &'b Handler) -> Self {
        Self { spec, handler }
    }

    pub(crate) fn check(&self) {
        for output in &self.spec.outputs {
            if let Some(extend) = output.extend.expr.as_ref() {
                self.check_expression(extend);
            }
            self.check_expression(&output.expression);
        }
        for trigger in &self.spec.trigger {
            self.check_expression(&trigger.expression);
        }
    }

    fn check_expression(&self, expr: &Expression) {
        expr.iter().for_each(|inner| Self::check_missing_paranthesis(self.handler, inner));
        expr.iter().for_each(|inner| Self::check_missing_expression(self.handler, inner));
        expr.iter().for_each(|inner| Self::check_direct_access(self.handler, inner));
        expr.iter().for_each(|inner| Self::check_field_access(self.handler, inner));
        expr.iter().for_each(|inner| Self::check_valid_offset(self.handler, inner));
        expr.iter().for_each(|inner| Self::check_sliding_window_duration(self.handler, inner));
    }

    fn check_missing_paranthesis(handler: &Handler, expr: &Expression) {
        use ExpressionKind::*;
        if let ParenthesizedExpression(left, _, right) = &expr.kind {
            if left.is_none() {
                handler.warn_with_span(
                    "missing opening parenthesis",
                    LabeledSpan::new(expr.span, "this expression is missing an opening parenthesis", true),
                )
            }
            if right.is_none() {
                handler.warn_with_span(
                    "missing closing parenthesis",
                    LabeledSpan::new(expr.span, "this expression is missing a closing parenthesis", true),
                )
            }
        }
    }

    fn check_missing_expression(handler: &Handler, expr: &Expression) {
        use ExpressionKind::*;
        if let MissingExpression = &expr.kind {
            handler.error_with_span(
                "missing expression",
                LabeledSpan::new(expr.span, "we expected an expression here.", true),
            );
        }
    }

    /// Currently, offsets are only allowed on direct stream access
    fn check_direct_access(handler: &Handler, expr: &Expression) {
        use ExpressionKind::*;
        match &expr.kind {
            Offset(inner, _) | SlidingWindowAggregation { expr: inner, .. } | StreamAccess(inner, _) => {
                if let Ident(_) = inner.kind {
                    // is a direct access
                } else {
                    handler.error_with_span(
                        "operation can be only applied to streams directly",
                        LabeledSpan::new(inner.span, "expected a stream variable", true),
                    );
                }
            }
            _ => {}
        }
    }

    /// Currently, offsets can only be negative
    fn check_valid_offset(handler: &Handler, expr: &Expression) {
        use ExpressionKind::*;
        if let Offset(_, offset) = &expr.kind {
            if let super::Offset::Discrete(val) = offset {
                if *val == 0 {
                    handler
                        .error_with_span("only non-zero offsets are permitted", LabeledSpan::new(expr.span, "", true));
                }
            } else if let super::Offset::RealTime(val, _) = offset {
                if !val.is_negative() {
                    handler
                        .error_with_span("only negative offsets are supported", LabeledSpan::new(expr.span, "", true));
                }
            }
        }
    }

    fn check_field_access(handler: &Handler, expr: &Expression) {
        use ExpressionKind::*;
        if let Field(_, ident) = &expr.kind {
            if ident.name.parse::<usize>().is_err() {
                handler.error_with_span(
                    "field access has to be an integer",
                    LabeledSpan::new(ident.span, "expected an integer", true),
                );
            }
        }
    }

    fn check_sliding_window_duration(handler: &Handler, expr: &Expression) {
        use ExpressionKind::*;
        if let SlidingWindowAggregation { duration, .. } = &expr.kind {
            match duration.parse_duration() {
                Err(_) => {
                    handler.error_with_span(
                        "aggregation duration invalid",
                        LabeledSpan::new(duration.span, "duration invalid", true),
                    );
                }
                Ok(dur) => {
                    if !dur.get::<second>().is_positive() {
                        handler.error_with_span(
                            "only positive aggregation durations are supported",
                            LabeledSpan::new(duration.span, "duration non-positive", true),
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::parse::parse;
    use crate::parse::SourceMapper;
    use crate::reporting::Handler;
    use crate::FrontendConfig;
    use std::path::PathBuf;

    /// Parses the content, runs AST verifier, and returns number of warnings
    fn number_of_warnings(content: &str) -> usize {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let ast = parse(content, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        super::Verifier::new(&ast, &handler).check();
        handler.emitted_warnings()
    }

    /// Parses the content, runs AST verifier, and returns number of errors
    fn number_of_errors(content: &str) -> usize {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let ast = parse(content, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        super::Verifier::new(&ast, &handler).check();
        handler.emitted_errors()
    }

    #[test]
    fn warn_about_missing_parenthesis() {
        assert_eq!(1, number_of_warnings("output test: Int8 := (3+3"))
    }

    #[test]
    fn do_not_warn_about_existing_parenthesis() {
        assert_eq!(0, number_of_warnings("output test: Int8 := (3+3)"))
    }

    #[test]
    fn warn_about_missing_expression() {
        assert_eq!(1, number_of_errors("output test: Int8 :="))
    }

    #[test]
    fn do_not_warn_about_existing_expression() {
        assert_eq!(0, number_of_errors("output test: Int8 := (3+3)"))
    }

    #[test]
    fn test_offsets_direct_access() {
        assert_eq!(0, number_of_errors("output a := x.offset(by: -1)"));
        assert_eq!(1, number_of_errors("output a := x.offset(by: 0)"));
        assert_eq!(0, number_of_errors("output a := x.offset(by: 1)"));
        assert_eq!(0, number_of_errors("output a := x.offset(by: -1s)"));
        assert_eq!(1, number_of_errors("output a := x.offset(by: 0s)"));
        assert_eq!(1, number_of_errors("output a := x.offset(by: 1s)"));
        assert_eq!(1, number_of_errors("output a := x.hold().offset(by: -1)"));
        assert_eq!(1, number_of_errors("output a := (x+1).hold()"));
        assert_eq!(1, number_of_errors("output a := (x+1).aggregate(over: 1h, using: avg)"));
    }
}
