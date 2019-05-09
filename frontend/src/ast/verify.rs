use super::*;
use crate::reporting::{Handler, LabeledSpan};

/// The grammar is an over-approximation of syntactical valid specifications
/// The verifier checks for those over-approximations and reports errors.
pub(crate) struct Verifier<'a, 'b> {
    spec: &'a LolaSpec,
    handler: &'b Handler,
}

impl<'a, 'b> Verifier<'a, 'b> {
    pub(crate) fn new(spec: &'a LolaSpec, handler: &'b Handler) -> Self {
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
        self.expression_walker(expr, &Self::check_missing_paranthesis);
        self.expression_walker(expr, &Self::check_missing_expression);
        self.expression_walker(expr, &Self::check_offsets_are_literals);
        self.expression_walker(expr, &Self::check_direct_access);
    }

    /// Iterates over the `Expression` AST and calls `check` on every node
    fn expression_walker<F>(&self, expr: &Expression, check: &F)
    where
        F: Fn(&Handler, &Expression) -> (),
    {
        check(self.handler, expr);

        // Recursion
        use ExpressionKind::*;
        match &expr.kind {
            Lit(_) | Ident(_) | MissingExpression => {}
            Unary(_, inner)
            | Field(inner, _)
            | StreamAccess(inner, _)
            | ParenthesizedExpression(_, inner, _) => {
                self.expression_walker(&inner, check);
            }
            Binary(_, left, right)
            | Default(left, right)
            | Offset(left, right)
            | SlidingWindowAggregation { expr: left, duration: right, .. } => {
                self.expression_walker(&left, check);
                self.expression_walker(&right, check);
            }
            Ite(cond, normal, alternative) => {
                self.expression_walker(&cond, check);
                self.expression_walker(&normal, check);
                self.expression_walker(&alternative, check);
            }
            Tuple(entries) | Function(_, _, entries) => {
                entries.iter().for_each(|entry| self.expression_walker(entry, check));
            }
            Method(base, _, _, arguments) => {
                self.expression_walker(&base, check);
                arguments.iter().for_each(|entry| self.expression_walker(entry, check));
            }
        }
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

    fn check_offsets_are_literals(handler: &Handler, expr: &Expression) {
        use ExpressionKind::*;
        match &expr.kind {
            Offset(_, offset) => {
                if let ExpressionKind::Lit(l) = &offset.kind {
                    if let LitKind::Numeric(_, _) = l.kind {
                        return;
                    }
                }
                handler.error_with_span(
                    "offsets have to be numeric constants, like `42` or `10sec`",
                    LabeledSpan::new(expr.span, "expected a numeric value", true),
                );
            }
            SlidingWindowAggregation { duration, .. } => {
                if let ExpressionKind::Lit(l) = &duration.kind {
                    if let LitKind::Numeric(_, Some(_)) = l.kind {
                        return;
                    }
                }
                handler.error_with_span(
                    "only durations are allowed in sliding windows",
                    LabeledSpan::new(expr.span, "expected a duration", true),
                );
            }
            _ => {}
        }
    }

    /// Currently, offsets are only allowed on direct stream access
    fn check_direct_access(handler: &Handler, expr: &Expression) {
        use ExpressionKind::*;
        match &expr.kind {
            Offset(inner, _) | SlidingWindowAggregation { expr: inner, .. } => {
                if let Ident(_) = inner.kind {
                    // is a direct access
                } else {
                    handler.error_with_span(
                        "operation can be only applied to streams directly",
                        LabeledSpan::new(inner.span, "expected a stream variable", true),
                    );
                }
            }
            StreamAccess(inner, _) => match inner.kind {
                Ident(_) => {}
                Offset(_, _) => {}
                _ => {
                    handler.error_with_span(
                        "operation can be only applied to streams directly or offset extressions",
                        LabeledSpan::new(inner.span, "expected a stream variable/offset expression", true),
                    );
                }
            },
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::parse::parse;
    use crate::parse::SourceMapper;
    use crate::reporting::Handler;
    use std::path::PathBuf;

    /// Parses the content, runs AST verifier, and returns number of warnings
    fn number_of_warnings(content: &str) -> usize {
        let ast = parse(content).unwrap_or_else(|e| panic!("{}", e));
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        super::Verifier::new(&ast, &handler).check();
        handler.emitted_warnings()
    }

    /// Parses the content, runs AST verifier, and returns number of errors
    fn number_of_errors(content: &str) -> usize {
        let ast = parse(content).unwrap_or_else(|e| panic!("{}", e));
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
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
    fn test_offsets_are_literals() {
        assert_eq!(0, number_of_errors("output a := x.offest(by: 1)"));
        assert_eq!(1, number_of_errors("output a := x.offset(by: y)"));
        assert_eq!(0, number_of_errors("output a := x.aggregate(over: 1h, using: avg)"));
        assert_eq!(1, number_of_errors("output a := x.aggregate(over: y, using: avg)"));
    }

    #[test]
    fn test_offsets_direct_access() {
        assert_eq!(0, number_of_errors("output a := x.offset(by: 1)"));
        assert_eq!(1, number_of_errors("output a := x.hold().offset(by: 1)"));
        assert_eq!(1, number_of_errors("output a := (x+1).hold()"));
        assert_eq!(1, number_of_errors("output a := (x+1).aggregate(over: 1h, using: avg)"));
    }
}
