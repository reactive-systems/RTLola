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
            self.check_expression(&output.expression);
        }
        for trigger in &self.spec.trigger {
            self.check_expression(&trigger.expression);
        }
    }

    fn check_expression(&self, expr: &Expression) {
        self.check_missing_paranthesis(expr);
        self.check_missing_expression(expr);
        self.check_offsets_are_literals(expr);
    }

    fn check_missing_paranthesis(&self, expr: &Expression) {
        use ExpressionKind::*;
        match &expr.kind {
            Lit(_) | Ident(_) | MissingExpression => {}
            Unary(_, inner) | Field(inner, _) | StreamAccess(inner, _) => {
                self.check_missing_paranthesis(&inner);
            }
            Binary(_, left, right)
            | Default(left, right)
            | Offset(left, right)
            | SlidingWindowAggregation { expr: left, duration: right, .. } => {
                self.check_missing_paranthesis(&left);
                self.check_missing_paranthesis(&right);
            }
            Ite(cond, normal, alternative) => {
                self.check_missing_paranthesis(&cond);
                self.check_missing_paranthesis(&normal);
                self.check_missing_paranthesis(&alternative);
            }
            ParenthesizedExpression(left, inner, right) => {
                if left.is_none() {
                    self.handler.warn_with_span(
                        "missing opening parenthesis",
                        LabeledSpan::new(expr.span, "This expression is missing an opening parenthesis.", true),
                    )
                }
                self.check_missing_paranthesis(&inner);
                if right.is_none() {
                    self.handler.warn_with_span(
                        "missing closing parenthesis",
                        LabeledSpan::new(expr.span, "This expression is missing a closing parenthesis.", true),
                    )
                }
            }
            Tuple(entries) | Function(_, _, entries) => {
                entries.iter().for_each(|entry| self.check_missing_paranthesis(entry));
            }
            Method(base, _, _, arguments) => {
                arguments.iter().for_each(|entry| self.check_missing_paranthesis(entry));
                self.check_missing_paranthesis(&base);
            }
        }
    }

    fn check_missing_expression(&self, expr: &Expression) {
        use ExpressionKind::*;
        match &expr.kind {
            Lit(_) | Ident(_) => {}
            MissingExpression => {
                self.handler.error_with_span(
                    "missing expression",
                    LabeledSpan::new(expr.span, "We expected an expression here.", true),
                );
            }
            Unary(_, inner) | Field(inner, _) | StreamAccess(inner, _) => {
                self.check_missing_expression(&inner);
            }
            Binary(_, left, right)
            | Default(left, right)
            | Offset(left, right)
            | SlidingWindowAggregation { expr: left, duration: right, .. } => {
                self.check_missing_expression(&left);
                self.check_missing_expression(&right);
            }
            Ite(cond, normal, alternative) => {
                self.check_missing_expression(&cond);
                self.check_missing_expression(&normal);
                self.check_missing_expression(&alternative);
            }
            ParenthesizedExpression(_, inner, _) => self.check_missing_expression(&inner),
            Tuple(entries) | Function(_, _, entries) => {
                entries.iter().for_each(|entry| self.check_missing_expression(entry))
            }
            Method(base, _, _, arguments) => {
                arguments.iter().for_each(|entry| self.check_missing_expression(entry));
                self.check_missing_expression(&base);
            }
        }
    }

    fn check_offsets_are_literals(&self, expr: &Expression) {
        use ExpressionKind::*;
        match &expr.kind {
            Lit(_) | Ident(_) => {}
            MissingExpression => {}
            Unary(_, inner) | Field(inner, _) | StreamAccess(inner, _) => {
                self.check_missing_expression(&inner);
            }
            Binary(_, left, right) | Default(left, right) => {
                self.check_missing_expression(&left);
                self.check_missing_expression(&right);
            }
            Offset(expr, offset) => {
                self.check_missing_expression(&expr);
                match &offset.kind {
                    ExpressionKind::Lit(l) => match l.kind {
                        LitKind::Numeric(_, _) => {
                            return;
                        }
                        _ => {}
                    },
                    _ => {}
                }
                self.handler.error_with_span(
                    "Offsets have to be numeric constants, like `42` or `10sec`",
                    LabeledSpan::new(expr.span, "Expected a numeric value", true),
                );
            }
            SlidingWindowAggregation { expr, duration, .. } => {
                self.check_missing_expression(&expr);
                match &duration.kind {
                    ExpressionKind::Lit(l) => match l.kind {
                        LitKind::Numeric(_, Some(_)) => {
                            return;
                        }
                        _ => {}
                    },
                    _ => {}
                }
                self.handler.error_with_span(
                    "Only durations are allowed in sliding windows",
                    LabeledSpan::new(expr.span, "Expected a duration", true),
                );
            }
            Ite(cond, normal, alternative) => {
                self.check_missing_expression(&cond);
                self.check_missing_expression(&normal);
                self.check_missing_expression(&alternative);
            }
            ParenthesizedExpression(_, inner, _) => self.check_missing_expression(&inner),
            Tuple(entries) | Function(_, _, entries) => {
                entries.iter().for_each(|entry| self.check_missing_expression(entry))
            }
            Method(base, _, _, arguments) => {
                arguments.iter().for_each(|entry| self.check_missing_expression(entry));
                self.check_missing_expression(&base);
            }
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
}
