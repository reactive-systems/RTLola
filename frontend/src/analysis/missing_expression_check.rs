use crate::ast::Expression;
use crate::ast::ExpressionKind;
use crate::ast::LolaSpec;
use crate::reporting::Handler;
use crate::reporting::LabeledSpan;

pub(crate) fn any_expression_missing(spec: &LolaSpec, handler: &Handler) -> bool {
    let mut any_missing = false;
    for output in &spec.outputs {
        any_missing |= any_expression_missing_in_expression(&output.expression, handler);
    }
    for trigger in &spec.trigger {
        any_missing |= any_expression_missing_in_expression(&trigger.expression, handler);
    }
    any_missing
}

pub(crate) fn any_expression_missing_in_expression(expr: &Expression, handler: &Handler) -> bool {
    match &expr.kind {
        ExpressionKind::Lit(_) | ExpressionKind::Lookup(_, _, _) | ExpressionKind::Ident(_) => false,
        ExpressionKind::MissingExpression => {
            handler.error_with_span(
                "missing expression",
                LabeledSpan::new(expr.span, "We expected an expression here.", true),
            );
            true
        }
        ExpressionKind::Unary(_, inner)
        | ExpressionKind::Field(inner, _)
        | ExpressionKind::StreamAccess(inner, _) => any_expression_missing_in_expression(&inner, handler),
        ExpressionKind::Binary(_, left, right)
        | ExpressionKind::Default(left, right)
        | ExpressionKind::Offset(left, right)
        | ExpressionKind::SlidingWindowAggregation { expr: left, duration: right, aggregation: _ } => {
            let left_missing = any_expression_missing_in_expression(&*left, handler);
            let right_missing = any_expression_missing_in_expression(&*right, handler);
            left_missing || right_missing
        }
        ExpressionKind::Ite(cond, normal, alternative) => {
            let in_cond_missing = any_expression_missing_in_expression(&*cond, handler);
            let in_normal_missing = any_expression_missing_in_expression(&*normal, handler);
            let in_alternative_missing = any_expression_missing_in_expression(&*alternative, handler);
            in_cond_missing || in_normal_missing || in_alternative_missing
        }
        ExpressionKind::ParenthesizedExpression(_, inner, _) => any_expression_missing_in_expression(&inner, handler),
        ExpressionKind::Tuple(entries) | ExpressionKind::Function(_, _, entries) => {
            entries.iter().map(|entry| any_expression_missing_in_expression(&**entry, handler)).any(|e| e)
        }
        ExpressionKind::Method(base, _, _, arguments) => {
            let missing_expression_in_arguments =
                arguments.iter().map(|entry| any_expression_missing_in_expression(&**entry, handler)).any(|e| e);
            let missing_in_base = any_expression_missing_in_expression(&base, handler);
            missing_expression_in_arguments || missing_in_base
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::parse::parse;
    use crate::parse::SourceMapper;
    use crate::reporting::Handler;
    use std::path::PathBuf;

    /// Parses the content, runs naming analysis, and returns number of errors
    fn number_of_missing_expression_errors(content: &str) -> usize {
        let ast = parse(content).unwrap_or_else(|e| panic!("{}", e));
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        super::any_expression_missing(&ast, &handler);
        handler.emitted_errors()
    }

    #[test]
    fn warn_about_missing_parenthesis() {
        assert_eq!(1, number_of_missing_expression_errors("output test: Int8 :="))
    }

    #[test]
    fn do_not_warn_about_existing_parenthesis() {
        assert_eq!(0, number_of_missing_expression_errors("output test: Int8 := (3+3)"))
    }
}
