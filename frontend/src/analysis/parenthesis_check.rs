use crate::ast::Expression;
use crate::ast::ExpressionKind;
use crate::ast::LolaSpec;
use crate::reporting::Handler;
use crate::reporting::LabeledSpan;

pub(crate) fn warn_about_missing_parenthesis(spec: &LolaSpec, handler: &Handler) {
    for output in &spec.outputs {
        warn_about_missing_parenthesis_in_expression(&output.expression, handler);
    }
    for trigger in &spec.trigger {
        warn_about_missing_parenthesis_in_expression(&trigger.expression, handler);
    }
}

pub(crate) fn warn_about_missing_parenthesis_in_expression(expr: &Expression, handler: &Handler) {
    match &expr.kind {
        ExpressionKind::Lit(_)
        | ExpressionKind::Lookup(_, _, _)
        | ExpressionKind::Ident(_)
        | ExpressionKind::MissingExpression => {}
        ExpressionKind::Unary(_, inner) | ExpressionKind::Field(inner, _) => {
            warn_about_missing_parenthesis_in_expression(&inner, handler);
        }
        ExpressionKind::Binary(_, left, right)
        | ExpressionKind::Hold(left, right)
        | ExpressionKind::Default(left, right) => {
            warn_about_missing_parenthesis_in_expression(&*left, handler);
            warn_about_missing_parenthesis_in_expression(&*right, handler);
        }
        ExpressionKind::Ite(cond, normal, alternative) => {
            warn_about_missing_parenthesis_in_expression(&*cond, handler);
            warn_about_missing_parenthesis_in_expression(&*normal, handler);
            warn_about_missing_parenthesis_in_expression(&*alternative, handler);
        }
        ExpressionKind::ParenthesizedExpression(left, inner, right) => {
            if left.is_none() {
                handler.warn_with_span(
                    "missing opening parenthesis",
                    LabeledSpan::new(expr.span, "This expression is missing an opening parenthesis.", true),
                )
            }
            warn_about_missing_parenthesis_in_expression(&inner, handler);
            if right.is_none() {
                handler.warn_with_span(
                    "missing closing parenthesis",
                    LabeledSpan::new(expr.span, "This expression is missing a closing parenthesis.", true),
                )
            }
        }
        ExpressionKind::Tuple(entries) | ExpressionKind::Function(_, _, entries) => {
            entries.iter().for_each(|entry| warn_about_missing_parenthesis_in_expression(&**entry, handler));
        }
        ExpressionKind::Method(base, _, _, arguments) => {
            arguments.iter().for_each(|entry| warn_about_missing_parenthesis_in_expression(&**entry, handler));
            warn_about_missing_parenthesis_in_expression(&base, handler);
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
    fn number_of_missing_parenthesis_warnings(content: &str) -> usize {
        let ast = parse(content).unwrap_or_else(|e| panic!("{}", e));
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        super::warn_about_missing_parenthesis(&ast, &handler);
        handler.emitted_warnings()
    }

    #[test]
    fn warn_about_missing_parenthesis() {
        assert_eq!(1, number_of_missing_parenthesis_warnings("output test: Int8 := (3+3"))
    }

    #[test]
    fn do_not_warn_about_existing_parenthesis() {
        assert_eq!(0, number_of_missing_parenthesis_warnings("output test: Int8 := (3+3)"))
    }
}
