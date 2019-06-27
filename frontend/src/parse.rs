//! This module contains the parser for the Lola Language.

use super::ast::*;
use crate::reporting::{Handler, LabeledSpan};
use lazy_static::lazy_static;
use pest;
use pest::iterators::{Pair, Pairs};
use pest::prec_climber::{Assoc, Operator, PrecClimber};
use pest::Parser;
use pest_derive::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[grammar = "lola.pest"]
pub(crate) struct LolaParser;

lazy_static! {
    // precedence taken from C/C++: https://en.wikipedia.org/wiki/Operators_in_C_and_C++
    // Precedence climber can be used to build the AST, see https://pest-parser.github.io/book/ for more details
    static ref PREC_CLIMBER: PrecClimber<Rule> = {
        use self::Assoc::*;
        use self::Rule::*;

        PrecClimber::new(vec![
            Operator::new(Or, Right),
            Operator::new(And, Right),
            Operator::new(Equal, Left) | Operator::new(NotEqual, Left),
            Operator::new(LessThan, Left) | Operator::new(LessThanOrEqual, Left) | Operator::new(MoreThan, Left) | Operator::new(MoreThanOrEqual, Left),
            Operator::new(Add, Left) | Operator::new(Subtract, Left),
            Operator::new(Multiply, Left) | Operator::new(Divide, Left) | Operator::new(Mod, Left),
            Operator::new(Power, Right),
            Operator::new(Dot, Left),
            Operator::new(OpeningBracket, Left),
        ])
    };
}

/**
 * Transforms a textual representation of a Lola specification into
 * an AST representation.
 */
pub(crate) fn parse<'a, 'b>(content: &'a str, handler: &'b Handler) -> Result<LolaSpec, pest::error::Error<Rule>> {
    let mut pairs = LolaParser::parse(Rule::Spec, content)?;
    let mut spec = LolaSpec::new();
    assert!(pairs.clone().count() == 1, "Spec must not be empty.");
    let spec_pair = pairs.next().unwrap();
    assert!(spec_pair.as_rule() == Rule::Spec);
    for pair in spec_pair.into_inner() {
        match pair.as_rule() {
            Rule::LanguageSpec => {
                spec.language = Some(LanguageSpec::from(pair.as_str()));
            }
            Rule::ImportStmt => {
                let import = parse_import(&mut spec, pair);
                spec.imports.push(import);
            }
            Rule::ConstantStream => {
                let constant = parse_constant(&mut spec, pair);
                spec.constants.push(constant);
            }
            Rule::InputStream => {
                let input = parse_inputs(&mut spec, pair);
                spec.inputs.extend(input);
            }
            Rule::OutputStream => {
                let output = parse_output(&mut spec, pair, handler);
                spec.outputs.push(output);
            }
            Rule::Trigger => {
                let trigger = parse_trigger(&mut spec, pair, handler);
                spec.trigger.push(trigger);
            }
            Rule::TypeDecl => {
                let type_decl = parse_type_declaration(&mut spec, pair);
                spec.type_declarations.push(type_decl);
            }
            Rule::EOI => {}
            _ => unreachable!(),
        }
    }
    crate::analysis::id_assignment::assign_ids(&mut spec);
    Ok(spec)
}

fn parse_import(_spec: &mut LolaSpec, pair: Pair<Rule>) -> Import {
    assert_eq!(pair.as_rule(), Rule::ImportStmt);
    let span = pair.as_span().into();
    let mut pairs = pair.into_inner();
    let name = parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
    Import { name, id: NodeId::DUMMY, span }
}

/**
 * Transforms a `Rule::ConstantStream` into `Constant` AST node.
 * Panics if input is not `Rule::ConstantStream`.
 * The constant rule consists of the following tokens:
 * - `Rule::Ident`
 * - `Rule::Type`
 * - `Rule::Literal`
 */
fn parse_constant(spec: &mut LolaSpec, pair: Pair<'_, Rule>) -> Constant {
    assert_eq!(pair.as_rule(), Rule::ConstantStream);
    let span = pair.as_span().into();
    let mut pairs = pair.into_inner();
    let name = parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
    let ty = parse_type(spec, pairs.next().expect("mismatch between grammar and AST"));
    let literal = parse_literal(pairs.next().expect("mismatch between grammar and AST"));
    Constant { id: NodeId::DUMMY, name, ty: Some(ty), literal, span }
}

/**
 * Transforms a `Rule::InputStream` into `Input` AST node.
 * Panics if input is not `Rule::InputStream`.
 * The input rule consists of non-empty sequences of following tokens:
 * - `Rule::Ident`
 * - (`Rule::ParamList`)?
 * - `Rule::Type`
 */
fn parse_inputs(spec: &mut LolaSpec, pair: Pair<'_, Rule>) -> Vec<Input> {
    assert_eq!(pair.as_rule(), Rule::InputStream);
    let mut inputs = Vec::new();
    let mut pairs = pair.into_inner();
    while let Some(pair) = pairs.next() {
        let start = pair.as_span().start();
        let name = parse_ident(&pair);

        let mut pair = pairs.next().expect("mismatch between grammar and AST");
        let params = if let Rule::ParamList = pair.as_rule() {
            let res = parse_parameter_list(spec, pair.into_inner());
            pair = pairs.next().expect("mismatch between grammar and AST");
            res
        } else {
            Vec::new()
        };
        let end = pair.as_span().end();
        let ty = parse_type(spec, pair);
        inputs.push(Input { id: NodeId::DUMMY, name, params, ty, span: Span { start, end } })
    }

    assert!(!inputs.is_empty());
    inputs
}

/**
 * Transforms a `Rule::OutputStream` into `Output` AST node.
 * Panics if input is not `Rule::OutputStream`.
 * The output rule consists of the following tokens:
 * - `Rule::Ident`
 * - `Rule::Type`
 * - `Rule::Expr`
 */
fn parse_output(spec: &mut LolaSpec, pair: Pair<'_, Rule>, handler: &Handler) -> Output {
    assert_eq!(pair.as_rule(), Rule::OutputStream);
    let span = pair.as_span().into();
    let mut pairs = pair.into_inner();
    let name = parse_ident(&pairs.next().expect("mismatch between grammar and AST"));

    let mut pair = pairs.next().expect("mismatch between grammar and AST");
    let params = if let Rule::ParamList = pair.as_rule() {
        let res = parse_parameter_list(spec, pair.into_inner());
        pair = pairs.next().expect("mismatch between grammar and AST");
        res
    } else {
        Vec::new()
    };

    let ty = if let Rule::Type = pair.as_rule() {
        let ty = parse_type(spec, pair);
        pair = pairs.next().expect("mismatch between grammar and AST");
        ty
    } else {
        Type::new_inferred()
    };

    // Parse the `@ [Expr]` part of output declaration
    let extend = if let Rule::ActivationCondition = pair.as_rule() {
        let span: Span = pair.as_span().into();
        let expr = build_expression_ast(spec, pair.into_inner(), handler);
        pair = pairs.next().expect("mismatch between grammar and AST");
        ActivationCondition { expr: Some(expr), id: NodeId::DUMMY, span }
    } else {
        ActivationCondition { expr: None, id: NodeId::DUMMY, span: Span::unknown() }
    };

    let mut tspec = None;
    if let Rule::TemplateSpec = pair.as_rule() {
        tspec = Some(parse_template_spec(spec, pair, handler));
        pair = pairs.next().expect("mismatch between grammar and AST");
    };

    // Parse expression
    let expression = build_expression_ast(spec, pair.into_inner(), handler);
    Output { id: NodeId::DUMMY, name, ty, extend, params, template_spec: tspec, expression, span }
}

fn parse_parameter_list(spec: &mut LolaSpec, param_list: Pairs<'_, Rule>) -> Vec<Parameter> {
    let mut params = Vec::new();
    for param_decl in param_list {
        assert_eq!(Rule::ParameterDecl, param_decl.as_rule());
        let span = param_decl.as_span().into();
        let mut decl = param_decl.into_inner();
        let name = parse_ident(&decl.next().expect("mismatch between grammar and AST"));
        let ty = if let Some(type_pair) = decl.next() {
            assert_eq!(Rule::Type, type_pair.as_rule());
            parse_type(spec, type_pair)
        } else {
            Type::new_inferred()
        };
        params.push(Parameter { name, ty, id: NodeId::DUMMY, span });
    }
    params
}

fn parse_template_spec(spec: &mut LolaSpec, pair: Pair<'_, Rule>, handler: &Handler) -> TemplateSpec {
    let span = pair.as_span().into();
    let mut decls = pair.into_inner();
    let mut pair = decls.next();
    let mut rule = pair.as_ref().map(Pair::as_rule);

    let mut inv_spec = None;
    if let Some(Rule::InvokeDecl) = rule {
        inv_spec = Some(parse_inv_spec(spec, pair.unwrap(), handler));
        pair = decls.next();
        rule = pair.as_ref().map(Pair::as_rule);
    }
    let mut ext_spec = None;
    if let Some(Rule::ExtendDecl) = rule {
        ext_spec = Some(parse_ext_spec(spec, pair.unwrap(), handler));
        pair = decls.next();
        rule = pair.as_ref().map(Pair::as_rule);
    }
    let mut ter_spec = None;
    if let Some(Rule::TerminateDecl) = rule {
        let exp = pair.unwrap();
        let span_ter = exp.as_span().into();
        let expr = exp.into_inner().next().expect("mismatch between grammar and AST");
        let expr = build_expression_ast(spec, expr.into_inner(), handler);
        ter_spec = Some(TerminateSpec { target: expr, id: NodeId::DUMMY, span: span_ter });
    }
    TemplateSpec { inv: inv_spec, ext: ext_spec, ter: ter_spec, id: NodeId::DUMMY, span }
}

fn parse_ext_spec(spec: &mut LolaSpec, ext_pair: Pair<'_, Rule>, handler: &Handler) -> ExtendSpec {
    let span_ext = ext_pair.as_span().into();
    let mut children = ext_pair.into_inner();

    let first_child = children.next().expect("mismatch between grammar and ast");
    let target = match first_child.as_rule() {
        Rule::Expr => build_expression_ast(spec, first_child.into_inner(), handler),
        _ => unreachable!(),
    };
    ExtendSpec { target, id: NodeId::DUMMY, span: span_ext }
}

fn parse_inv_spec(spec: &mut LolaSpec, inv_pair: Pair<'_, Rule>, handler: &Handler) -> InvokeSpec {
    let span_inv = inv_pair.as_span().into();
    let mut inv_children = inv_pair.into_inner();
    let expr_pair = inv_children.next().expect("mismatch between grammar and AST");
    let inv_target = build_expression_ast(spec, expr_pair.into_inner(), handler);
    // Compute invocation condition:
    let mut is_if = false;
    let mut cond_expr = None;
    if let Some(inv_cond_pair) = inv_children.next() {
        is_if = match inv_cond_pair.as_rule() {
            Rule::InvokeIf => true,
            Rule::InvokeUnless => false,
            _ => unreachable!(),
        };
        let condition = inv_cond_pair.into_inner().next().expect("mismatch between grammar and AST");
        cond_expr = Some(build_expression_ast(spec, condition.into_inner(), handler))
    }
    InvokeSpec { condition: cond_expr, is_if, target: inv_target, id: NodeId::DUMMY, span: span_inv }
}

/**
 * Transforms a `Rule::Trigger` into `Trigger` AST node.
 * Panics if input is not `Rule::Trigger`.
 * The output rule consists of the following tokens:
 * - (`Rule::Ident`)?
 * - `Rule::Expr`
 * - (`Rule::StringLiteral`)?
 */
fn parse_trigger(spec: &mut LolaSpec, pair: Pair<'_, Rule>, handler: &Handler) -> Trigger {
    assert_eq!(pair.as_rule(), Rule::Trigger);
    let span = pair.as_span().into();
    let mut pairs = pair.into_inner();

    let mut name = None;
    let mut message = None;

    let mut pair = pairs.next().expect("mismatch between grammar and AST");
    // first token is either expression or identifier
    if let Rule::Ident = pair.as_rule() {
        name = Some(parse_ident(&pair));
        pair = pairs.next().expect("mismatch between grammar and AST");
    }
    let expression = build_expression_ast(spec, pair.into_inner(), handler);

    if let Some(pair) = pairs.next() {
        assert_eq!(pair.as_rule(), Rule::String);
        message = Some(pair.as_str().to_string());
    }

    Trigger { id: NodeId::DUMMY, name, expression, message, span }
}

/**
 * Transforms a `Rule::Ident` into `Ident` AST node.
 * Panics if input is not `Rule::Ident`.
 */
fn parse_ident(pair: &Pair<'_, Rule>) -> Ident {
    assert_eq!(pair.as_rule(), Rule::Ident);
    let name = pair.as_str().to_string();
    Ident::new(name, pair.as_span().into())
}

/**
 * Transforms a `Rule::TypeDecl` into `TypeDeclaration` AST node.
 * Panics if input is not `Rule::TypeDecl`.
 */
fn parse_type_declaration(spec: &mut LolaSpec, pair: Pair<'_, Rule>) -> TypeDeclaration {
    assert_eq!(pair.as_rule(), Rule::TypeDecl);
    let span = pair.as_span().into();
    let mut pairs = pair.into_inner();
    let name = parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
    let mut fields = Vec::new();
    while let Some(pair) = pairs.next() {
        let field_name = pair.as_str().to_string();
        let ty = parse_type(spec, pairs.next().expect("mismatch between grammar and AST"));
        fields.push(Box::new(TypeDeclField { name: field_name, ty, id: NodeId::DUMMY, span: pair.as_span().into() }));
    }

    TypeDeclaration { name: Some(name), span, id: NodeId::DUMMY, fields }
}

/**
 * Transforms a `Rule::Type` into `Type` AST node.
 * Panics if input is not `Rule::Type`.
 */
fn parse_type(spec: &mut LolaSpec, pair: Pair<'_, Rule>) -> Type {
    assert_eq!(pair.as_rule(), Rule::Type);
    let span = pair.as_span();
    let mut tuple = Vec::new();
    for pair in pair.into_inner() {
        match pair.as_rule() {
            Rule::Ident => {
                return Type::new_simple(pair.as_str().to_string(), pair.as_span().into());
            }
            Rule::Type => tuple.push(parse_type(spec, pair)),
            _ => unreachable!("{:?} is not a type", pair.as_rule()),
        }
    }
    Type::new_tuple(tuple, span.into())
}

/**
 * Transforms a `Rule::Literal` into `Literal` AST node.
 * Panics if input is not `Rule::Literal`.
 */
fn parse_literal(pair: Pair<'_, Rule>) -> Literal {
    assert_eq!(pair.as_rule(), Rule::Literal);
    let inner = pair.into_inner().next().expect("Rule::Literal has exactly one child");
    match inner.as_rule() {
        Rule::String => {
            let str_rep = inner.as_str();
            Literal::new_str(str_rep, inner.as_span().into())
        }
        Rule::RawString => {
            let str_rep = inner.as_str();
            Literal::new_raw_str(str_rep, inner.as_span().into())
        }
        Rule::NumberLiteral => {
            let span = inner.as_span();
            let mut pairs = inner.into_inner();
            let value = pairs.next().expect("Mismatch between AST and grammar");

            let str_rep: &str = value.as_str();
            let unit = match pairs.next() {
                None => None,
                Some(unit) => Some(unit.as_str().to_string()),
            };

            Literal::new_numeric(str_rep, unit, span.into())
        }
        Rule::True => Literal::new_bool(true, inner.as_span().into()),
        Rule::False => Literal::new_bool(false, inner.as_span().into()),
        _ => unreachable!(),
    }
}

#[allow(clippy::vec_box)]
fn parse_vec_of_expressions(spec: &mut LolaSpec, pairs: Pairs<'_, Rule>, handler: &Handler) -> Vec<Box<Expression>> {
    pairs.map(|expr| build_expression_ast(spec, expr.into_inner(), handler)).map(Box::new).collect()
}

fn parse_vec_of_types(spec: &mut LolaSpec, pairs: Pairs<'_, Rule>) -> Vec<Type> {
    pairs.map(|expr| parse_type(spec, expr)).collect()
}

fn build_function_expression(spec: &mut LolaSpec, pair: Pair<'_, Rule>, span: Span, handler: &Handler) -> Expression {
    let mut children = pair.into_inner();
    let fun_name = parse_ident(&children.next().unwrap());
    let mut next = children.next().expect("Mismatch between AST and parser");
    let type_params = match next.as_rule() {
        Rule::GenericParam => {
            let params = parse_vec_of_types(spec, next.into_inner());
            next = children.next().expect("Mismatch between AST and parser");
            params
        }
        Rule::FunctionArgs => Vec::new(),
        _ => unreachable!(),
    };
    assert_eq!(next.as_rule(), Rule::FunctionArgs);
    let mut args = Vec::new();
    let mut arg_names = Vec::new();
    for pair in next.into_inner() {
        assert_eq!(pair.as_rule(), Rule::FunctionArg);
        let mut pairs = pair.into_inner();
        let mut pair = pairs.next().expect("Mismatch between AST and parser");
        if pair.as_rule() == Rule::Ident {
            // named argument
            arg_names.push(Some(parse_ident(&pair)));
            pair = pairs.next().expect("Mismatch between AST and parser");
        } else {
            arg_names.push(None);
        }
        args.push(build_expression_ast(spec, pair.into_inner(), handler).into());
    }
    let name = FunctionName { name: fun_name, arg_names };
    Expression::new(ExpressionKind::Function(name, type_params, args), span)
}

/**
 * Builds the Expr AST.
 */
fn build_expression_ast(spec: &mut LolaSpec, pairs: Pairs<'_, Rule>, handler: &Handler) -> Expression {
    PREC_CLIMBER.climb(
        pairs,
        |pair: Pair<'_, Rule>| build_term_ast(spec, pair, handler),
        |lhs: Expression, op: Pair<'_, Rule>, rhs: Expression| {
            // Reduce function combining `Expression`s to `Expression`s with the correct precs
            let span = Span { start: lhs.span.start, end: rhs.span.end };
            let op = match op.as_rule() {
                Rule::Add => BinOp::Add,
                Rule::Subtract => BinOp::Sub,
                Rule::Multiply => BinOp::Mul,
                Rule::Divide => BinOp::Div,
                Rule::Mod => BinOp::Rem,
                Rule::Power => BinOp::Pow,
                Rule::And => BinOp::And,
                Rule::Or => BinOp::Or,
                Rule::LessThan => BinOp::Lt,
                Rule::LessThanOrEqual => BinOp::Le,
                Rule::MoreThan => BinOp::Gt,
                Rule::MoreThanOrEqual => BinOp::Ge,
                Rule::Equal => BinOp::Eq,
                Rule::NotEqual => BinOp::Ne,
                // bubble up the unary operator on the lhs (if it exists) to fix precedence
                Rule::Dot => {
                    let (unop, binop_span, inner) = match lhs.kind {
                        ExpressionKind::Unary(unop, inner) => {
                            (Some(unop), Span { start: inner.span.start, end: rhs.span.end }, inner)
                        }
                        _ => (None, span, Box::new(lhs)),
                    };
                    match rhs.kind {
                        // access to a tuple
                        ExpressionKind::Lit(l) => {
                            let ident = match l.kind {
                                LitKind::Numeric(val, unit) => {
                                    assert!(unit.is_none());
                                    Ident::new(val.clone(), l.span)
                                }
                                _ => {
                                    handler.error_with_span(
                                        &format!("expected unsigned integer, found {}", l),
                                        LabeledSpan::new(rhs.span, "unexpected", true),
                                    );
                                    std::process::exit(1);
                                }
                            };
                            let binop_expr = Expression::new(ExpressionKind::Field(inner, ident), binop_span);
                            match unop {
                                None => return binop_expr,
                                Some(unop) => {
                                    return Expression::new(ExpressionKind::Unary(unop, Box::new(binop_expr)), span)
                                }
                            }
                        }
                        ExpressionKind::Function(name, types, args) => {
                            // match for builtin function names and transform them into appropriate AST nodes
                            let kind = match name.as_string().as_str() {
                                "defaults(to:)" => {
                                    assert_eq!(args.len(), 1);
                                    ExpressionKind::Default(inner, args[0].clone())
                                }
                                "offset(by:)" => {
                                    assert_eq!(args.len(), 1);
                                    let offset_expr = &args[0];
                                    let offset = match offset_expr.parse_offset() {
                                        Ok(offset) => offset,
                                        Err(reason) => {
                                            handler.error_with_span(
                                                "failed to parse offset",
                                                LabeledSpan::new(rhs.span, &format!("{}", reason), true),
                                            );
                                            std::process::exit(1);
                                        }
                                    };
                                    ExpressionKind::Offset(inner, offset)
                                }
                                "hold()" => {
                                    assert_eq!(args.len(), 0);
                                    ExpressionKind::StreamAccess(inner, StreamAccessKind::Hold)
                                }
                                "get()" => {
                                    assert_eq!(args.len(), 0);
                                    ExpressionKind::StreamAccess(inner, StreamAccessKind::Optional)
                                }
                                "aggregate(over:using:)" => {
                                    assert_eq!(args.len(), 2);
                                    let window_op = match &args[1].kind {
                                        ExpressionKind::Ident(i) => match i.name.as_str() {
                                            "Σ" | "sum" => WindowOperation::Sum,
                                            "#" | "count" => WindowOperation::Count,
                                            "Π" | "prod" => WindowOperation::Product,
                                            "∫" | "integral" => WindowOperation::Integral,
                                            "avg" => WindowOperation::Average,
                                            fun => {
                                                handler.error_with_span(
                                                    &format!("unknown aggregation function {}", fun),
                                                    LabeledSpan::new(i.span, "try count, sum, average", true),
                                                );
                                                std::process::exit(1);
                                            }
                                        },
                                        _ => {
                                            handler.error_with_span(
                                                "expected aggregation function",
                                                LabeledSpan::new(args[1].span, "try count, sum, average", true),
                                            );
                                            std::process::exit(1);
                                        }
                                    };
                                    ExpressionKind::SlidingWindowAggregation {
                                        expr: inner,
                                        duration: args[0].clone(),
                                        aggregation: window_op,
                                    }
                                }
                                _ => ExpressionKind::Method(inner, name, types, args),
                            };
                            let binop_expr = Expression::new(kind, binop_span);
                            match unop {
                                None => return binop_expr,
                                Some(unop) => {
                                    return Expression::new(ExpressionKind::Unary(unop, Box::new(binop_expr)), span)
                                }
                            }
                        }
                        _ => {
                            handler.error_with_span(
                                &format!("expected method call or tuple access, found {}", rhs),
                                LabeledSpan::new(rhs.span, "unexpected", true),
                            );
                            std::process::exit(1);
                        }
                    }
                }
                Rule::OpeningBracket => {
                    let offset = match rhs.parse_offset() {
                        Ok(offset) => offset,
                        Err(reason) => {
                            handler.error_with_span(
                                "failed to parse offset expression",
                                LabeledSpan::new(rhs.span, &format!("{}", reason), true),
                            );
                            std::process::exit(1);
                        }
                    };
                    match lhs.kind {
                        ExpressionKind::Unary(unop, inner) => {
                            let inner_span = Span { start: inner.span.start, end: rhs.span.end };
                            let new_inner = Expression::new(ExpressionKind::Offset(inner, offset), inner_span);
                            return Expression::new(ExpressionKind::Unary(unop, Box::new(new_inner)), span);
                        }
                        _ => return Expression::new(ExpressionKind::Offset(lhs.into(), offset), span),
                    }
                }
                _ => unreachable!(),
            };
            Expression::new(ExpressionKind::Binary(op, Box::new(lhs), Box::new(rhs)), span)
        },
    )
}

fn build_term_ast(spec: &mut LolaSpec, pair: Pair<'_, Rule>, handler: &Handler) -> Expression {
    let span = pair.as_span();
    match pair.as_rule() {
        // Map function from `Pair` to AST data structure `Expression`
        Rule::Literal => Expression::new(ExpressionKind::Lit(parse_literal(pair)), span.into()),
        Rule::Ident => Expression::new(ExpressionKind::Ident(parse_ident(&pair)), span.into()),
        Rule::ParenthesizedExpression => {
            let mut inner = pair.into_inner();
            let opp = inner
                .next()
                .expect("Rule::ParenthesizedExpression has a token for the (potentialy missing) opening parenthesis");
            let opening_parenthesis = if let Rule::OpeningParenthesis = opp.as_rule() {
                Some(Box::new(Parenthesis::new(opp.as_span().into())))
            } else {
                None
            };

            let inner_expression =
                inner.next().expect("Rule::ParenthesizedExpression has a token for the contained expression");

            let closing = inner
                .next()
                .expect("Rule::ParenthesizedExpression has a token for the (potentialy missing) closing parenthesis");
            let closing_parenthesis = if let Rule::ClosingParenthesis = closing.as_rule() {
                Some(Box::new(Parenthesis::new(closing.as_span().into())))
            } else {
                None
            };

            Expression::new(
                ExpressionKind::ParenthesizedExpression(
                    opening_parenthesis,
                    Box::new(build_expression_ast(spec, inner_expression.into_inner(), handler)),
                    closing_parenthesis,
                ),
                span.into(),
            )
        }
        Rule::UnaryExpr => {
            // First child is the operator, second the operand.
            let mut children = pair.into_inner();
            let pest_operator = children.next().expect("Unary expressions need to have an operator.");
            let operand = children.next().expect("Unary expressions need to have an operand.");
            let operand = build_term_ast(spec, operand, handler);
            let operator = match pest_operator.as_rule() {
                Rule::Add => return operand, // Discard unary plus because it is semantically null.
                Rule::Subtract => UnOp::Neg,
                Rule::Neg => UnOp::Not,
                _ => unreachable!(),
            };
            Expression::new(ExpressionKind::Unary(operator, Box::new(operand)), span.into())
        }
        Rule::TernaryExpr => {
            let mut children = parse_vec_of_expressions(spec, pair.into_inner(), handler);
            assert_eq!(children.len(), 3, "A ternary expression needs exactly three children.");
            Expression::new(
                ExpressionKind::Ite(children.remove(0), children.remove(0), children.remove(0)),
                span.into(),
            )
        }
        Rule::Tuple => {
            let elements = parse_vec_of_expressions(spec, pair.into_inner(), handler);
            assert!(elements.len() != 1, "Tuples may not have exactly one element.");
            Expression::new(ExpressionKind::Tuple(elements), span.into())
        }
        Rule::Expr => build_expression_ast(spec, pair.into_inner(), handler),
        Rule::FunctionExpr => build_function_expression(spec, pair, span.into(), handler),
        Rule::IntegerLiteral => {
            let span = span.into();
            Expression::new(ExpressionKind::Lit(Literal::new_numeric(pair.as_str(), None, span)), span)
        }
        Rule::MissingExpression => {
            let span = span.into();
            Expression::new(ExpressionKind::MissingExpression, span)
        }
        _ => unreachable!("Unexpected rule when parsing expression ast: {:?}", pair.as_rule()),
    }
}

#[derive(Debug, Clone, Eq)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub fn new(name: String, span: Span) -> Ident {
        Ident { name, span }
    }
}

/// In the equality definition of `Ident`, we only compare the string values
/// and ignore the `Span` info
impl PartialEq for Ident {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// Every node in the AST gets a unique id, represented by a 32bit unsiged integer.
/// They are used in the later analysis phases to store information about AST nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u32);

impl NodeId {
    pub fn new(x: usize) -> NodeId {
        assert!(x < (u32::max_value() as usize));
        NodeId(x as u32)
    }

    pub fn from_u32(x: u32) -> NodeId {
        NodeId(x)
    }

    /// When parsing, we initially give all AST nodes this AST node id.
    /// Then later, in the renumber pass, we renumber them to have small, positive ids.
    pub const DUMMY: NodeId = NodeId(!0);
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A span marks a range in a file.
/// Start and end positions are *byte* offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    // TODO Do we need this here or do we want to keep a mapping from byte positions to lines in the LSP part.
    // line: usize,
    // /// The LSP uses UTF-16 code units (2 bytes) as their unit for offsets.
    // lineOffsetLSP: usize,
}

impl Span {
    pub fn unknown() -> Span {
        use std::usize;
        Span { start: usize::max_value(), end: usize::max_value() }
    }
}

impl<'a> From<pest::Span<'a>> for Span {
    fn from(span: pest::Span<'a>) -> Self {
        Span { start: span.start(), end: span.end() }
    }
}

/// A mapper from `Span` to actual source code
#[derive(Debug)]
pub(crate) struct SourceMapper {
    path: PathBuf,
    content: String,
}

#[derive(Debug, Eq, Ord)]
pub(crate) struct CodeLine {
    pub(crate) path: PathBuf,
    pub(crate) line_number: usize,
    pub(crate) column_number: usize,
    pub(crate) line: String,
    pub(crate) highlight: CharSpan,
}

impl PartialEq for CodeLine {
    /// the equality on code lines is given by the equality of the tuples `(path, line_number, column_number)`
    fn eq(&self, rhs: &CodeLine) -> bool {
        (&self.path, self.line_number, self.column_number).eq(&(&rhs.path, rhs.line_number, rhs.column_number))
    }
}

impl PartialOrd for CodeLine {
    /// the partial order on code lines is given by the lexicographic order of `(path, line_number, column_number)`
    fn partial_cmp(&self, rhs: &CodeLine) -> Option<std::cmp::Ordering> {
        (&self.path, self.line_number, self.column_number).partial_cmp(&(&rhs.path, rhs.line_number, rhs.column_number))
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct CharSpan {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

impl SourceMapper {
    pub(crate) fn new(path: PathBuf, content: &str) -> SourceMapper {
        SourceMapper { path, content: content.to_string() }
    }

    #[allow(dead_code)]
    pub(crate) fn get_line(&self, span: Span) -> Option<CodeLine> {
        let mut byte_offset = 0;
        for (num, line) in self.content.split('\n').enumerate() {
            assert!(byte_offset <= span.start);
            let line_end = byte_offset + line.len() + 1; // +1 as it is excluding newline character

            if span.start < line_end {
                if span.end > line_end {
                    // not a single line
                    return None;
                } else {
                    // get column
                    let mut column = 0;
                    let mut start: Option<usize> = None;
                    let mut end: Option<usize> = None;
                    let mut i = 0;
                    for (index, _) in line.char_indices() {
                        i = index;
                        if index < span.start - byte_offset {
                            column += 1;
                        } else if index == span.start - byte_offset {
                            start = Some(index);
                        } else if index == span.end - byte_offset {
                            end = Some(index);
                            break;
                        }
                    }
                    let (start, end) = (
                        start.unwrap_or(i), // we might hit the end of the line/EOI
                        end.unwrap_or(i + 1),
                    );

                    return Some(CodeLine {
                        path: self.path.clone(),
                        line_number: num + 1,
                        column_number: column + 1,
                        line: line.to_string(),
                        highlight: CharSpan { start, end },
                    });
                }
            }
            byte_offset = line_end;
        }
        unreachable!();
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use pest::{consumes_to, parses_to};

    fn cmp_ast_spec(ast: &LolaSpec, spec: &str) -> bool {
        // Todo: Make more robust, e.g. against changes in whitespace.
        assert_eq!(format!("{}", ast), spec);
        true
    }

    #[test]
    fn parse_simple() {
        let _ = LolaParser::parse(Rule::Spec, "input in: Int\noutput out: Int := in\ntrigger in ≠ out")
            .unwrap_or_else(|e| panic!("{}", e));
    }

    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn parse_constant() {
        parses_to! {
            parser: LolaParser,
            input:  "constant five : Int := 5",
            rule:   Rule::ConstantStream,
            tokens: [
                ConstantStream(0, 24, [
                    Ident(9, 13, []),
                    Type(16, 19, [
                        Ident(16, 19, []),
                    ]),
                    Literal(23, 24, [
                        NumberLiteral(23, 24, [
                            NumberLiteralValue(23, 24, [])
                        ]),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn parse_constant_ast() {
        let pair = LolaParser::parse(Rule::ConstantStream, "constant five : Int := 5")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let ast = super::parse_constant(&mut spec, pair);
        assert_eq!(format!("{}", ast), "constant five: Int := 5")
    }

    #[test]
    fn parse_constant_double() {
        let pair = LolaParser::parse(Rule::ConstantStream, "constant fiveoh: Double := 5.0")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let ast = super::parse_constant(&mut spec, pair);
        assert_eq!(format!("{}", ast), "constant fiveoh: Double := 5.0")
    }

    #[test]
    fn parse_input() {
        parses_to! {
            parser: LolaParser,
            input:  "input in: Int",
            rule:   Rule::InputStream,
            tokens: [
                InputStream(0, 13, [
                    Ident(6, 8, []),
                    Type(10, 13, [
                        Ident(10, 13, []),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn parse_input_ast() {
        let pair = LolaParser::parse(Rule::InputStream, "input a: Int, b: Int, c: Bool")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let inputs = super::parse_inputs(&mut spec, pair);
        assert_eq!(inputs.len(), 3);
        assert_eq!(format!("{}", inputs[0]), "input a: Int");
        assert_eq!(format!("{}", inputs[1]), "input b: Int");
        assert_eq!(format!("{}", inputs[2]), "input c: Bool");
    }

    #[test]
    fn build_ast_parameterized_input() {
        let spec = "input in <ab: Int8>: Int8\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn parse_output() {
        parses_to! {
            parser: LolaParser,
            input:  "output out: Int := in + 1",
            rule:   Rule::OutputStream,
            tokens: [
                OutputStream(0, 25, [
                    Ident(7, 10, []),
                    Type(12, 15, [
                        Ident(12, 15, []),
                    ]),
                    Expr(19, 25, [
                        Ident(19, 21, []),
                        Add(22, 23, []),
                        Literal(24, 25, [
                            NumberLiteral(24, 25, [
                                NumberLiteralValue(24, 25, [])
                            ]),
                        ]),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn parse_output_ast() {
        let spec = "output out: Int := in + 1";
        let pair = LolaParser::parse(Rule::OutputStream, spec).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let mut lola = LolaSpec::new();
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = super::parse_output(&mut lola, pair, &handler);
        assert_eq!(format!("{}", ast), spec)
    }

    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn parse_trigger() {
        parses_to! {
            parser: LolaParser,
            input:  "trigger in != out \"some message\"",
            rule:   Rule::Trigger,
            tokens: [
                Trigger(0, 32, [
                    Expr(8, 17, [
                        Ident(8, 10, []),
                        NotEqual(11, 13, []),
                        Ident(14, 17, []),
                    ]),
                    String(19, 31, []),
                ]),
            ]
        };
    }

    #[test]
    fn parse_trigger_ast() {
        let spec = "trigger in ≠ out \"some message\"";
        let pair = LolaParser::parse(Rule::Trigger, spec).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let mut lola = LolaSpec::new();
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = super::parse_trigger(&mut lola, pair, &handler);
        assert_eq!(format!("{}", ast), "trigger in ≠ out \"some message\"")
    }

    #[test]
    fn parse_expression() {
        let content = "in + 1";
        let expr = LolaParser::parse(Rule::Expr, content).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let mut spec = LolaSpec::new();
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let ast = build_expression_ast(&mut spec, expr.into_inner(), &handler);
        assert_eq!(format!("{}", ast), content)
    }

    #[test]
    fn parse_expression_precedence() {
        let content = "(a ∨ b ∧ c)";
        let expr = LolaParser::parse(Rule::Expr, content).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let mut spec = LolaSpec::new();
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let ast = build_expression_ast(&mut spec, expr.into_inner(), &handler);
        assert_eq!(format!("{}", ast), content)
    }

    #[test]
    fn parse_missing_closing_parenthesis() {
        let content = "(a ∨ b ∧ c";
        let expr = LolaParser::parse(Rule::Expr, content).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let mut spec = LolaSpec::new();
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let ast = build_expression_ast(&mut spec, expr.into_inner(), &handler);
        assert_eq!(format!("{}", ast), content)
    }

    #[test]
    fn build_simple_ast() {
        let spec = "input in: Int\noutput out: Int := in\ntrigger in ≠ out\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_ast_input() {
        let spec = "input in: Int\ninput in2: Int\ninput in3: (Int, Bool)\ninput in4: Bool\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_parenthesized_expression() {
        let spec = "output s: Bool := (true ∨ true)\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_lookup_expression_default() {
        let spec = "output s: Int := s.offset(by: -1).defaults(to: (3 * 4))\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_lookup_expression_hold() {
        let spec = "output s: Int := s.offset(by: -1).hold().defaults(to: 3 * 4)\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_ternary_expression() {
        let spec = "input in: Int\noutput s: Int := if in = 3 then 4 else in + 2\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_function_expression() {
        let spec = "input in: (Int, Bool)\noutput s: Int := nroot(1, sin(1, in))\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_trigger() {
        let spec = "input in: Int\ntrigger in > 5\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_complex_expression() {
        let spec =
            "output s: Double := if !((s.offset(by: -1).defaults(to: (3 * 4)) + -4) = 12) ∨ true = false then 2.0 else 4.1\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_type_declaration() {
        let spec = "type VerifiedUser { name: String }\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_parameter_list() {
        let spec = "output s <a: B, c: D>: E := 3\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_template_spec() {
        let spec = "output s: Int { invoke inp unless 3 > 5 extend b terminate false } := 3\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        // 0.5GHz correspond to 2ns.
        let spec = spec.replace("0.5GHz", "2ns");
        cmp_ast_spec(&ast, spec.as_str());
    }

    #[test]
    fn build_tuple_expression() {
        let spec = "input in: (Int, Bool)\noutput s: Int := (1, in.0).1\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_string() {
        let spec = r#"constant s: String := "a string with \n newline"
"#;
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_raw_string() {
        let spec = r##"constant s: String := r#"a raw \ string that " needs padding"#
"##;
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_import() {
        let spec = "import math\ninput in: UInt8\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_method_call() {
        let spec = "output count := count.offset(-1).default(0) + 1\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_method_call_with_param() {
        let spec = "output count := count.offset<Int8>(-1).default(0) + 1\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_realtime_offset() {
        let spec = "output a := b.offset(by: -1s)\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_function_argument_name() {
        let spec = "output a := b.hold().defaults(to: 0)\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn parse_precedence_not_regression() {
        parses_to! {
            parser: LolaParser,
            input:  "!(fast | false) & fast",
            rule:   Rule::Expr,
            tokens: [
                Expr(0, 22, [
                    UnaryExpr(0, 15, [
                        Neg(0, 1, []),
                        ParenthesizedExpression(1, 15, [
                            OpeningParenthesis(1, 2, []),
                            Expr(2, 14, [
                                Ident(2, 6, []),
                                Or(7, 9, []),
                                Literal(9, 14, [
                                    False(9, 14, [])
                                ])
                            ]),
                            ClosingParenthesis(14, 15, [])
                        ])
                    ]),
                    And(16, 18, []),
                    Ident(18, 22, [])
                ]),
            ]
        };
    }
}
