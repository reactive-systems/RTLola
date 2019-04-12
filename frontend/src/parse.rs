//! This module contains the parser for the Lola Language.

use super::ast::*;
use crate::analysis::graph_based_analysis::space_requirements::dur_as_nanos;
use lazy_static::lazy_static;
use num::{BigInt, BigRational, FromPrimitive, One, Signed, ToPrimitive};
use pest;
use pest::iterators::{Pair, Pairs};
use pest::prec_climber::{Assoc, Operator, PrecClimber};
use pest::Parser;
use pest_derive::Parser;
use std::path::PathBuf;
use std::time::Duration;

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
            Operator::new(Or, Left),
            Operator::new(And, Left),
            Operator::new(Equal, Left) | Operator::new(NotEqual, Left),
            Operator::new(LessThan, Left) | Operator::new(LessThanOrEqual, Left) | Operator::new(MoreThan, Left) | Operator::new(MoreThanOrEqual, Left),
            Operator::new(Add, Left) | Operator::new(Subtract, Left),
            Operator::new(Multiply, Left) | Operator::new(Divide, Left) | Operator::new(Mod, Left),
            Operator::new(Power, Right),
            Operator::new(Hold, Left),
            Operator::new(Default, Left),
            Operator::new(Dot, Left),
        ])
    };
}

/**
 * Transforms a textual representation of a Lola specification into
 * an AST representation.
 */
pub(crate) fn parse(content: &str) -> Result<LolaSpec, pest::error::Error<Rule>> {
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
                let output = parse_output(&mut spec, pair);
                spec.outputs.push(output);
            }
            Rule::Trigger => {
                let trigger = parse_trigger(&mut spec, pair);
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
fn parse_output(spec: &mut LolaSpec, pair: Pair<'_, Rule>) -> Output {
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

    let ty;
    if let Rule::Type = pair.as_rule() {
        ty = parse_type(spec, pair);
        pair = pairs.next().expect("mismatch between grammar and AST");
    } else {
        ty = Type::new_inferred();
    }

    let mut tspec = None;
    if let Rule::TemplateSpec = pair.as_rule() {
        tspec = Some(parse_template_spec(spec, pair));
        pair = pairs.next().expect("mismatch between grammar and AST");
    };

    // Parse expression
    let expr_span = pair.as_span();
    let expression = build_expression_ast(spec, pair.into_inner(), expr_span.into());
    Output { id: NodeId::DUMMY, name, ty, params, template_spec: tspec, expression, span }
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

fn parse_template_spec(spec: &mut LolaSpec, pair: Pair<'_, Rule>) -> TemplateSpec {
    let span = pair.as_span().into();
    let mut decls = pair.into_inner();
    let mut pair = decls.next();
    let mut rule = pair.as_ref().map(|p| p.as_rule());

    let mut inv_spec = None;
    if let Some(Rule::InvokeDecl) = rule {
        inv_spec = Some(parse_inv_spec(spec, pair.unwrap()));
        pair = decls.next();
        rule = pair.as_ref().map(|p| p.as_rule());
    }
    let mut ext_spec = None;
    if let Some(Rule::ExtendDecl) = rule {
        ext_spec = Some(parse_ext_spec(spec, pair.unwrap()));
        pair = decls.next();
        rule = pair.as_ref().map(|p| p.as_rule());
    }
    let mut ter_spec = None;
    if let Some(Rule::TerminateDecl) = rule {
        let exp = pair.unwrap();
        let span_ter = exp.as_span().into();
        let expr = exp.into_inner().next().expect("mismatch between grammar and AST");
        let expr_span = expr.as_span().into();
        let expr = build_expression_ast(spec, expr.into_inner(), expr_span);
        ter_spec = Some(TerminateSpec { target: expr, id: NodeId::DUMMY, span: span_ter });
    }
    TemplateSpec { inv: inv_spec, ext: ext_spec, ter: ter_spec, id: NodeId::DUMMY, span }
}

pub(crate) fn build_time_spec(expr: Expression, unit_str: &str, span: Span) -> TimeSpec {
    let (factor, invert): (BigRational, bool) = match unit_str {
        "ns" => (BigRational::from_u64(1_u64).unwrap(), false),
        "μs" | "us" => (BigRational::from_u64(10_u64.pow(3)).unwrap(), false),
        "ms" => (BigRational::from_u64(10_u64.pow(6)).unwrap(), false),
        "s" => (BigRational::from_u64(10_u64.pow(9)).unwrap(), false),
        "min" => (BigRational::from_u64(10_u64.pow(9) * 60).unwrap(), false),
        "h" => (BigRational::from_u64(10_u64.pow(9) * 60 * 60).unwrap(), false),
        "d" => (BigRational::from_u64(10_u64.pow(9) * 60 * 60 * 24).unwrap(), false),
        "w" => (BigRational::from_u64(10_u64.pow(9) * 60 * 60 * 24 * 7).unwrap(), false),
        "a" => (BigRational::from_u64(10_u64.pow(9) * 60 * 60 * 24 * 365).unwrap(), false),
        "μHz" | "uHz" => (BigRational::from_u64(10_u64.pow(15)).unwrap(), true),
        "mHz" => (BigRational::from_u64(10_u64.pow(12)).unwrap(), true),
        "Hz" => (BigRational::from_u64(10_u64.pow(9)).unwrap(), true),
        "kHz" => (BigRational::from_u64(10_u64.pow(6)).unwrap(), true),
        "MHz" => (BigRational::from_u64(10_u64.pow(3)).unwrap(), true),
        "GHz" => (BigRational::from_u64(1).unwrap(), true),
        _ => unreachable!(),
    };

    if let ExpressionKind::Lit(l) = expr.kind {
        match l.kind {
            LitKind::Int(i) => {
                let mut period: BigRational = BigRational::from_integer(BigInt::from(i));
                if invert {
                    period = num::BigRational::one() / period;
                }
                period *= factor;
                if period.is_negative() {
                    let rounded_period: Duration = Duration::from_nanos(
                        (-period.clone()).to_integer().to_u64().expect("Period [ns] too large for u64!"),
                    );
                    TimeSpec {
                        period: rounded_period,
                        exact_period: period,
                        signum: match dur_as_nanos(rounded_period) {
                            0 => 0,
                            _ => -1,
                        },
                        id: NodeId::DUMMY,
                        span,
                    }
                } else {
                    let rounded_period: Duration =
                        Duration::from_nanos(period.to_integer().to_u64().expect("Period [ns] too large for u64!"));
                    TimeSpec {
                        period: rounded_period,
                        exact_period: period,
                        signum: match dur_as_nanos(rounded_period) {
                            0 => 0,
                            _ => 1,
                        },
                        id: NodeId::DUMMY,
                        span,
                    }
                }
            }
            LitKind::Float(_, precise) => {
                let mut period: BigRational = precise.clone();
                if invert {
                    period = BigRational::one() / period;
                }
                period *= factor;
                if period.is_negative() {
                    let rounded_period: Duration = Duration::from_nanos(
                        (-period.clone()).to_integer().to_u64().expect("Period [ns] too large for u64!"),
                    );
                    TimeSpec {
                        period: rounded_period,
                        exact_period: period,
                        signum: match dur_as_nanos(rounded_period) {
                            0 => 0,
                            _ => -1,
                        },
                        id: NodeId::DUMMY,
                        span,
                    }
                } else {
                    let rounded_period: Duration =
                        Duration::from_nanos(period.to_integer().to_u64().expect("Period [ns] too large for u64!"));
                    TimeSpec {
                        period: rounded_period,
                        exact_period: period,
                        signum: match dur_as_nanos(rounded_period) {
                            0 => 0,
                            _ => 1,
                        },
                        id: NodeId::DUMMY,
                        span,
                    }
                }
            }
            _ => panic!("Needs to be numeric!"),
        }
    } else {
        panic!("Expression needs to be a literal. ")
    }
}

fn parse_frequency(spec: &mut LolaSpec, freq: Pair<'_, Rule>) -> TimeSpec {
    let freq_rule = freq.as_rule();
    let freq_span = freq.as_span().into();
    let mut children = freq.into_inner();
    let expr = children.next().expect("mismatch between grammar and AST");
    let span = expr.as_span().into();
    let expr = build_expression_ast(spec, expr.into_inner(), span);
    let unit_pair = children.next().expect("mismatch between grammar and AST");
    let unit_str = unit_pair.as_str();
    match freq_rule {
        Rule::Frequency => {
            assert_eq!(unit_pair.as_rule(), Rule::UnitOfFreq);
            build_time_spec(expr, unit_str, freq_span)
        }
        Rule::Duration => {
            assert_eq!(unit_pair.as_rule(), Rule::UnitOfTime);
            build_time_spec(expr, unit_str, freq_span)
        }
        _ => unreachable!(),
    }
}

fn parse_ext_spec(spec: &mut LolaSpec, ext_pair: Pair<'_, Rule>) -> ExtendSpec {
    let span_ext = ext_pair.as_span().into();
    let mut children = ext_pair.into_inner();

    let first_child = children.next().expect("mismatch between grammar and ast");
    let mut freq = None;
    let mut target = None;

    match first_child.as_rule() {
        Rule::Frequency | Rule::Duration => freq = Some(parse_frequency(spec, first_child)),
        Rule::Expr => {
            let span = first_child.as_span().into();
            target = Some(build_expression_ast(spec, first_child.into_inner(), span));
            if let Some(freq_pair) = children.next() {
                freq = Some(parse_frequency(spec, freq_pair));
            }
        }
        _ => unreachable!(),
    }
    assert!(freq.is_some() || target.is_some());
    ExtendSpec { target, freq, id: NodeId::DUMMY, span: span_ext }
}

fn parse_inv_spec(spec: &mut LolaSpec, inv_pair: Pair<'_, Rule>) -> InvokeSpec {
    let span_inv = inv_pair.as_span().into();
    let mut inv_children = inv_pair.into_inner();
    let expr_pair = inv_children.next().expect("mismatch between grammar and AST");
    let expr_span = expr_pair.as_span().into();
    let inv_target = build_expression_ast(spec, expr_pair.into_inner(), expr_span);
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
        let cond_expr_span = condition.as_span().into();
        cond_expr = Some(build_expression_ast(spec, condition.into_inner(), cond_expr_span))
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
fn parse_trigger(spec: &mut LolaSpec, pair: Pair<'_, Rule>) -> Trigger {
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
    let expr_span = pair.as_span();
    let expression = build_expression_ast(spec, pair.into_inner(), expr_span.into());

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

fn parse_rational(repr: &str) -> BigRational {
    // precondition: repr is a valid floating point literal
    assert!(repr.parse::<f64>().is_ok());

    let mut value: BigRational = num::Zero::zero();
    let mut char_indices = repr.char_indices();
    let mut negated = false;

    let ten = num::BigRational::from_i64(10).unwrap();
    let zero: BigRational = num::Zero::zero();
    let one = num::BigRational::from_i64(1).unwrap();
    let two = num::BigRational::from_i64(2).unwrap();
    let three = num::BigRational::from_i64(3).unwrap();
    let four = num::BigRational::from_i64(4).unwrap();
    let five = num::BigRational::from_i64(5).unwrap();
    let six = num::BigRational::from_i64(6).unwrap();
    let seven = num::BigRational::from_i64(7).unwrap();
    let eight = num::BigRational::from_i64(8).unwrap();
    let nine = num::BigRational::from_i64(9).unwrap();

    let mut contains_fractional = false;
    let mut contains_exponent = false;

    //parse the before the point/exponent

    loop {
        match char_indices.next() {
            Some((_, '+')) => {}
            Some((_, '-')) => {
                negated = true;
            }
            Some((_, '.')) => {
                contains_fractional = true;
                break;
            }
            Some((_, 'e')) => {
                contains_exponent = true;
                break;
            }
            Some((_, '0')) => {
                value *= &ten;
            }
            Some((_, '1')) => {
                value *= &ten;
                value += &one;
            }
            Some((_, '2')) => {
                value *= &ten;
                value += &two;
            }
            Some((_, '3')) => {
                value *= &ten;
                value += &three;
            }
            Some((_, '4')) => {
                value *= &ten;
                value += &four;
            }
            Some((_, '5')) => {
                value *= &ten;
                value += &five;
            }
            Some((_, '6')) => {
                value *= &ten;
                value += &six;
            }
            Some((_, '7')) => {
                value *= &ten;
                value += &seven;
            }
            Some((_, '8')) => {
                value *= &ten;
                value += &eight;
            }
            Some((_, '9')) => {
                value *= &ten;
                value += &nine;
            }
            Some((_, _)) => unreachable!(),
            None => {
                break;
            }
        }
    }

    if contains_fractional {
        let mut number_of_fractional_positions: BigRational = zero.clone();
        loop {
            match char_indices.next() {
                Some((_, 'e')) => {
                    contains_exponent = true;
                    break;
                }
                Some((_, '0')) => {
                    value *= &ten;
                    number_of_fractional_positions += &one;
                }
                Some((_, '1')) => {
                    value *= &ten;
                    value += &one;
                    number_of_fractional_positions += &one;
                }
                Some((_, '2')) => {
                    value *= &ten;
                    value += &two;
                    number_of_fractional_positions += &one;
                }
                Some((_, '3')) => {
                    value *= &ten;
                    value += &three;
                    number_of_fractional_positions += &one;
                }
                Some((_, '4')) => {
                    value *= &ten;
                    value += &four;
                    number_of_fractional_positions += &one;
                }
                Some((_, '5')) => {
                    value *= &ten;
                    value += &five;
                    number_of_fractional_positions += &one;
                }
                Some((_, '6')) => {
                    value *= &ten;
                    value += &six;
                    number_of_fractional_positions += &one;
                }
                Some((_, '7')) => {
                    value *= &ten;
                    value += &seven;
                    number_of_fractional_positions += &one;
                }
                Some((_, '8')) => {
                    value *= &ten;
                    value += &eight;
                    number_of_fractional_positions += &one;
                }
                Some((_, '9')) => {
                    value *= &ten;
                    value += &nine;
                    number_of_fractional_positions += &one;
                }
                Some((_, _)) => unreachable!(),
                None => {
                    break;
                }
            }
        }
        while number_of_fractional_positions > zero {
            value /= &ten;
            number_of_fractional_positions -= &one;
        }
    }

    if contains_exponent {
        let mut negated_exponent = false;
        let mut exponent: BigRational = zero.clone();
        loop {
            match char_indices.next() {
                Some((_, '+')) => {}
                Some((_, '-')) => {
                    negated_exponent = true;
                }
                Some((_, '0')) => {
                    exponent *= &ten;
                }
                Some((_, '1')) => {
                    exponent *= &ten;
                    exponent += &one;
                }
                Some((_, '2')) => {
                    exponent *= &ten;
                    exponent += &two;
                }
                Some((_, '3')) => {
                    exponent *= &ten;
                    exponent += &three;
                }
                Some((_, '4')) => {
                    exponent *= &ten;
                    exponent += &four;
                }
                Some((_, '5')) => {
                    exponent *= &ten;
                    exponent += &five;
                }
                Some((_, '6')) => {
                    exponent *= &ten;
                    exponent += &six;
                }
                Some((_, '7')) => {
                    exponent *= &ten;
                    exponent += &seven;
                }
                Some((_, '8')) => {
                    exponent *= &ten;
                    exponent += &eight;
                }
                Some((_, '9')) => {
                    exponent *= &ten;
                    exponent += &nine;
                }
                Some((_, _)) => unreachable!(),
                None => {
                    break;
                }
            }
        }
        let mut new_value = value.clone();
        if negated_exponent {
            while exponent > zero {
                new_value /= &ten;
                exponent -= &one;
            }
        } else {
            while exponent > zero {
                new_value *= &ten;
                exponent -= &one;
            }
        }
        value = new_value;
    }
    if negated {
        value = -value;
    }

    value
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
            let str_rep: &str = inner.as_str();

            if let Result::Ok(i) = str_rep.parse::<i128>() {
                return Literal::new_int(i, inner.as_span().into());
            } else if let Result::Ok(f) = str_rep.parse::<f64>() {
                let ratio = parse_rational(str_rep);
                return Literal::new_float(f, ratio, inner.as_span().into());
            } else {
                unreachable!();
            }
        }
        Rule::True => Literal::new_bool(true, inner.as_span().into()),
        Rule::False => Literal::new_bool(false, inner.as_span().into()),
        _ => unreachable!(),
    }
}

fn parse_stream_instance(spec: &mut LolaSpec, instance: Pair<'_, Rule>) -> StreamInstance {
    let span = instance.as_span().into();
    let mut children = instance.into_inner();
    // Parse the stream identifier in isolation.
    let stream_ident = parse_ident(&children.next().unwrap());
    // Parse remaining children, aka the arguments.
    let args = parse_vec_of_expressions(spec, children);
    StreamInstance { stream_identifier: stream_ident, arguments: args, id: NodeId::DUMMY, span }
}

#[allow(clippy::vec_box)]
fn parse_vec_of_expressions(spec: &mut LolaSpec, pairs: Pairs<'_, Rule>) -> Vec<Box<Expression>> {
    pairs
        .map(|expr| {
            let span = expr.as_span().into();
            build_expression_ast(spec, expr.into_inner(), span)
        })
        .map(Box::new)
        .collect()
}

fn parse_vec_of_types(spec: &mut LolaSpec, pairs: Pairs<'_, Rule>) -> Vec<Type> {
    pairs.map(|expr| parse_type(spec, expr)).collect()
}

fn parse_lookup_expression(spec: &mut LolaSpec, pair: Pair<'_, Rule>, span: Span) -> Expression {
    let mut children = pair.into_inner();
    let stream_instance = children.next().expect("Lookups need to have a target stream instance.");
    let stream_instance = parse_stream_instance(spec, stream_instance);
    let second_child = children.next().unwrap();
    let second_child_span = second_child.as_span();
    match second_child.as_rule() {
        Rule::Expr => {
            // Discrete offset
            let offset = build_expression_ast(spec, second_child.into_inner(), second_child_span.into());
            let offset = Offset::DiscreteOffset(Box::new(offset));
            Expression::new(ExpressionKind::Lookup(stream_instance, offset, None), span)
        }
        Rule::Duration => {
            // Real time offset
            let duration_span = second_child.as_span().into();
            let mut duration_children = second_child.into_inner();
            let time_interval = duration_children.next().expect("Duration needs a time span.");
            let time_interval_span = time_interval.as_span().into();
            let val = time_interval.as_str().parse().expect("number literal can be parsed into integer");
            let unit_string = duration_children.next().expect("Duration needs a time unit.").as_str();
            let duration_value =
                Expression::new(ExpressionKind::Lit(Literal::new_int(val, duration_span)), time_interval_span);
            let time_spec = build_time_spec(duration_value, unit_string, time_interval_span);
            let offset = Offset::RealTimeOffset(time_spec);
            // Now check whether it is a window or not.
            let aggregation = match children.next().map(|x| x.as_rule()) {
                Some(Rule::Sum) => Some(WindowOperation::Sum),
                Some(Rule::Product) => Some(WindowOperation::Product),
                Some(Rule::Average) => Some(WindowOperation::Average),
                Some(Rule::Count) => Some(WindowOperation::Count),
                Some(Rule::Integral) => Some(WindowOperation::Integral),
                None => None,
                _ => unreachable!(),
            };
            Expression::new(ExpressionKind::Lookup(stream_instance, offset, aggregation), span)
        }
        _ => unreachable!(),
    }
}

fn build_function_expression(spec: &mut LolaSpec, pair: Pair<'_, Rule>, span: Span) -> Expression {
    let mut children = pair.into_inner();
    let ident = parse_ident(&children.next().unwrap());
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
    let args = parse_vec_of_expressions(spec, next.into_inner());
    Expression::new(ExpressionKind::Function(ident, type_params, args), span)
}

/**
 * Builds the Expr AST.
 */
fn build_expression_ast(spec: &mut LolaSpec, pairs: Pairs<'_, Rule>, span: Span) -> Expression {
    PREC_CLIMBER.climb(
        pairs,
        |pair: Pair<'_, Rule>| build_term_ast(spec, pair),
        |lhs: Expression, op: Pair<'_, Rule>, rhs: Expression| {
            // Reduce function combining `Expression`s to `Expression`s with the correct precs
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
                Rule::Dot => {
                    match rhs.kind {
                        // access to a tuple
                        ExpressionKind::Lit(l) => {
                            let ident = match l.kind {
                                LitKind::Int(i) => {
                                    assert!(i >= 0); // TODO check this and otherwise give an error
                                    Ident::new(format!("{}", i), l.span)
                                }
                                _ => {
                                    panic!("expected unsigned integer, found {}", l);
                                }
                            };
                            return Expression::new(ExpressionKind::Field(Box::new(lhs), ident), span);
                        }
                        ExpressionKind::Function(ident, types, args) => {
                            return Expression::new(ExpressionKind::Method(Box::new(lhs), ident, types, args), span);
                        }
                        _ => panic!("tuple accesses require a number"),
                    }
                }
                Rule::Default => return Expression::new(ExpressionKind::Default(Box::new(lhs), Box::new(rhs)), span),
                Rule::Hold => return Expression::new(ExpressionKind::Hold(Box::new(lhs), Box::new(rhs)), span),
                _ => unreachable!(),
            };
            Expression::new(ExpressionKind::Binary(op, Box::new(lhs), Box::new(rhs)), span)
        },
    )
}

fn build_term_ast(spec: &mut LolaSpec, pair: Pair<'_, Rule>) -> Expression {
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

            let inner_span = inner_expression.as_span().into();
            Expression::new(
                ExpressionKind::ParenthesizedExpression(
                    opening_parenthesis,
                    Box::new(build_expression_ast(spec, inner_expression.into_inner(), inner_span)),
                    closing_parenthesis,
                ),
                span.into(),
            )
        }
        Rule::LookupExpr => parse_lookup_expression(spec, pair, span.into()),
        Rule::UnaryExpr => {
            // First child is the operator, second the operand.
            let mut children = pair.into_inner();
            let pest_operator = children.next().expect("Unary expressions need to have an operator.");
            let operand = children.next().expect("Unary expressions need to have an operand.");
            let operand = build_term_ast(spec, operand);
            let operator = match pest_operator.as_rule() {
                Rule::Add => return operand, // Discard unary plus because it is semantically null.
                Rule::Subtract => UnOp::Neg,
                Rule::Neg => UnOp::Not,
                _ => unreachable!(),
            };
            Expression::new(ExpressionKind::Unary(operator, Box::new(operand)), span.into())
        }
        Rule::TernaryExpr => {
            let mut children = parse_vec_of_expressions(spec, pair.into_inner());
            assert_eq!(children.len(), 3, "A ternary expression needs exactly three children.");
            Expression::new(
                ExpressionKind::Ite(children.remove(0), children.remove(0), children.remove(0)),
                span.into(),
            )
        }
        Rule::Tuple => {
            let elements = parse_vec_of_expressions(spec, pair.into_inner());
            assert!(elements.len() != 1, "Tuples may not have exactly one element.");
            Expression::new(ExpressionKind::Tuple(elements), span.into())
        }
        Rule::Expr => {
            let span = pair.as_span();
            build_expression_ast(spec, pair.into_inner(), span.into())
        }
        Rule::FunctionExpr => build_function_expression(spec, pair, span.into()),
        Rule::IntegerLiteral => {
            let span = span.into();
            Expression::new(ExpressionKind::Lit(Literal::new_int(pair.as_str().parse().unwrap(), span)), span)
        }
        Rule::MissingExpression => {
            let span = span.into();
            Expression::new(ExpressionKind::MissingExpression, span)
        }
        _ => panic!("Unexpected rule when parsing expression ast: {:?}", pair.as_rule()),
    }
}

#[derive(Debug, Clone)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub fn new(name: String, span: Span) -> Ident {
        Ident { name, span }
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

    #[allow(clippy::cyclomatic_complexity)]
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
                        NumberLiteral(23, 24, []),
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
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[allow(clippy::cyclomatic_complexity)]
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
                            NumberLiteral(24, 25, []),
                        ]),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn parse_output_ast() {
        let pair = LolaParser::parse(Rule::OutputStream, "output out: Int := in + 1")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let ast = super::parse_output(&mut spec, pair);
        assert_eq!(format!("{}", ast), "output out: Int := in + 1")
    }

    #[allow(clippy::cyclomatic_complexity)]
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
        let pair = LolaParser::parse(Rule::Trigger, "trigger in ≠ out \"some message\"")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let ast = super::parse_trigger(&mut spec, pair);
        assert_eq!(format!("{}", ast), "trigger in ≠ out \"some message\"")
    }

    #[test]
    fn parse_expression() {
        let expr = LolaParser::parse(Rule::Expr, "in + 1\n").unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let mut spec = LolaSpec::new();
        let span = expr.as_span();
        let ast = build_expression_ast(&mut spec, expr.into_inner(), span.into());
        assert_eq!(format!("{}", ast), "in + 1")
    }

    #[test]
    fn parse_expression_precedence() {
        let expr = LolaParser::parse(Rule::Expr, "(a ∨ b ∧ c)").unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let mut spec = LolaSpec::new();
        let span = expr.as_span();
        let ast = build_expression_ast(&mut spec, expr.into_inner(), span.into());
        assert_eq!(format!("{}", ast), "(a ∨ b ∧ c)")
    }

    #[test]
    fn parse_missing_closing_parenthesis() {
        let expr = LolaParser::parse(Rule::Expr, "(a ∨ b ∧ c").unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let mut spec = LolaSpec::new();
        let span = expr.as_span();
        let ast = build_expression_ast(&mut spec, expr.into_inner(), span.into());
        assert_eq!(format!("{}", ast), "(a ∨ b ∧ c")
    }

    #[test]
    fn build_simple_ast() {
        let spec = "input in: Int\noutput out: Int := in\ntrigger in ≠ out\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_ast_input() {
        let spec = "input in: Int\ninput in2: Int\ninput in3: (Int, Bool)\ninput in4: Bool\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_parenthesized_expression() {
        let spec = "output s: Bool := (true ∨ true)\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_lookup_expression_default() {
        let spec = "output s: Int := s[-1] ? (3 * 4)\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_lookup_expression_hold() {
        let spec = "output s: Int := s[-1] ! (3 * 4)\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_ternary_expression() {
        let spec = "input in: Int\noutput s: Int := if in = 3 then 4 else in + 2\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_function_expression() {
        let spec = "input in: (Int, Bool)\noutput s: Int := nroot(1, sin(1, in))\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_trigger() {
        let spec = "input in: Int\ntrigger in > 5\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_complex_expression() {
        let spec = "output s: Double := if !((s[-1] ? (3 * 4) + -4) = 12) ∨ true = false then 2.0 else 4.1\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_type_declaration() {
        let spec = "type VerifiedUser { name: String }\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_parameter_list() {
        let spec = "output s <a: B, c: D>: E := 3\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_template_spec() {
        let spec = "output s: Int { invoke inp unless 3 > 5 extend b @ 0.5GHz terminate false } := 3\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        // 0.5GHz correspond to 2ns.
        let spec = spec.replace("0.5GHz", "2ns");
        cmp_ast_spec(&ast, spec.as_str());
    }

    #[test]
    fn build_tuple_expression() {
        let spec = "input in: (Int, Bool)\noutput s: Int := (1, in.0).1\n";
        let throw = |e| panic!("{}", e);
        let ast = parse(spec).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_string() {
        let spec = r#"constant s: String := "a string with \n newline"
"#;
        let ast = parse(spec).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_raw_string() {
        let spec = r##"constant s: String := r#"a raw \ string that " needs padding"#
"##;
        let ast = parse(spec).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_import() {
        let spec = "import math\ninput in: UInt8\n";
        let ast = parse(spec).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_method_call() {
        let spec = "output count := count.offset(-1).default(0) + 1\n";
        let ast = parse(spec).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_method_call_with_param() {
        let spec = "output count := count.offset<Int8>(-1).default(0) + 1\n";
        let ast = parse(spec).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_realtime_offset() {
        let spec = "output a := b[-1s]\n";
        let ast = parse(spec).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    fn time_spec_int(val: i128, unit: &str) -> Duration {
        build_time_spec(
            Expression::new(ExpressionKind::Lit(Literal::new_int(val, Span::unknown())), Span::unknown()),
            unit,
            Span::unknown(),
        )
        .period
    }

    #[test]
    fn test_time_spec_to_duration_conversion() {
        assert_eq!(time_spec_int(1, "s"), Duration::new(1, 0));
        assert_eq!(time_spec_int(2, "min"), Duration::new(2 * 60, 0));
        assert_eq!(time_spec_int(33, "h"), Duration::new(33 * 60 * 60, 0));
        assert_eq!(time_spec_int(12354, "ns"), Duration::new(0, 12354));
        assert_eq!(time_spec_int(90351, "us"), Duration::new(0, 90351 * 1_000));
        assert_eq!(time_spec_int(248, "ms"), Duration::new(0, 248 * 1_000_000));
        assert_eq!(time_spec_int(29_489_232, "ms"), Duration::new(29_489, 232 * 1_000_000));
    }

    #[test]
    fn test_frequency_to_duration_conversion() {
        assert_eq!(time_spec_int(1, "Hz"), Duration::new(1, 0));
        assert_eq!(time_spec_int(10, "Hz"), Duration::new(0, 100_000_000));
        assert_eq!(time_spec_int(400, "uHz"), Duration::new(2_500, 0));
        assert_eq!(time_spec_int(20, "mHz"), Duration::new(50, 0));
    }

    #[test]
    fn parse_precedence_not_regression() {
        parses_to! {
            parser: LolaParser,
            input:  "!(fast[-1] ? false) & fast",
            rule:   Rule::Expr,
            tokens: [
                Expr(0, 26, [
                    UnaryExpr(0, 19, [
                        Neg(0, 1, []),
                        ParenthesizedExpression(1, 19, [
                            OpeningParenthesis(1, 2, []),
                            Expr(2, 18, [
                                LookupExpr(2, 10, [
                                    StreamInstance(2, 6, [
                                        Ident(2, 6, [])
                                    ]),
                                    Expr(7, 9, [
                                        Literal(7, 9, [
                                            NumberLiteral(7, 9, [])
                                        ])
                                    ]),
                                ]),
                                Default(11, 12, []),
                                Literal(13, 18, [
                                    False(13, 18, [])
                                ])
                            ]),
                            ClosingParenthesis(18, 19, [])
                        ])
                    ]),
                    And(20, 22, []),
                    Ident(22, 26, [])
                ]),
            ]
        };
    }
}
