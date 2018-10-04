//! This module contains the parser for the Lola Language.

use ast::*;
use pest;
use pest::iterators::{Pair, Pairs};
use pest::prec_climber::{Assoc, Operator, PrecClimber};
use pest::Parser;
use std::collections::HashMap;

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
        ])
    };
}

/**
 * Transforms a textual representation of a Lola specification into
 * an AST representation.
 */
fn parse(content: &str) -> Result<LolaSpec, pest::error::Error<Rule>> {
    let pairs = LolaParser::parse(Rule::Spec, content)?;
    let mut spec = LolaSpec::new();
    for pair in pairs {
        match pair.as_rule() {
            Rule::LanguageSpec => {
                spec.language = Some(LanguageSpec::from(pair.as_str()));
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
            Rule::Trigger => unimplemented!(),
            _ => unreachable!(),
        }
    }
    Ok(spec)
}

/**
 * Transforms a `Rule::ConstantStrean` into `Constant` AST node.
 * Panics if input is not `Rule::ConstantStrean`.
 * The constant rule consists of the following tokens:
 * - Rule::Ident
 * - Rule::Type
 * - Rule::Literal
 */
fn parse_constant(spec: &mut LolaSpec, pair: Pair<Rule>) -> Constant {
    assert_eq!(pair.as_rule(), Rule::ConstantStream);
    let span = pair.as_span().into();
    let mut pairs = pair.into_inner();
    let name = parse_ident(
        spec,
        pairs.next().expect("mismatch between grammar and AST"),
    );
    let ty = parse_type(
        spec,
        pairs.next().expect("mismatch between grammar and AST"),
    );
    let literal = parse_literal(
        spec,
        pairs.next().expect("mismatch between grammar and AST"),
    );
    Constant {
        name,
        ty,
        literal,
        span,
    }
}

/**
 * Transforms a `Rule::InputStrean` into `Input` AST node.
 * Panics if input is not `Rule::InputStrean`.
 * The input rule consists of non-empty sequences of following tokens:
 * - Rule::Ident
 * - (Rule::ParamList)?
 * - Rule::Type
 */
fn parse_inputs(spec: &mut LolaSpec, pair: Pair<Rule>) -> Vec<Input> {
    assert_eq!(pair.as_rule(), Rule::InputStream);
    let mut inputs = Vec::new();
    let mut pairs = pair.into_inner();
    while let Some(pair) = pairs.next() {
        let start = pair.as_span().start();
        let name = parse_ident(spec, pair);

        let pair = pairs.next().expect("mismatch between grammar and AST");
        let end = pair.as_span().end();
        let ty = parse_type(spec, pair);
        inputs.push(Input {
            name,
            ty,
            span: Span { start, end },
        })
    }

    assert!(!inputs.is_empty());
    inputs
}

/**
 * Transforms a `Rule::OutputStrean` into `Output` AST node.
 * Panics if input is not `Rule::OutputStrean`.
 * The output rule consists of the following tokens:
 * - Rule::Ident
 * - Rule::Type
 * - Rule::Expr
 */
fn parse_output(spec: &mut LolaSpec, pair: Pair<Rule>) -> Output {
    assert_eq!(pair.as_rule(), Rule::OutputStream);
    let span = pair.as_span().into();
    let mut pairs = pair.into_inner();
    let name = parse_ident(
        spec,
        pairs.next().expect("mismatch between grammar and AST"),
    );
    let ty = parse_type(
        spec,
        pairs.next().expect("mismatch between grammar and AST"),
    );
    let pair = pairs.next().expect("mismatch between grammar and AST");
    let expr_span = pair.as_span();
    let expression = build_expression_ast(spec, pair.into_inner(), expr_span.into());
    Output {
        name,
        ty,
        expression,
        span,
    }
}

/**
 * Transforms a `Rule::Trigger` into `Trigger` AST node.
 * Panics if input is not `Rule::Trigger`.
 * The output rule consists of the following tokens:
 * - (Rule::Ident)?
 * - Rule::Expr
 * - (Rule::StringLiteral)?
 */
fn parse_trigger(spec: &mut LolaSpec, pair: Pair<Rule>) -> Trigger {
    assert_eq!(pair.as_rule(), Rule::Trigger);
    let span = pair.as_span().into();
    let mut pairs = pair.into_inner();

    let mut name = None;
    let mut message = None;

    let mut pair = pairs.next().expect("mistmatch between grammar and AST");
    // first token is either expression or identifier
    match pair.as_rule() {
        Rule::Ident => {
            name = Some(parse_ident(spec, pair));
            pair = pairs.next().expect("mistmatch between grammar and AST");
        }
        _ => (),
    }
    let expr_span = pair.as_span();
    let expression = build_expression_ast(spec, pair.into_inner(), expr_span.into());

    if let Some(pair) = pairs.next() {
        assert_eq!(pair.as_rule(), Rule::String);
        message = Some(spec.symbols.get_symbol_for(pair.as_str()));
    }

    Trigger {
        name,
        expression,
        message,
        span,
    }
}

/**
 * Transforms a `Rule::Ident` into `Ident` AST node.
 * Panics if input is not `Rule::Ident`.
 */
fn parse_ident(spec: &mut LolaSpec, pair: Pair<Rule>) -> Ident {
    assert_eq!(pair.as_rule(), Rule::Ident);
    let name = pair.as_str();
    let symbol = spec.symbols.get_symbol_for(name);
    Ident::new(symbol, pair.as_span().into())
}

/**
 * Transforms a `Rule::Type` into `Type` AST node.
 * Panics if input is not `Rule::Type`.
 */
fn parse_type(spec: &mut LolaSpec, pair: Pair<Rule>) -> Type {
    assert_eq!(pair.as_rule(), Rule::Type);
    let span = pair.as_span();
    let mut tuple = Vec::new();
    for pair in pair.into_inner() {
        match pair.as_rule() {
            Rule::Ident => {
                let ty = Type::new_simple(
                    spec.symbols.get_symbol_for(pair.as_str()),
                    pair.as_span().into(),
                );
                return ty;
            }
            Rule::Type => tuple.push(Box::new(parse_type(spec, pair))),
            _ => unreachable!(),
        }
    }
    Type::new_tuple(tuple, span.into())
}

/**
 * Transforms a `Rule::Literal` into `Literal` AST node.
 * Panics if input is not `Rule::Literal`.
 */
fn parse_literal(spec: &mut LolaSpec, pair: Pair<Rule>) -> Literal {
    assert_eq!(pair.as_rule(), Rule::Literal);
    let inner = pair
        .into_inner()
        .next()
        .expect("Rule::Literal has exactly one child");
    match inner.as_rule() {
        Rule::String => unimplemented!(),
        Rule::NumberLiteral => Literal::new_int(
            inner.as_str().parse::<i128>().unwrap(),
            inner.as_span().into(),
        ),
        Rule::TupleLiteral => unimplemented!(),
        Rule::True => Literal::new_bool(true, inner.as_span().into()),
        Rule::False => Literal::new_bool(false, inner.as_span().into()),
        _ => unreachable!(),
    }
}

/**
 * Builds the Expr AST.
 */
fn build_expression_ast(spec: &mut LolaSpec, pairs: Pairs<Rule>, span: Span) -> Expression {
    println!("{:#?}", pairs);
    PREC_CLIMBER.climb(
        pairs,
        |pair: Pair<Rule>| match pair.as_rule() {
            Rule::Literal => {
                let span = pair.as_span();
                Expression::new(ExpressionKind::Lit(parse_literal(spec, pair)), span.into())
            }
            Rule::Ident => {
                let span = pair.as_span();
                Expression::new(ExpressionKind::Ident(parse_ident(spec, pair)), span.into())
            }
            Rule::DefaultExpr => unimplemented!(),
            Rule::LookupExpr => unimplemented!(),
            Rule::FunctionExpr => unimplemented!(),
            Rule::UnaryExpr => unimplemented!(),
            Rule::TernaryExpr => unimplemented!(),
            Rule::Tuple => unimplemented!(),
            Rule::Expr => {
                let span = pair.as_span();
                build_expression_ast(spec, pair.into_inner(), span.into())
            }
            _ => unreachable!(),
        },
        |lhs: Expression, op: Pair<Rule>, rhs: Expression| {
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
                _ => unreachable!(),
            };
            Expression::new(
                ExpressionKind::Binary(op, Box::new(lhs), Box::new(rhs)),
                span,
            )
        },
    )
}

/// A symbol is a reference to an entry in SymbolTable
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct Symbol(u32);

impl Symbol {
    pub fn new(name: u32) -> Symbol {
        Symbol(name)
    }

    fn to_usize(&self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug)]
pub struct Ident {
    pub name: Symbol,
    pub span: Span,
}

impl Ident {
    fn new(name: Symbol, span: Span) -> Ident {
        Ident { name, span }
    }
}

/// A span marks a range in a file.
/// Start and end positions are *byte* offsets.
#[derive(Debug, Clone, Copy)]
pub struct Span {
    start: usize,
    end: usize,
}

impl<'a> From<pest::Span<'a>> for Span {
    fn from(span: pest::Span<'a>) -> Self {
        Span {
            start: span.start(),
            end: span.end(),
        }
    }
}

/// A SymbolTable is a bi-directional mapping between strings and symbols
#[derive(Debug)]
pub(crate) struct SymbolTable {
    names: HashMap<Box<str>, Symbol>,
    strings: Vec<Box<str>>,
}

impl SymbolTable {
    pub(crate) fn new() -> SymbolTable {
        SymbolTable {
            names: HashMap::new(),
            strings: Vec::new(),
        }
    }

    pub(crate) fn get_symbol_for(&mut self, string: &str) -> Symbol {
        // check if already presents
        if let Some(&name) = self.names.get(string) {
            return name;
        }

        // insert in symboltable
        let name = Symbol(self.strings.len() as u32);
        let copy = string.to_string().into_boxed_str();
        self.strings.push(copy.clone());
        self.names.insert(copy, name);

        name
    }

    pub(crate) fn get_string(&self, symbol: Symbol) -> &str {
        self.strings[symbol.to_usize()].as_ref()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn parse_simple() {
        let _ = LolaParser::parse(
            Rule::Spec,
            "input in: Int\noutput out: Int := in\ntrigger in != out",
        ).unwrap_or_else(|e| panic!("{}", e));
    }

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
        let formatted = format!("{:?}", ast);
        assert_eq!(formatted, "Constant { name: Ident { name: Symbol(0), span: Span { start: 9, end: 13 } }, ty: Type { kind: Simple(Symbol(1)), span: Span { start: 16, end: 19 } }, literal: Literal { kind: Int(5), span: Span { start: 23, end: 24 } }, span: Span { start: 0, end: 24 } }")
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
        println!("{:?}", inputs);
        assert_eq!(inputs.len(), 3);
    }

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
        let formatted = format!("{:?}", ast);
        assert_eq!(formatted, "Output { name: Ident { name: Symbol(0), span: Span { start: 7, end: 10 } }, ty: Type { kind: Simple(Symbol(1)), span: Span { start: 12, end: 15 } }, expression: Expression { kind: Binary(Add, Expression { kind: Ident(Ident { name: Symbol(2), span: Span { start: 19, end: 21 } }), span: Span { start: 19, end: 21 } }, Expression { kind: Lit(Literal { kind: Int(1), span: Span { start: 24, end: 25 } }), span: Span { start: 24, end: 25 } }), span: Span { start: 19, end: 25 } }, span: Span { start: 0, end: 25 } }")
    }

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
        let pair = LolaParser::parse(Rule::Trigger, "trigger in != out \"some message\"")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let ast = super::parse_trigger(&mut spec, pair);
        let formatted = format!("{:?}", ast);
        assert_eq!(formatted, "Trigger { name: None, expression: Expression { kind: Binary(Ne, Expression { kind: Ident(Ident { name: Symbol(0), span: Span { start: 8, end: 10 } }), span: Span { start: 8, end: 10 } }, Expression { kind: Ident(Ident { name: Symbol(1), span: Span { start: 14, end: 17 } }), span: Span { start: 14, end: 17 } }), span: Span { start: 8, end: 17 } }, message: Some(Symbol(2)), span: Span { start: 0, end: 32 } }")
    }

    #[test]
    fn parse_expression() {
        let expr = LolaParser::parse(Rule::Expr, "in + 1\n")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let span = expr.as_span();
        let ast = build_expression_ast(&mut spec, expr.into_inner(), span.into());
        let formatted = format!("{:?}", ast);
        assert_eq!(formatted, "Expression { kind: Binary(Add, Expression { kind: Ident(Ident { name: Symbol(0), span: Span { start: 0, end: 2 } }), span: Span { start: 0, end: 2 } }, Expression { kind: Lit(Literal { kind: Int(1), span: Span { start: 5, end: 6 } }), span: Span { start: 5, end: 6 } }), span: Span { start: 0, end: 6 } }")
    }

    #[test]
    fn parse_expression_precedence() {
        let expr = LolaParser::parse(Rule::Expr, "(a || b & c)")
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();
        let mut spec = LolaSpec::new();
        let span = expr.as_span();
        let ast = build_expression_ast(&mut spec, expr.into_inner(), span.into());
        let formatted = format!("{:?}", ast);
        assert_eq!(formatted, "Expression { kind: Binary(Or, Expression { kind: Ident(Ident { name: Symbol(0), span: Span { start: 1, end: 2 } }), span: Span { start: 1, end: 2 } }, Expression { kind: Binary(And, Expression { kind: Ident(Ident { name: Symbol(1), span: Span { start: 6, end: 7 } }), span: Span { start: 6, end: 7 } }, Expression { kind: Ident(Ident { name: Symbol(2), span: Span { start: 10, end: 11 } }), span: Span { start: 10, end: 11 } }), span: Span { start: 1, end: 11 } }), span: Span { start: 1, end: 11 } }")
    }

    #[test]
    fn symbol_table() {
        let mut symboltable = SymbolTable::new();
        let sym_a = symboltable.get_symbol_for("a");
        let sym_b = symboltable.get_symbol_for("b");
        assert_ne!(sym_a, sym_b);
        assert_eq!(sym_a, symboltable.get_symbol_for("a"));
        assert_eq!(symboltable.get_string(sym_a), "a");
        assert_eq!(symboltable.get_string(sym_b), "b");
    }
}
