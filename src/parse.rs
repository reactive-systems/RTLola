//! This module contains the parser for the Lola Language.

use ast::*;
use pest;
use pest::iterators::{Pair, Pairs};
use pest::prec_climber::{Assoc, Operator, PrecClimber};
use pest::Parser;
use std::collections::HashMap;

#[derive(Parser)]
#[grammar = "lola.pest"]
struct LolaParser;

lazy_static! {
    // precedence taken from C/C++: https://en.wikipedia.org/wiki/Operators_in_C_and_C++
    // Precedence climber can be used to build the AST, see https://pest-parser.github.io/book/ for more details
    static ref PREC_CLIMBER: PrecClimber<Rule> = {
        use self::Assoc::*;
        use self::Rule::*;

        PrecClimber::new(vec![
            Operator::new(Or, Left),
            Operator::new(And, Left),
            Operator::new(Add, Left) | Operator::new(Subtract, Left),
            Operator::new(Multiply, Left) | Operator::new(Divide, Left) | Operator::new(Mod, Left),
            Operator::new(Power, Right),
        ])
    };
}

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
            _ => unimplemented!(),
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
    let ident = parse_ident(spec, pairs.next().expect("constants have exactly 3 pairs"));
    let ty = parse_type(spec, pairs.next().expect("constants have exactly 3 pairs"));
    let lit = parse_literal(spec, pairs.next().expect("constants have exactly 3 pairs"));
    Constant {
        name: ident,
        ty,
        literal: lit,
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
            _ => unreachable!(),
        },
        |lhs: Expression, op: Pair<Rule>, rhs: Expression| match op.as_rule() {
            Rule::Add => Expression::new(
                ExpressionKind::Binary(BinOp::Add, Box::new(lhs), Box::new(rhs)),
                span,
            ),
            _ => unreachable!(),
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
        let pairs = LolaParser::parse(Rule::Spec, "input in: Int\noutput out: Int := in\n")
            .unwrap_or_else(|e| panic!("{}", e));
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
