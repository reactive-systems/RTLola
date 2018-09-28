//! This module contains the parser for the Lola Language.

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

/// A symbol is a reference to an entry in SymbolTable
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub(crate) struct Symbol(u32);

impl Symbol {
    pub fn new(name: u32) -> Symbol {
        Symbol(name)
    }

    fn to_usize(&self) -> usize {
        self.0 as usize
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
    fn test_parse_simple() {
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
                    NumberLiteral(23, 24, []),
                ]),
            ]
        };
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
                        NumberLiteral(24, 25, []),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn test_symbol_table() {
        let mut symboltable = SymbolTable::new();
        let sym_a = symboltable.get_symbol_for("a");
        let sym_b = symboltable.get_symbol_for("b");
        assert_ne!(sym_a, sym_b);
        assert_eq!(sym_a, symboltable.get_symbol_for("a"));
        assert_eq!(symboltable.get_string(sym_a), "a");
        assert_eq!(symboltable.get_string(sym_b), "b");
    }
}
