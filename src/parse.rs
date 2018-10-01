use pest::prec_climber::{Assoc, Operator, PrecClimber};
use pest::Parser;

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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_parse_simple() {
        let pairs = LolaParser::parse(Rule::Spec, "input Int32 in\noutput Int32 out := in\n")
            .unwrap_or_else(|e| panic!("{}", e));
    }

    #[test]
    fn parse_constant() {
        parses_to! {
            parser: LolaParser,
            input:  "constant Int five := 5",
            rule:   Rule::ConstantStream,
            tokens: [
                ConstantStream(0, 22, [
                    Type(9, 12, [
                        Ident(9, 12, []),
                    ]),
                    Ident(13, 17, []),
                    NumberLiteral(21, 22, []),
                ]),
            ]
        };
    }

    #[test]
    fn parse_input() {
        parses_to! {
            parser: LolaParser,
            input:  "input Int in",
            rule:   Rule::InputStream,
            tokens: [
                InputStream(0, 12, [
                    Type(6, 9, [
                        Ident(6, 9, []),
                    ]),
                    Ident(10, 12, [])
                ]),
            ]
        };
    }

    #[test]
    fn parse_output() {
        parses_to! {
            parser: LolaParser,
            input:  "output Int out := in + 1",
            rule:   Rule::OutputStream,
            tokens: [
                OutputStream(0, 24, [
                    Type(7, 10, [
                        Ident(7, 10, []),
                    ]),
                    Ident(11, 14, []),
                    Expr(18, 24, [
                        Ident(18, 20, []),
                        Add(21, 22, []),
                        NumberLiteral(23, 24, []),
                    ]),
                ]),
            ]
        };
    }
}
