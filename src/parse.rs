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
}
