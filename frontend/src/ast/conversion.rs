use super::{Expression, ExpressionKind, LitKind, NodeId, Offset, TimeSpec, TimeUnit};
use crate::ast::Literal;
use num::{BigInt, BigRational, FromPrimitive, One, Signed, ToPrimitive};
use std::str::FromStr;
use std::time::Duration;

impl Expression {
    pub(crate) fn parse_offset(&self) -> Result<Offset, String> {
        if let Some(val) = self.parse_literal::<i16>() {
            Ok(Offset::Discrete(val))
        } else {
            // has to be a real-time expression
            let (val, unit) = match &self.kind {
                ExpressionKind::Lit(l) => match &l.kind {
                    LitKind::Numeric(val, Some(unit)) => (val, unit),
                    _ => return Err(format!("expected numeric value with unit, found `{}`", l)),
                },
                _ => return Err(format!("expected numeric value with unit, found `{}`", self)),
            };
            Ok(Offset::RealTime(parse_rational(val), TimeUnit::from_str(unit)?))
        }
    }

    pub(crate) fn parse_timespec(&self) -> Option<TimeSpec> {
        let (val, unit) = match &self.kind {
            ExpressionKind::Lit(l) => match &l.kind {
                LitKind::Numeric(val, Some(unit)) => (val, unit),
                _ => return None,
            },
            _ => return None,
        };

        let (factor, invert): (BigRational, bool) = match unit.as_str() {
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

        let mut period: BigRational = parse_rational(val);
        if invert {
            period = BigRational::one() / period;
        }
        period *= factor;
        let (rounded_period, signum) = if period.is_negative() {
            let rounded_period: Duration =
                Duration::from_nanos((-period.clone()).to_integer().to_u64().expect("Period [ns] too large for u64!"));
            (
                rounded_period,
                match rounded_period.as_nanos() {
                    0 => 0,
                    _ => -1,
                },
            )
        } else {
            let rounded_period: Duration =
                Duration::from_nanos(period.to_integer().to_u64().expect("Period [ns] too large for u64!"));
            (
                rounded_period,
                match rounded_period.as_nanos() {
                    0 => 0,
                    _ => 1,
                },
            )
        };
        Some(TimeSpec { period: rounded_period, exact_period: period, signum, id: NodeId::DUMMY, span: self.span })
    }
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

impl Expression {
    /// Attempts to extract the numeric, constant, unit-less value out of an `Expression::Lit`.
    pub(crate) fn parse_literal<T>(&self) -> Option<T>
    where
        T: FromStr,
    {
        match &self.kind {
            ExpressionKind::Lit(l) => l.parse_numeric(),
            _ => None,
        }
    }
}

impl Literal {
    pub(crate) fn parse_numeric<T>(&self) -> Option<T>
    where
        T: FromStr,
    {
        match &self.kind {
            LitKind::Numeric(val, unit) => {
                if unit.is_some() {
                    return None;
                }
                val.parse::<T>().ok()
            }
            _ => None,
        }
    }
}

impl Expression {
    /// Attempts to extract the numeric, constant, unit-less value out of an `Expression::Lit`.
    pub(crate) fn to_uom_string(&self) -> Option<String> {
        match &self.kind {
            ExpressionKind::Lit(l) => match &l.kind {
                LitKind::Numeric(val, Some(unit)) => {
                    let parsed = parse_rational(val);
                    Some(format!("{} {}", parsed, unit))
                }
                _ => None,
            },
            _ => None,
        }
    }
}

impl Offset {
    pub(crate) fn to_uom_time(&self) -> Option<uom::si::bigrational::Time> {
        match self {
            Offset::Discrete(_) => None,
            Offset::RealTime(val, unit) => {
                let coefficient = match unit {
                    TimeUnit::Nanosecond => BigRational::new(BigInt::from(1), BigInt::from(10_u64.pow(9))),
                    TimeUnit::Microsecond => BigRational::new(BigInt::from(1), BigInt::from(10_u64.pow(6))),
                    TimeUnit::Millisecond => BigRational::new(BigInt::from(1), BigInt::from(10_u64.pow(3))),
                    TimeUnit::Second => BigRational::from_u64(1_u64).unwrap(),
                    TimeUnit::Minute => BigRational::from_u64(60).unwrap(),
                    TimeUnit::Hour => BigRational::from_u64(60 * 60).unwrap(),
                    TimeUnit::Day => BigRational::from_u64(60 * 60 * 24).unwrap(),
                    TimeUnit::Week => BigRational::from_u64(60 * 60 * 24 * 7).unwrap(),
                    TimeUnit::Year => BigRational::from_u64(60 * 60 * 24 * 365).unwrap(),
                };
                let time = val * coefficient;
                Some(uom::si::bigrational::Time::new::<uom::si::time::second>(time))
            }
        }
    }
}

impl FromStr for TimeUnit {
    type Err = String;
    fn from_str(unit: &str) -> Result<Self, Self::Err> {
        match unit {
            "ns" => Ok(TimeUnit::Nanosecond),
            "μs" | "us" => Ok(TimeUnit::Microsecond),
            "ms" => Ok(TimeUnit::Millisecond),
            "s" => Ok(TimeUnit::Second),
            "min" => Ok(TimeUnit::Minute),
            "h" => Ok(TimeUnit::Hour),
            "d" => Ok(TimeUnit::Day),
            "w" => Ok(TimeUnit::Week),
            "a" => Ok(TimeUnit::Year),
            _ => Err(format!("unknown time unit `{}`", unit)),
        }
    }
}

impl Expression {
    /// Tries to resolve a tuple index access
    pub(crate) fn get_expr_from_tuple(&self, idx: usize) -> Option<&Expression> {
        use ExpressionKind::*;
        match &self.kind {
            Tuple(entries) => Some(entries[idx].as_ref()),
            Ident(_) => None,
            _ => unimplemented!(),
        }
    }

    /// A recursive iterator over an `Expression` tree
    /// Inspired by https://amos.me/blog/2019/recursive-iterators-rust/
    pub(crate) fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = &Expression> + 'a> {
        use ExpressionKind::*;
        match &self.kind {
            Lit(_) | Ident(_) | MissingExpression => Box::new(std::iter::once(self)),
            Unary(_, inner)
            | Field(inner, _)
            | StreamAccess(inner, _)
            | Offset(inner, _)
            | ParenthesizedExpression(_, inner, _) => Box::new(std::iter::once(self).chain(inner.iter())),
            Binary(_, left, right)
            | Default(left, right)
            | SlidingWindowAggregation { expr: left, duration: right, .. } => {
                Box::new(std::iter::once(self).chain(left.iter()).chain(right.iter()))
            }
            Ite(cond, normal, alternative) => {
                Box::new(std::iter::once(self).chain(cond.iter()).chain(normal.iter()).chain(alternative.iter()))
            }
            Tuple(entries) | Function(_, _, entries) => {
                Box::new(std::iter::once(self).chain(entries.iter().map(|entry| entry.iter()).flatten()))
            }
            Method(base, _, _, arguments) => Box::new(
                std::iter::once(self).chain(base.iter()).chain(arguments.iter().map(|entry| entry.iter()).flatten()),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Literal, Span};

    fn time_spec_int(val: &str, unit: &str) -> Duration {
        let expr = Expression::new(
            ExpressionKind::Lit(Literal::new_numeric(val, Some(unit.to_string()), Span::unknown())),
            Span::unknown(),
        );
        let time_spec = expr.parse_timespec().unwrap();
        time_spec.period
    }

    #[test]
    fn test_time_spec_to_duration_conversion() {
        assert_eq!(time_spec_int("1", "s"), Duration::new(1, 0));
        assert_eq!(time_spec_int("2", "min"), Duration::new(2 * 60, 0));
        assert_eq!(time_spec_int("33", "h"), Duration::new(33 * 60 * 60, 0));
        assert_eq!(time_spec_int("12354", "ns"), Duration::new(0, 12354));
        assert_eq!(time_spec_int("90351", "us"), Duration::new(0, 90351 * 1_000));
        assert_eq!(time_spec_int("248", "ms"), Duration::new(0, 248 * 1_000_000));
        assert_eq!(time_spec_int("29489232", "ms"), Duration::new(29_489, 232 * 1_000_000));
    }

    #[test]
    fn test_frequency_to_duration_conversion() {
        assert_eq!(time_spec_int("1", "Hz"), Duration::new(1, 0));
        assert_eq!(time_spec_int("10", "Hz"), Duration::new(0, 100_000_000));
        assert_eq!(time_spec_int("400", "uHz"), Duration::new(2_500, 0));
        assert_eq!(time_spec_int("20", "mHz"), Duration::new(50, 0));
    }
}
