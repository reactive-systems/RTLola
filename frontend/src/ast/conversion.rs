use super::{Expression, ExpressionKind, LitKind, NodeId, Offset, TimeSpec, TimeUnit};
use crate::ast::Literal;
use lazy_static::lazy_static;
use num::rational::Rational64 as Rational;
use num::{FromPrimitive, One, Signed, ToPrimitive, Zero};
use std::str::FromStr;
use std::time::Duration;
use uom::si::rational64::Time as UOM_Time;

type RationalType = i64;

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

        let (factor, invert): (Rational, bool) = match unit.as_str() {
            "ns" => (Rational::from_u64(1_u64).unwrap(), false),
            "μs" | "us" => (Rational::from_u64(10_u64.pow(3)).unwrap(), false),
            "ms" => (Rational::from_u64(10_u64.pow(6)).unwrap(), false),
            "s" => (Rational::from_u64(10_u64.pow(9)).unwrap(), false),
            "min" => (Rational::from_u64(10_u64.pow(9) * 60).unwrap(), false),
            "h" => (Rational::from_u64(10_u64.pow(9) * 60 * 60).unwrap(), false),
            "d" => (Rational::from_u64(10_u64.pow(9) * 60 * 60 * 24).unwrap(), false),
            "w" => (Rational::from_u64(10_u64.pow(9) * 60 * 60 * 24 * 7).unwrap(), false),
            "a" => (Rational::from_u64(10_u64.pow(9) * 60 * 60 * 24 * 365).unwrap(), false),
            "μHz" | "uHz" => (Rational::from_u64(10_u64.pow(15)).unwrap(), true),
            "mHz" => (Rational::from_u64(10_u64.pow(12)).unwrap(), true),
            "Hz" => (Rational::from_u64(10_u64.pow(9)).unwrap(), true),
            "kHz" => (Rational::from_u64(10_u64.pow(6)).unwrap(), true),
            "MHz" => (Rational::from_u64(10_u64.pow(3)).unwrap(), true),
            "GHz" => (Rational::from_u64(1).unwrap(), true),
            _ => unreachable!(),
        };

        let mut period: Rational = parse_rational(val);
        if invert {
            period = Rational::one() / period;
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

lazy_static! {
    static ref DIGITS: Vec<RationalType> = (0..=10).map(|d| RationalType::from_u8(d).unwrap()).collect();
}

fn parse_rational(repr: &str) -> Rational {
    // precondition: repr is a valid floating point literal
    assert!(repr.parse::<f64>().is_ok());

    let mut numer = RationalType::zero();
    let mut denom = RationalType::one();

    let mut negated = false;
    let mut decimal_places = None;
    let mut exponent = None;

    let mut chars = repr.chars();
    while let Some(c) = chars.next() {
        match c {
            '+' => {}
            '-' => {
                if let Some((_, ref mut negated)) = exponent {
                    *negated = true;
                } else {
                    negated = true;
                }
            }
            'e' => {
                // switch to parsing exponent
                exponent = Some((0, false));
            }
            '.' => {
                // start counting decimal places
                decimal_places = Some(0_u32);
            }
            _ if c.is_digit(10) => {
                let d = c.to_digit(10).unwrap();
                if let Some((ref mut exp, _)) = exponent {
                    *exp *= 10;
                    *exp += d;
                } else {
                    numer *= &DIGITS[10];
                    numer += &DIGITS[d as usize];
                    if let Some(ref mut n) = decimal_places {
                        *n += 1
                    }
                }
            }
            _ => unreachable!("incorrectly formated float"),
        }
    }

    if negated {
        numer = -numer;
    }

    if let Some(n) = decimal_places {
        denom *= DIGITS[10].pow(n);
    }

    if let Some((exp, negated)) = exponent {
        let e_factor = DIGITS[10].pow(exp);
        if negated {
            denom *= e_factor;
        } else {
            numer *= e_factor;
        }
    }

    Rational::new(numer, denom)
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
    pub(crate) fn to_uom_time(&self) -> Option<UOM_Time> {
        match self {
            Offset::Discrete(_) => None,
            Offset::RealTime(val, unit) => {
                let coefficient = match unit {
                    TimeUnit::Nanosecond => Rational::new(
                        RationalType::from_u64(1).unwrap(),
                        RationalType::from_u64(10_u64.pow(9)).unwrap(),
                    ),
                    TimeUnit::Microsecond => Rational::new(
                        RationalType::from_u64(1).unwrap(),
                        RationalType::from_u64(10_u64.pow(6)).unwrap(),
                    ),
                    TimeUnit::Millisecond => Rational::new(
                        RationalType::from_u64(1).unwrap(),
                        RationalType::from_u64(10_u64.pow(3)).unwrap(),
                    ),
                    TimeUnit::Second => Rational::from_u64(1).unwrap(),
                    TimeUnit::Minute => Rational::from_u64(60).unwrap(),
                    TimeUnit::Hour => Rational::from_u64(60 * 60).unwrap(),
                    TimeUnit::Day => Rational::from_u64(60 * 60 * 24).unwrap(),
                    TimeUnit::Week => Rational::from_u64(60 * 60 * 24 * 7).unwrap(),
                    TimeUnit::Year => Rational::from_u64(60 * 60 * 24 * 365).unwrap(),
                };
                let time = val * coefficient;
                Some(UOM_Time::new::<uom::si::time::second>(time))
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

    #[test]
    fn test_parse_rational() {
        macro_rules! check_on {
            ($f:expr) => {
                let f_string = format!("{}", $f);
                let f = f_string.parse::<f64>().unwrap();
                assert_eq!(parse_rational(f_string.as_str()), Rational::from_f64(f).unwrap());
            };
        };
        check_on!(0);
        check_on!(42);
        check_on!(-1);
        check_on!(0.1);
        check_on!(42.12);
        check_on!(-1.123);
        check_on!(0.1e-0);
        check_on!(42.12e+1);
        check_on!(-1.123e-2);
    }

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
