use super::{Expression, ExpressionKind, LitKind, Offset, TimeUnit};
use crate::ast::Literal;
use lazy_static::lazy_static;
use num::rational::Rational64 as Rational;
use num::{traits::Inv, FromPrimitive, One, Signed, Zero};
use std::str::FromStr;
use uom::si::frequency::hertz;
use uom::si::rational64::Frequency as UOM_Frequency;
use uom::si::rational64::Time as UOM_Time;
use uom::si::time::second;

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

    pub(crate) fn parse_duration(&self) -> Option<UOM_Time> {
        let (val, unit) = match &self.kind {
            ExpressionKind::Lit(l) => match &l.kind {
                LitKind::Numeric(val, Some(unit)) => (parse_rational(val), unit),
                _ => return None,
            },
            _ => return None,
        };

        match unit.as_str() {
            "ns" | "μs" | "us" | "ms" | "s" | "min" | "h" | "d" | "w" | "a" => {
                use uom::si::time::*;
                let factor = match unit.as_str() {
                    "ns" => UOM_Time::new::<nanosecond>(Rational::one()),
                    "μs" | "us" => UOM_Time::new::<microsecond>(Rational::one()),
                    "ms" => UOM_Time::new::<millisecond>(Rational::one()),
                    "s" => UOM_Time::new::<second>(Rational::one()),
                    "min" => UOM_Time::new::<minute>(Rational::one()),
                    "h" => UOM_Time::new::<hour>(Rational::one()),
                    "d" => UOM_Time::new::<day>(Rational::one()),
                    "w" => UOM_Time::new::<day>(Rational::from_u64(7).unwrap()),
                    "a" => UOM_Time::new::<day>(Rational::from_u64(365).unwrap()),
                    _ => unreachable!(),
                };
                let duration = val * factor.get::<second>();
                Some(UOM_Time::new::<second>(duration))
            }
            _ => None,
        }
    }

    pub(crate) fn parse_frequency(&self) -> Option<UOM_Frequency> {
        let (val, unit) = match &self.kind {
            ExpressionKind::Lit(l) => match &l.kind {
                LitKind::Numeric(val, Some(unit)) => (parse_rational(val), unit),
                _ => return None,
            },
            _ => return None,
        };

        assert!(val.is_positive());

        match unit.as_str() {
            "μHz" | "uHz" | "mHz" | "Hz" | "kHz" | "MHz" | "GHz" => {
                use uom::si::frequency::*;
                let factor = match unit.as_str() {
                    "μHz" | "uHz" => UOM_Frequency::new::<microhertz>(Rational::one()),
                    "mHz" => UOM_Frequency::new::<millihertz>(Rational::one()),
                    "Hz" => UOM_Frequency::new::<hertz>(Rational::one()),
                    "kHz" => UOM_Frequency::new::<kilohertz>(Rational::one()),
                    "MHz" => UOM_Frequency::new::<megahertz>(Rational::one()),
                    "GHz" => UOM_Frequency::new::<gigahertz>(Rational::one()),
                    _ => unreachable!(),
                };
                let freq = val * factor.get::<hertz>();
                Some(UOM_Frequency::new::<hertz>(freq))
            }
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn parse_freqspec(&self) -> Option<UOM_Frequency> {
        if let Some(freq) = self.parse_frequency() {
            Some(freq)
        } else if let Some(period) = self.parse_duration() {
            let seconds = period.get::<second>();
            assert!(seconds.is_positive());
            Some(UOM_Frequency::new::<hertz>(seconds.inv()))
        } else {
            None
        }
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

impl Offset {
    pub(crate) fn to_uom_time(&self) -> Option<UOM_Time> {
        match self {
            Offset::Discrete(_) => None,
            Offset::RealTime(val, unit) => {
                let seconds = val * &unit.to_uom_time().get::<second>();
                Some(UOM_Time::new::<second>(seconds))
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

impl TimeUnit {
    fn to_uom_time(&self) -> UOM_Time {
        let f = match self {
            TimeUnit::Nanosecond => {
                Rational::new(RationalType::from_u64(1).unwrap(), RationalType::from_u64(10_u64.pow(9)).unwrap())
            }
            TimeUnit::Microsecond => {
                Rational::new(RationalType::from_u64(1).unwrap(), RationalType::from_u64(10_u64.pow(6)).unwrap())
            }
            TimeUnit::Millisecond => {
                Rational::new(RationalType::from_u64(1).unwrap(), RationalType::from_u64(10_u64.pow(3)).unwrap())
            }
            TimeUnit::Second => Rational::from_u64(1).unwrap(),
            TimeUnit::Minute => Rational::from_u64(60).unwrap(),
            TimeUnit::Hour => Rational::from_u64(60 * 60).unwrap(),
            TimeUnit::Day => Rational::from_u64(60 * 60 * 24).unwrap(),
            TimeUnit::Week => Rational::from_u64(60 * 60 * 24 * 7).unwrap(),
            TimeUnit::Year => Rational::from_u64(60 * 60 * 24 * 365).unwrap(),
        };
        UOM_Time::new::<second>(f)
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
    use num::ToPrimitive;
    use std::time::Duration;

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
        let freq = expr.parse_freqspec().unwrap();
        let period = UOM_Time::new::<second>(freq.get::<hertz>().inv());
        Duration::from_nanos(period.get::<uom::si::time::nanosecond>().to_integer().to_u64().unwrap())
    }

    #[test]
    fn test_time_spec_to_duration_conversion() {
        assert_eq!(time_spec_int("1", "s"), Duration::new(1, 0));
        assert_eq!(time_spec_int("2", "min"), Duration::new(2 * 60, 0));
        assert_eq!(time_spec_int("33", "h"), Duration::new(33 * 60 * 60, 0));
        assert_eq!(time_spec_int("12354", "ns"), Duration::from_nanos(12354));
        assert_eq!(time_spec_int("90351", "us"), Duration::from_nanos(90351 * 1_000));
        assert_eq!(time_spec_int("248", "ms"), Duration::from_nanos(248 * 1_000_000));
        assert_eq!(time_spec_int("29489232", "ms"), Duration::from_nanos(29489232 * 1_000_000));
    }

    #[test]
    fn test_frequency_to_duration_conversion() {
        assert_eq!(time_spec_int("1", "Hz"), Duration::new(1, 0));
        assert_eq!(time_spec_int("10", "Hz"), Duration::new(0, 100_000_000));
        assert_eq!(time_spec_int("400", "uHz"), Duration::new(2_500, 0));
        assert_eq!(time_spec_int("20", "mHz"), Duration::new(50, 0));
    }
}
