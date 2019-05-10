//! This module contains the basic definition of types
//!
//! It is inspired by <https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/index.html>

pub(crate) mod check;
pub(crate) mod unifier;

use lazy_static::lazy_static;
use num::{BigRational, Integer, Zero};
use unifier::{StreamVar, ValueVar};
use uom::si::bigrational::Frequency;
use uom::si::frequency::hertz;

/// The type of an expression consists of both, a value type (`Bool`, `String`, etc.) and
/// a stream type (periodic or event-based).
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub struct Ty {
    value: ValueTy,
    stream: StreamTy,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub enum StreamTy {
    /// An event stream with the given dependencies
    Event(Activation),
    // A real-time stream with given frequency
    RealTime(Freq),
    /// The type of the stream should be inferred as the conjunction of the given `StreamTy`
    Infer(Vec<StreamVar>),
}

/// The `value` type, storing information about the stored values (`Bool`, `UInt8`, etc.)
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub enum ValueTy {
    Bool,
    Int(IntTy),
    UInt(UIntTy),
    Float(FloatTy),
    // an abstract data type, e.g., structs, enums, etc.
    //Adt(AdtDef),
    String,
    Tuple(Vec<ValueTy>),
    /// an optional value type, e.g., resulting from accessing a stream with offset -1
    Option(Box<ValueTy>),
    /// Used during type inference
    Infer(ValueVar),
    Constr(TypeConstraint),
    /// A reference to a generic parameter in a function declaration, e.g. `T` in `a<T>(x:T) -> T`
    Param(u8, String),
    Error,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
}
use self::IntTy::*;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum UIntTy {
    U8,
    U16,
    U32,
    U64,
}
use self::UIntTy::*;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum FloatTy {
    F32,
    F64,
}
use self::FloatTy::*;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub struct Freq {
    pub(crate) freq: Frequency,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum Activation {
    Conjunction(Vec<Activation>),
    Disjunction(Vec<Activation>),
    Stream(StreamVar),
}

impl std::fmt::Display for Freq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.freq.clone().into_format_args(uom::si::frequency::hertz, uom::fmt::DisplayStyle::Abbreviation)
        )
    }
}

impl StreamTy {
    pub(crate) fn new_event(activation: Activation) -> StreamTy {
        StreamTy::Event(activation)
    }

    pub(crate) fn new_periodic(freq: Freq) -> StreamTy {
        StreamTy::RealTime(freq)
    }

    pub(crate) fn new_inferred() -> StreamTy {
        StreamTy::Infer(Vec::new())
    }
}

impl Freq {
    pub(crate) fn new(freq: Frequency) -> Self {
        Freq { freq }
    }

    pub(crate) fn is_multiple_of(&self, other: &Freq) -> bool {
        if self.freq.get::<hertz>() < other.freq.get::<hertz>() {
            return false;
        }
        (self.freq.get::<hertz>() % other.freq.get::<hertz>()).is_zero()
    }

    pub(crate) fn conjunction(&self, other: &Freq) -> Freq {
        let numer_left = self.freq.get::<hertz>().numer().clone();
        let numer_right = other.freq.get::<hertz>().numer().clone();
        let denom_left = self.freq.get::<hertz>().denom().clone();
        let denom_right = other.freq.get::<hertz>().denom().clone();
        // lcm(self, other) = lcm(numer_left, numer_right) / gcd(denom_left, denom_right)
        // only works if rational numbers are reduced, which ist the default for `BigRational`
        Freq {
            freq: Frequency::new::<hertz>(BigRational::new(numer_left.lcm(&numer_right), denom_left.gcd(&denom_right))),
        }
    }
}

lazy_static! {
    static ref PRIMITIVE_TYPES: Vec<(&'static str, &'static ValueTy)> = vec![
        ("Bool", &ValueTy::Bool),
        ("Int8", &ValueTy::Int(I8)),
        ("Int16", &ValueTy::Int(I16)),
        ("Int32", &ValueTy::Int(I32)),
        ("Int64", &ValueTy::Int(I64)),
        ("UInt8", &ValueTy::UInt(U8)),
        ("UInt16", &ValueTy::UInt(U16)),
        ("UInt32", &ValueTy::UInt(U32)),
        ("UInt64", &ValueTy::UInt(U64)),
        ("Float32", &ValueTy::Float(F32)),
        ("Float64", &ValueTy::Float(F64)),
        ("String", &ValueTy::String),
    ];
}

impl ValueTy {
    pub(crate) fn primitive_types() -> std::slice::Iter<'static, (&'static str, &'static ValueTy)> {
        PRIMITIVE_TYPES.iter()
    }

    pub(crate) fn satisfies(&self, constraint: &TypeConstraint) -> bool {
        use self::TypeConstraint::*;
        use self::ValueTy::*;
        match constraint {
            Unconstrained => true,
            Comparable | Equatable => self.is_primitive(),
            Numeric => self.satisfies(&Integer) || self.satisfies(&FloatingPoint),
            FloatingPoint => match self {
                Float(_) => true,
                _ => false,
            },
            Integer => self.satisfies(&SignedInteger) || self.satisfies(&UnsignedInteger),
            SignedInteger => match self {
                Int(_) => true,
                _ => false,
            },
            UnsignedInteger => match self {
                UInt(_) => true,
                _ => false,
            },
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_error(&self) -> bool {
        use self::ValueTy::*;
        match self {
            Error => true,
            Tuple(args) => args.iter().any(|el| el.is_error()),
            Option(ty) => ty.is_error(),
            _ => false,
        }
    }

    pub(crate) fn is_primitive(&self) -> bool {
        use self::ValueTy::*;
        match self {
            Bool | Int(_) | UInt(_) | Float(_) | String => true,
            _ => false,
        }
    }

    /// Replaces parameters by the given list
    pub(crate) fn replace_params(&self, infer_vars: &[ValueVar]) -> ValueTy {
        match self {
            &ValueTy::Param(id, _) => ValueTy::Infer(infer_vars[id as usize]),
            ValueTy::Option(t) => ValueTy::Option(t.replace_params(infer_vars).into()),
            ValueTy::Infer(_) | ValueTy::Constr(_) => self.clone(),
            _ if self.is_primitive() => self.clone(),
            _ => unreachable!("replace_param for {}", self),
        }
    }

    /// Replaces constraints by default values
    pub(crate) fn replace_constr(&self) -> ValueTy {
        match &self {
            ValueTy::Tuple(t) => ValueTy::Tuple(t.iter().map(|el| el.replace_constr()).collect()),
            ValueTy::Option(ty) => ValueTy::Option(ty.replace_constr().into()),
            ValueTy::Constr(c) => match c.has_default() {
                Some(d) => d,
                None => ValueTy::Error,
            },
            ValueTy::Param(_, _) => self.clone(),
            _ if self.is_primitive() => self.clone(),
            _ => unreachable!("cannot replace_constr for {}", self),
        }
    }
}

impl std::fmt::Display for ValueTy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ValueTy::Bool => write!(f, "Bool"),
            ValueTy::Int(I8) => write!(f, "Int8"),
            ValueTy::Int(I16) => write!(f, "Int16"),
            ValueTy::Int(I32) => write!(f, "Int32"),
            ValueTy::Int(I64) => write!(f, "Int64"),
            ValueTy::UInt(U8) => write!(f, "UInt8"),
            ValueTy::UInt(U16) => write!(f, "UInt16"),
            ValueTy::UInt(U32) => write!(f, "UInt32"),
            ValueTy::UInt(U64) => write!(f, "UInt64"),
            ValueTy::Float(F32) => write!(f, "Float32"),
            ValueTy::Float(F64) => write!(f, "Float64"),
            ValueTy::String => write!(f, "String"),
            ValueTy::Option(ty) => write!(f, "Option<{}>", ty),
            ValueTy::Tuple(inner) => {
                let joined: Vec<String> = inner.iter().map(|e| format!("{}", e)).collect();
                write!(f, "({})", joined.join(", "))
            }
            ValueTy::Infer(id) => write!(f, "?{}", id),
            ValueTy::Constr(constr) => write!(f, "{{{}}}", constr),
            ValueTy::Param(_, name) => write!(f, "{}", name),
            ValueTy::Error => write!(f, "Error"),
        }
    }
}

impl std::fmt::Display for StreamTy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            StreamTy::Event(activation) => write!(f, "EventStream({})", activation),
            StreamTy::RealTime(freq) => write!(f, "PeriodicStream({})", freq),
            StreamTy::Infer(vars) => write!(f, "InferedStream({:?})", vars),
        }
    }
}

impl std::fmt::Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use crate::ast::print::write_delim_list;
        match self {
            Activation::Conjunction(con) => {
                if con.is_empty() {
                    write!(f, "true")
                } else {
                    write_delim_list(f, &con, "(", ")", " & ")
                }
            }
            Activation::Disjunction(dis) => write_delim_list(f, &dis, "(", ")", " | "),
            Activation::Stream(v) => write!(f, "{}", v),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub enum TypeConstraint {
    SignedInteger,
    UnsignedInteger,
    FloatingPoint,
    /// signed + unsigned integer
    Integer,
    /// integer + floating point
    Numeric,
    /// Types that can be comparaed, i.e., implement `==`
    Equatable,
    /// Types that can be ordered, i.e., implement `<`, `>`,
    Comparable,
    Unconstrained,
}

impl TypeConstraint {
    pub(crate) fn has_default(&self) -> Option<ValueTy> {
        use self::TypeConstraint::*;
        match self {
            Integer | SignedInteger | Numeric => Some(ValueTy::Int(I32)),
            UnsignedInteger => Some(ValueTy::UInt(U32)),
            FloatingPoint => Some(ValueTy::Float(F32)),
            _ => None,
        }
    }

    pub(crate) fn conjunction<'a>(&'a self, other: &'a TypeConstraint) -> Option<&'a TypeConstraint> {
        use self::TypeConstraint::*;
        if self > other {
            return other.conjunction(self);
        }
        if self == other {
            return Some(self);
        }
        assert!(self < other);
        match other {
            Unconstrained | Comparable | Equatable | Numeric => Some(self),
            Integer => match self {
                FloatingPoint => None,
                _ => Some(self),
            },
            FloatingPoint | SignedInteger | UnsignedInteger => None,
        }
    }
}

impl std::fmt::Display for TypeConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::TypeConstraint::*;
        match self {
            SignedInteger => write!(f, "signed integer"),
            UnsignedInteger => write!(f, "unsigned integer"),
            Integer => write!(f, "integer"),
            FloatingPoint => write!(f, "floating point"),
            Numeric => write!(f, "numeric type"),
            Equatable => write!(f, "equatable type"),
            Comparable => write!(f, "comparable type"),
            Unconstrained => write!(f, "unconstrained type"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::traits::cast::FromPrimitive;

    #[test]
    fn test_freq_conjunction() {
        let a = Freq::new(Frequency::new::<hertz>(BigRational::from_i64(2).unwrap()));
        let b = Freq::new(Frequency::new::<hertz>(BigRational::from_i64(3).unwrap()));
        let c = Freq::new(Frequency::new::<hertz>(BigRational::from_i64(6).unwrap()));
        assert_eq!(a.conjunction(&b), c)
    }
}
