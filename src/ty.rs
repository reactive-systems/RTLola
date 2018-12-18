//! This module contains the basic definition of types
//!
//! It is inspired by https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/index.html

use crate::analysis::typing::ValueVar;
use std::time::Duration;

/// The `stream` type, storing information about timing of a stream (event-based, real-time).
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub struct StreamTy {
    pub parameters: Vec<ValueTy>,
    pub timing: TimingInfo,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub enum TimingInfo {
    /// An event stream with the given dependencies
    Event,
    // A real-time stream with given frequency
    RealTime(Freq),
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
    repr: String,
    d: Duration,
}

impl std::fmt::Display for Freq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.repr)
    }
}

impl StreamTy {
    pub(crate) fn new(timing: TimingInfo) -> StreamTy {
        StreamTy {
            parameters: vec![],
            timing,
        }
    }

    pub(crate) fn new_parametric(parameters: Vec<ValueTy>, timing: TimingInfo) -> StreamTy {
        StreamTy { parameters, timing }
    }

    pub(crate) fn is_parametric(&self) -> bool {
        !self.parameters.is_empty()
    }
}

impl Freq {
    pub(crate) fn new(repr: &str, d: Duration) -> Self {
        Freq {
            repr: repr.to_string(),
            d,
        }
    }

    const NANOS_PER_SEC: u32 = 1_000_000_000;

    pub(crate) fn is_multiple_of(&self, other: &Freq) -> bool {
        if self.d < other.d {
            return false;
        }
        // TODO: replace by self.as_nanos() when stabilized
        let left_nanos =
            self.d.as_secs() as u128 * Freq::NANOS_PER_SEC as u128 + self.d.subsec_nanos() as u128;
        let right_nanos = other.d.as_secs() as u128 * Freq::NANOS_PER_SEC as u128
            + other.d.subsec_nanos() as u128;
        assert!(left_nanos >= right_nanos);
        left_nanos % right_nanos == 0
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
            Comparable => self.is_primitive(),
            Equatable => self.is_primitive(),
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
            ValueTy::Infer(_) => self.clone(),
            ValueTy::Constr(_) => self.clone(),
            _ if self.is_primitive() => self.clone(),
            _ => unreachable!("replace_param for {}", self),
        }
    }

    /// Replaces constraints by default values
    pub(crate) fn replace_constr(&self) -> ValueTy {
        match &self {
            ValueTy::Tuple(t) => {
                ValueTy::Tuple(t.into_iter().map(|el| el.replace_constr()).collect())
            }
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
        if self.is_parametric() {
            let joined: Vec<String> = self.parameters.iter().map(|e| format!("{}", e)).collect();
            match &self.timing {
                TimingInfo::Event => write!(f, "Parametric<({}), EventStream>", joined.join(", ")),
                TimingInfo::RealTime(freq) => write!(
                    f,
                    "Parametric<({}), RealTimeStream<{}>>",
                    joined.join(", "),
                    freq
                ),
            }
        } else {
            match &self.timing {
                TimingInfo::Event => write!(f, "EventStream"),
                TimingInfo::RealTime(freq) => write!(f, "RealTimeStream<{}>", freq),
            }
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

    pub(crate) fn conjunction<'a>(
        &'a self,
        other: &'a TypeConstraint,
    ) -> Option<&'a TypeConstraint> {
        use self::TypeConstraint::*;
        if self > other {
            return other.conjunction(self);
        }
        if self == other {
            return Some(self);
        }
        assert!(self < other);
        match other {
            Unconstrained => Some(self),
            Comparable => Some(self),
            Equatable => Some(self),
            Numeric => Some(self),
            Integer => match self {
                FloatingPoint => None,
                _ => Some(self),
            },
            FloatingPoint => None,
            SignedInteger => None,
            UnsignedInteger => None,
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
