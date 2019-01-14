//! This module contains the basic definition of types
//!
//! It is inspired by https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/index.html

use crate::analysis::typing::InferVar;
use ast_node::NodeId;
use std::time::Duration;

/// Representation of types
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub enum Ty {
    Bool,
    Int(IntTy),
    UInt(UIntTy),
    Float(FloatTy),
    // an abstract data type, e.g., structs, enums, etc.
    //Adt(AdtDef),
    String,
    Tuple(Vec<Ty>),
    /// An event stream of the given type
    EventStream(Box<Ty>, Vec<NodeId>),
    /// A parameterized stream
    Parameterized(Box<Ty>, Vec<Ty>),
    TimedStream(Box<Ty>, Freq),
    /// Representation of a sliding window
    Window(Box<Ty>, Duration),
    /// an optional value type, e.g., accessing a stream with offset -1
    Option(Box<Ty>),
    /// Used during type inference
    Infer(InferVar),
    Constr(TypeConstraint),
    /// A reference to a generic parameter in a function declaration, e.g. `T` in `a<T>(x:T) -> T`
    Param(u8, String),
    Error,
}

/// Representation of types, consisting of two orthogonal information.
/// * The `stream` type, storing information about timing of a stream (event-based, real-time).
/// * The `value` type, storing information about the stored values (`Bool`, `UInt8`, etc.)
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
struct NewTy {
    stream: StreamTy,
    value: ValueTy,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
struct StreamTy {
    parameters: Vec<ValueTy>,
    timing: TimingInfo,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
enum TimingInfo {
    /// An event stream with the given dependencies
    Event(Vec<NodeId>),
    // A real-time stream with given frequency
    RealTime(Freq),
    /// Used during type inference
    Infer(InferVar),
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
enum ValueTy {
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
    Infer(InferVar),
    Constr(TypeConstraint),
    /// A reference to a generic parameter in a function declaration, e.g. `T` in `a<T>(x:T) -> T`
    Param(u8, String),
    Error,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
}
use self::IntTy::*;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum UIntTy {
    U8,
    U16,
    U32,
    U64,
}
use self::UIntTy::*;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
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
    static ref PRIMITIVE_TYPES: Vec<(&'static str, &'static Ty)> = vec![
        ("Bool", &Ty::Bool),
        ("Int8", &Ty::Int(I8)),
        ("Int16", &Ty::Int(I16)),
        ("Int32", &Ty::Int(I32)),
        ("Int64", &Ty::Int(I64)),
        ("UInt8", &Ty::UInt(U8)),
        ("UInt16", &Ty::UInt(U16)),
        ("UInt32", &Ty::UInt(U32)),
        ("UInt64", &Ty::UInt(U64)),
        ("Float32", &Ty::Float(F32)),
        ("Float64", &Ty::Float(F64)),
        ("String", &Ty::String),
    ];
}

impl Ty {
    pub(crate) fn primitive_types() -> std::slice::Iter<'static, (&'static str, &'static Ty)> {
        PRIMITIVE_TYPES.iter()
    }

    pub(crate) fn satisfies(&self, constraint: &TypeConstraint) -> bool {
        use self::Ty::*;
        use self::TypeConstraint::*;
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
            TypeConstraint::Stream(ty_l) => match self {
                Ty::EventStream(ty, _) => ty == ty_l,
                Ty::TimedStream(ty, _) => ty == ty_l,
                _ => false,
            },
        }
    }

    pub(crate) fn is_error(&self) -> bool {
        use self::Ty::*;
        match self {
            Error => true,
            Tuple(args) => args.iter().any(|el| el.is_error()),
            EventStream(ty, _) => ty.is_error(),
            Parameterized(ty, params) => ty.is_error() || params.iter().any(|e| e.is_error()),
            TimedStream(ty, _) => ty.is_error(),
            Option(ty) => ty.is_error(),
            _ => false,
        }
    }

    pub(crate) fn is_primitive(&self) -> bool {
        use self::Ty::*;
        match self {
            Bool | Int(_) | UInt(_) | Float(_) | String => true,
            _ => false,
        }
    }

    pub(crate) fn replace_params(&self, infer_vars: &[InferVar]) -> Ty {
        match self {
            &Ty::Param(id, _) => Ty::Infer(infer_vars[id as usize]),
            Ty::EventStream(t, deps) => {
                Ty::EventStream(t.replace_params(infer_vars).into(), deps.clone())
            }
            Ty::Parameterized(t, params) => Ty::Parameterized(
                Box::new(t.replace_params(infer_vars)),
                params
                    .iter()
                    .map(|e| e.replace_params(infer_vars))
                    .collect(),
            ),
            Ty::Option(t) => Ty::Option(Box::new(t.replace_params(infer_vars))),
            Ty::Window(t, d) => Ty::Window(Box::new(t.replace_params(infer_vars)), *d),
            Ty::Infer(_) => self.clone(),
            Ty::Constr(_) => self.clone(),
            _ if self.is_primitive() => self.clone(),
            _ => unimplemented!("replace_param for {}", self),
        }
    }
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Ty::Bool => write!(f, "Bool"),
            Ty::Int(I8) => write!(f, "Int8"),
            Ty::Int(I16) => write!(f, "Int16"),
            Ty::Int(I32) => write!(f, "Int32"),
            Ty::Int(I64) => write!(f, "Int64"),
            Ty::UInt(U8) => write!(f, "UInt8"),
            Ty::UInt(U16) => write!(f, "UInt16"),
            Ty::UInt(U32) => write!(f, "UInt32"),
            Ty::UInt(U64) => write!(f, "UInt64"),
            Ty::Float(F32) => write!(f, "Float32"),
            Ty::Float(F64) => write!(f, "Float64"),
            Ty::String => write!(f, "String"),
            Ty::EventStream(ty, deps) => {
                if deps.is_empty() {
                    write!(f, "EventStream<{}>", ty)
                } else {
                    let joined: Vec<String> = deps.iter().map(|e| format!("{}", e)).collect();
                    write!(f, "EventStream<{}, {}>", ty, joined.join(", "))
                }
            }
            Ty::Parameterized(ty, params) => {
                let joined: Vec<String> = params.iter().map(|e| format!("{}", e)).collect();
                write!(f, "Paramaterized<{}, ({})>", ty, joined.join(", "))
            }
            Ty::TimedStream(ty, freq) => write!(f, "TimedStream<{}, {}>", ty, freq),
            Ty::Window(t, d) => write!(f, "Window<{}, {:?}>", t, d),
            Ty::Option(ty) => write!(f, "Option<{}>", ty),
            Ty::Tuple(inner) => {
                let joined: Vec<String> = inner.iter().map(|e| format!("{}", e)).collect();
                write!(f, "({})", joined.join(", "))
            }
            Ty::Infer(id) => write!(f, "?{}", id),
            Ty::Constr(constr) => write!(f, "{{{}}}", constr),
            Ty::Param(_, name) => write!(f, "{}", name),
            Ty::Error => write!(f, "Error"),
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
    /// A stream of given type
    Stream(Box<Ty>),
    Unconstrained,
}

impl TypeConstraint {
    pub(crate) fn has_default(self) -> Option<Ty> {
        use self::TypeConstraint::*;
        match self {
            Integer | SignedInteger | Numeric => Some(Ty::Int(I32)),
            UnsignedInteger => Some(Ty::UInt(U32)),
            FloatingPoint => Some(Ty::Float(F32)),
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
            Stream(_) => None,
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
            Stream(ty) => write!(f, "Stream<{}>", ty),
            Unconstrained => write!(f, "unconstrained type"),
        }
    }
}
