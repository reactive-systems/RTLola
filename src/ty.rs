//! This module contains the basic definition of types
//!
//! It is inspired by https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/index.html

use crate::analysis::typing::InferVar;

/// Representation of types
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum Ty {
    Bool,
    Int(IntTy),
    UInt(UIntTy),
    Float(FloatTy),
    // an abstract data type, e.g., structs, enums, etc.
    //Adt(AdtDef),
    String,
    Tuple(Vec<Ty>),
    EventStream(Box<Ty>), // todo: probably need info if parametric
    TimedStream(Box<Ty>), // todo: probably need frequency as well
    /// an optional value type, e.g., accessing a stream with offset -1
    Option(Box<Ty>),
    /// Used during type inference
    Infer(InferVar),
    Constr(TypeConstraint),
    Error,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
}
use self::IntTy::*;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum UIntTy {
    U8,
    U16,
    U32,
    U64,
}
use self::UIntTy::*;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum FloatTy {
    F32,
    F64,
}
use self::FloatTy::*;

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

    pub(crate) fn satisfies(&self, constraint: TypeConstraint) -> bool {
        use self::Ty::*;
        use self::TypeConstraint::*;
        match constraint {
            Unconstrained => true,
            Comparable => self.is_primitive(),
            Equatable => self.is_primitive(),
            Numeric => self.satisfies(Integer) || self.satisfies(FloatingPoint),
            FloatingPoint => match self {
                Float(_) => true,
                _ => false,
            },
            Integer => self.satisfies(SignedInteger) || self.satisfies(UnsignedInteger),
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
        use self::Ty::*;
        match self {
            Error => true,
            Tuple(args) => args.iter().fold(false, |val, el| val || el.is_error()),
            EventStream(ty) => ty.is_error(),
            TimedStream(ty) => ty.is_error(),
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

    pub(crate) fn can_be_converted_to(&self, candidate: &Ty) -> bool {
        match self {
            Ty::EventStream(ty) => ty.as_ref() == candidate,
            _ => false,
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
            Ty::EventStream(ty) => write!(f, "EventStream<{}>", ty),
            Ty::TimedStream(_) => unimplemented!(),
            Ty::Option(ty) => write!(f, "Option<{}>", ty),
            Ty::Tuple(inner) => {
                let joined: Vec<String> = inner.iter().map(|e| format!("{}", e)).collect();
                write!(f, "({})", joined.join(", "))
            }
            Ty::Infer(id) => write!(f, "?{}", id),
            Ty::Constr(constr) => write!(f, "{{{}}}", constr),
            Ty::Error => write!(f, "Error"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    pub(crate) fn has_default(&self) -> Option<Ty> {
        use self::TypeConstraint::*;
        match self {
            Integer | SignedInteger | Numeric => Some(Ty::Int(I32)),
            UnsignedInteger => Some(Ty::UInt(U32)),
            FloatingPoint => Some(Ty::Float(F32)),
            _ => None,
        }
    }

    pub(crate) fn conjunction(&self, other: TypeConstraint) -> Option<TypeConstraint> {
        use self::TypeConstraint::*;
        if *self > other {
            return other.conjunction(*self);
        }
        if *self == other {
            return Some(*self);
        }
        assert!(*self < other);
        match other {
            Unconstrained => Some(*self),
            Comparable => Some(*self),
            Equatable => Some(*self),
            Numeric => Some(*self),
            Integer => match self {
                FloatingPoint => None,
                _ => Some(*self),
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
            _ => unimplemented!(),
        }
    }
}
