//! This module contains the basic definition of types
//!
//! It is inspired by https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/index.html

use ast_node::NodeId;

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
    /// Used during type inference
    Infer(NodeId),
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

    pub(crate) fn satisfies(&self, constraint: GenricTypeConstraint) -> bool {
        use self::GenricTypeConstraint::*;
        use self::Ty::*;
        match (self, constraint) {
            (Float(_), FloatingPoint) => true,
            (Float(_), Integer) => false,
            (_, _) => unimplemented!(),
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
            Ty::Infer(id) => write!(f, "?{}", id),
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenricTypeConstraint {
    Integer,
    SignedInteger,
    UnsignedInteger,
    FloatingPoint,
}

impl std::fmt::Display for GenricTypeConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::GenricTypeConstraint::*;
        match self {
            Integer => write!(f, "integer"),
            FloatingPoint => write!(f, "floating point"),
            _ => unimplemented!(),
        }
    }
}
