//! This module contains the basic definition of types
//!
//! It is inspired by https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/index.html

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
    InvocationTime,
    Error,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum UIntTy {
    U8,
    U16,
    U32,
    U64,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum FloatTy {
    F32,
    F64,
}

#[derive(Debug)]
pub enum GenricTypeConstraint {
    SignedInteger,
    UnsignedInteger,
    FloatingPoint,
}
