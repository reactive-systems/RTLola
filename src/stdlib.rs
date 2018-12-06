//! This module contains the Lola standard library.

use crate::ty::{GenricTypeConstraint, Ty};

#[derive(Debug)]
struct Generic {
    constraint: GenricTypeConstraint,
}

#[derive(Debug)]
enum Parameter {
    Type(Ty),
    /// Index into associated generics array
    Generic(u8),
}

/// different kinds of type declarations, can be e.g., alias, newtype, struct, enum
#[derive(Debug)]
pub enum TypeDecl {
    //Alias(String, Ty),
//NewType(String, Ty),
//Struct(String, Vec<(String, Ty)>),
}

/// A (possibly generic) function declaration
#[derive(Debug)]
pub struct FuncDecl {
    name: String,
    generics: Vec<Generic>,
    parameters: Vec<Parameter>,
    return_type: Parameter,
}

use crate::analysis::naming::{Declaration, ScopedDecl};

lazy_static! {
    // fn sqrt<T: FloatingPoint>(T) -> T
    static ref SQRT: FuncDecl = FuncDecl {
        name: "sqrt".to_string(),
        generics: vec![Generic {
            constraint: GenricTypeConstraint::FloatingPoint,
        }],
        parameters: vec![Parameter::Generic(0)],
        return_type: Parameter::Generic(0),
    };
}

pub(crate) fn import_math_module<'a>(scope: &mut ScopedDecl<'a>) {
    scope.add_decl_for(&SQRT.name, Declaration::Func(&SQRT))
}
