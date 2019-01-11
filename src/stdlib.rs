//! This module contains the Lola standard library.

use crate::ast::{BinOp, UnOp};
use crate::ty::{Ty, TypeConstraint};

#[derive(Debug)]
pub struct Generic {
    pub constraint: Ty,
}

/// different kinds of type declarations, can be e.g., alias, newtype, struct, enum
#[derive(Debug)]
pub enum TypeDecl {
    //Alias(String, Ty),
//NewType(String, Ty),
//Struct(String, Vec<(String, Ty)>),
//Enum(String, Vec<(String, Ty)>),
}

/// A (possibly generic) function declaration
#[derive(Debug)]
pub struct FuncDecl {
    pub name: String,
    pub generics: Vec<Generic>,
    pub parameters: Vec<Ty>,
    pub return_type: Ty,
}

impl BinOp {
    pub(crate) fn get_func_decl(self) -> FuncDecl {
        use self::BinOp::*;
        match self {
            Add | Sub | Mul | Div | Rem | Pow => FuncDecl {
                name: format!("{}", self),
                generics: vec![Generic {
                    constraint: Ty::Constr(TypeConstraint::Numeric),
                }],
                parameters: vec![Ty::Param(0, "T".to_string()), Ty::Param(0, "T".to_string())],
                return_type: Ty::Param(0, "T".to_string()),
            },
            And | Or => FuncDecl {
                name: format!("{}", self),
                generics: vec![],
                parameters: vec![Ty::Bool, Ty::Bool],
                return_type: Ty::Bool,
            },
            Eq | Ne => FuncDecl {
                name: format!("{}", self),
                generics: vec![Generic {
                    constraint: Ty::Constr(TypeConstraint::Equatable),
                }],
                parameters: vec![Ty::Param(0, "T".to_string()), Ty::Param(0, "T".to_string())],
                return_type: Ty::Bool,
            },
            Lt | Le | Ge | Gt => FuncDecl {
                name: format!("{}", self),
                generics: vec![Generic {
                    constraint: Ty::Constr(TypeConstraint::Comparable),
                }],
                parameters: vec![Ty::Param(0, "T".to_string()), Ty::Param(0, "T".to_string())],
                return_type: Ty::Bool,
            },
        }
    }
}

impl UnOp {
    pub(crate) fn get_func_decl(self) -> FuncDecl {
        use self::UnOp::*;
        match self {
            Not => FuncDecl {
                name: format!("{}", self),
                generics: vec![],
                parameters: vec![Ty::Bool],
                return_type: Ty::Bool,
            },
            Neg => FuncDecl {
                name: format!("{}", self),
                generics: vec![Generic {
                    constraint: Ty::Constr(TypeConstraint::Numeric),
                }],
                parameters: vec![Ty::Param(0, "T".to_string())],
                return_type: Ty::Param(0, "T".to_string()),
            },
        }
    }
}

use crate::analysis::naming::{Declaration, ScopedDecl};

lazy_static! {
    // fn sqrt<T: FloatingPoint>(T) -> T
    static ref SQRT: FuncDecl = FuncDecl {
        name: "sqrt".to_string(),
        generics: vec![Generic {
            constraint: Ty::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![Ty::Param(0, "T".to_string())],
        return_type: Ty::Param(0, "T".to_string()),
    };
    // fn cos<T: FloatingPoint>(T) -> T
    static ref COS: FuncDecl = FuncDecl {
        name: "cos".to_string(),
        generics: vec![Generic {
            constraint: Ty::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![Ty::Param(0, "T".to_string())],
        return_type: Ty::Param(0, "T".to_string()),
    };
    // fn sin<T: FloatingPoint>(T) -> T
    static ref SIN: FuncDecl = FuncDecl {
        name: "sin".to_string(),
        generics: vec![Generic {
            constraint: Ty::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![Ty::Param(0, "T".to_string())],
        return_type: Ty::Param(0, "T".to_string()),
    };

    // fn matches_regexp(String, String) -> Bool
    static ref MATCHES_REGEX: FuncDecl = FuncDecl {
        name: "matches_regex".to_string(),
        generics: vec![],
        parameters: vec![Ty::String, Ty::String],
        return_type: Ty::Bool,
    };
}

pub(crate) fn import_math_module<'a>(scope: &mut ScopedDecl<'a>) {
    scope.add_decl_for(&SQRT.name, Declaration::Func(&SQRT));
    scope.add_decl_for(&COS.name, Declaration::Func(&COS));
    scope.add_decl_for(&SIN.name, Declaration::Func(&SIN));
}

pub(crate) fn import_regex_module<'a>(scope: &mut ScopedDecl<'a>) {
    scope.add_decl_for(&MATCHES_REGEX.name, Declaration::Func(&MATCHES_REGEX))
}

pub(crate) struct MethodLookup {}

impl MethodLookup {
    pub(crate) fn new() -> MethodLookup {
        MethodLookup {}
    }

    pub(crate) fn get(&self, ty: &Ty, name: &str) -> Option<FuncDecl> {
        unimplemented!("method lookup {} {}", ty, name)
    }
}
