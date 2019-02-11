//! This module contains the Lola standard library.

use crate::ast::{BinOp, UnOp};
use crate::ty::{TypeConstraint, ValueTy};
use lazy_static::lazy_static;

#[derive(Debug)]
pub struct Generic {
    pub constraint: ValueTy,
}

/// A (possibly generic) function declaration
#[derive(Debug)]
pub struct FuncDecl {
    pub name: String,
    pub generics: Vec<Generic>,
    pub parameters: Vec<ValueTy>,
    pub return_type: ValueTy,
}

impl BinOp {
    pub(crate) fn get_func_decl(self) -> FuncDecl {
        use self::BinOp::*;
        match self {
            Add | Sub | Mul | Div | Rem | Pow => FuncDecl {
                name: format!("{}", self),
                generics: vec![Generic {
                    constraint: ValueTy::Constr(TypeConstraint::Numeric),
                }],
                parameters: vec![
                    ValueTy::Param(0, "T".to_string()),
                    ValueTy::Param(0, "T".to_string()),
                ],
                return_type: ValueTy::Param(0, "T".to_string()),
            },
            And | Or => FuncDecl {
                name: format!("{}", self),
                generics: vec![],
                parameters: vec![ValueTy::Bool, ValueTy::Bool],
                return_type: ValueTy::Bool,
            },
            Eq | Ne => FuncDecl {
                name: format!("{}", self),
                generics: vec![Generic {
                    constraint: ValueTy::Constr(TypeConstraint::Equatable),
                }],
                parameters: vec![
                    ValueTy::Param(0, "T".to_string()),
                    ValueTy::Param(0, "T".to_string()),
                ],
                return_type: ValueTy::Bool,
            },
            Lt | Le | Ge | Gt => FuncDecl {
                name: format!("{}", self),
                generics: vec![Generic {
                    constraint: ValueTy::Constr(TypeConstraint::Comparable),
                }],
                parameters: vec![
                    ValueTy::Param(0, "T".to_string()),
                    ValueTy::Param(0, "T".to_string()),
                ],
                return_type: ValueTy::Bool,
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
                parameters: vec![ValueTy::Bool],
                return_type: ValueTy::Bool,
            },
            Neg => FuncDecl {
                name: format!("{}", self),
                generics: vec![Generic {
                    constraint: ValueTy::Constr(TypeConstraint::Numeric),
                }],
                parameters: vec![ValueTy::Param(0, "T".to_string())],
                return_type: ValueTy::Param(0, "T".to_string()),
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
            constraint: ValueTy::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
    // fn cos<T: FloatingPoint>(T) -> T
    static ref COS: FuncDecl = FuncDecl {
        name: "cos".to_string(),
        generics: vec![Generic {
            constraint: ValueTy::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
    // fn sin<T: FloatingPoint>(T) -> T
    static ref SIN: FuncDecl = FuncDecl {
        name: "sin".to_string(),
        generics: vec![Generic {
            constraint: ValueTy::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
     // fn arctan<T: FloatingPoint>(T) -> T
    static ref ARCTAN: FuncDecl = FuncDecl {
        name: "arctan".to_string(),
        generics: vec![Generic {
            constraint: ValueTy::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
    // fn abs<T: Numeric>(T) -> T
    static ref ABS: FuncDecl = FuncDecl {
        name: "abs".to_string(),
        generics: vec![Generic {
            constraint: ValueTy::Constr(TypeConstraint::Numeric),
        }],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };

    // fn matches_regexp(String, String) -> Bool
    static ref MATCHES_REGEX: FuncDecl = FuncDecl {
        name: "matches_regex".to_string(),
        generics: vec![],
        parameters: vec![ValueTy::String, ValueTy::String],
        return_type: ValueTy::Bool,
    };

    /// fn cast<T: Numeric, U: Numeric>(T) -> U
    /// allows for arbitrary conversion of numeric types T -> U
    static ref CAST: FuncDecl = FuncDecl {
        name: "cast".to_string(),
        generics: vec![Generic {
            constraint: ValueTy::Constr(TypeConstraint::Numeric),
        }, Generic {
            constraint: ValueTy::Constr(TypeConstraint::Numeric),
        }],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(1, "U".to_string()),
    };
}

pub(crate) fn import_implicit_module<'a>(scope: &mut ScopedDecl<'a>) {
    scope.add_decl_for(&CAST.name, Declaration::Func(&CAST));
}

pub(crate) fn import_math_module<'a>(scope: &mut ScopedDecl<'a>) {
    scope.add_decl_for(&SQRT.name, Declaration::Func(&SQRT));
    scope.add_decl_for(&COS.name, Declaration::Func(&COS));
    scope.add_decl_for(&SIN.name, Declaration::Func(&SIN));
    scope.add_decl_for(&ABS.name, Declaration::Func(&ABS));
    scope.add_decl_for(&ARCTAN.name, Declaration::Func(&ARCTAN));
}

pub(crate) fn import_regex_module<'a>(scope: &mut ScopedDecl<'a>) {
    scope.add_decl_for(&MATCHES_REGEX.name, Declaration::Func(&MATCHES_REGEX))
}

pub(crate) struct MethodLookup {}

impl MethodLookup {
    pub(crate) fn new() -> MethodLookup {
        MethodLookup {}
    }

    pub(crate) fn get(&self, ty: &ValueTy, name: &str) -> Option<FuncDecl> {
        unimplemented!("method lookup {} {}", ty, name)
    }
}
