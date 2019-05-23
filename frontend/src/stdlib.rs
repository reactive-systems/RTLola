//! This module contains the Lola standard library.

use crate::ast::{BinOp, FunctionName, UnOp};
use crate::ty::{TypeConstraint, ValueTy};
use lazy_static::lazy_static;

#[derive(Debug)]
pub struct Generic {
    pub constraint: ValueTy,
}

/// A (possibly generic) function declaration
#[derive(Debug)]
pub struct FuncDecl {
    pub name: FunctionName,
    pub generics: Vec<Generic>,
    pub parameters: Vec<ValueTy>,
    pub return_type: ValueTy,
}

impl BinOp {
    pub(crate) fn get_func_decl(self) -> FuncDecl {
        use self::BinOp::*;
        match self {
            Add | Sub | Mul | Div | Rem | Pow => FuncDecl {
                name: FunctionName::new(format!("{}", self), &[None, None]),
                generics: vec![Generic { constraint: ValueTy::Constr(TypeConstraint::Numeric) }],
                parameters: vec![ValueTy::Param(0, "T".to_string()), ValueTy::Param(0, "T".to_string())],
                return_type: ValueTy::Param(0, "T".to_string()),
            },
            And | Or => FuncDecl {
                name: FunctionName::new(format!("{}", self), &[None, None]),
                generics: vec![],
                parameters: vec![ValueTy::Bool, ValueTy::Bool],
                return_type: ValueTy::Bool,
            },
            Eq | Ne => FuncDecl {
                name: FunctionName::new(format!("{}", self), &[None, None]),
                generics: vec![Generic { constraint: ValueTy::Constr(TypeConstraint::Equatable) }],
                parameters: vec![ValueTy::Param(0, "T".to_string()), ValueTy::Param(0, "T".to_string())],
                return_type: ValueTy::Bool,
            },
            Lt | Le | Ge | Gt => FuncDecl {
                name: FunctionName::new(format!("{}", self), &[None, None]),
                generics: vec![Generic { constraint: ValueTy::Constr(TypeConstraint::Comparable) }],
                parameters: vec![ValueTy::Param(0, "T".to_string()), ValueTy::Param(0, "T".to_string())],
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
                name: FunctionName::new(format!("{}", self), &[None]),
                generics: vec![],
                parameters: vec![ValueTy::Bool],
                return_type: ValueTy::Bool,
            },
            Neg => FuncDecl {
                name: FunctionName::new(format!("{}", self), &[None]),
                generics: vec![Generic { constraint: ValueTy::Constr(TypeConstraint::Numeric) }],
                parameters: vec![ValueTy::Param(0, "T".to_string())],
                return_type: ValueTy::Param(0, "T".to_string()),
            },
        }
    }
}

use crate::analysis::naming::ScopedDecl;

lazy_static! {
    // fn sqrt<T: FloatingPoint>(T) -> T
    static ref SQRT: FuncDecl = FuncDecl {
        name: FunctionName::new("sqrt".to_string(), &[None]),
        generics: vec![Generic {
            constraint: ValueTy::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
    // fn cos<T: FloatingPoint>(T) -> T
    static ref COS: FuncDecl = FuncDecl {
        name: FunctionName::new("cos".to_string(), &[None]),
        generics: vec![Generic {
            constraint: ValueTy::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
    // fn sin<T: FloatingPoint>(T) -> T
    static ref SIN: FuncDecl = FuncDecl {
        name: FunctionName::new("sin".to_string(), &[None]),
        generics: vec![Generic {
            constraint: ValueTy::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
     // fn arctan<T: FloatingPoint>(T) -> T
    static ref ARCTAN: FuncDecl = FuncDecl {
        name: FunctionName::new("arctan".to_string(), &[None]),
        generics: vec![Generic {
            constraint: ValueTy::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
    // fn abs<T: Numeric>(T) -> T
    static ref ABS: FuncDecl = FuncDecl {
        name: FunctionName::new("abs".to_string(), &[None]),
        generics: vec![Generic {
            constraint: ValueTy::Constr(TypeConstraint::Numeric),
        }],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };

    // fn matches(String, regex: String) -> Bool
    static ref MATCHES_REGEX: FuncDecl = FuncDecl {
        name: FunctionName::new("matches".to_string(), &[None, Some("regex".to_string())]),
        generics: vec![],
        parameters: vec![ValueTy::String, ValueTy::String],
        return_type: ValueTy::Bool,
    };

    /// fn cast<T: Numeric, U: Numeric>(T) -> U
    /// allows for arbitrary conversion of numeric types T -> U
    static ref CAST: FuncDecl = FuncDecl {
        name: FunctionName::new("cast".to_string(), &[None]),
        generics: vec![Generic {
            constraint: ValueTy::Constr(TypeConstraint::Numeric),
        }, Generic {
            constraint: ValueTy::Constr(TypeConstraint::Numeric),
        }],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(1, "U".to_string()),
    };
}

pub(crate) fn import_implicit_module<'a>(fun_scope: &mut ScopedDecl<'a>) {
    fun_scope.add_fun_decl(&CAST);
}

pub(crate) fn import_math_module<'a>(fun_scope: &mut ScopedDecl<'a>) {
    fun_scope.add_fun_decl(&SQRT);
    fun_scope.add_fun_decl(&COS);
    fun_scope.add_fun_decl(&SIN);
    fun_scope.add_fun_decl(&ABS);
    fun_scope.add_fun_decl(&ARCTAN);
}

pub(crate) fn import_regex_module<'a>(fun_scope: &mut ScopedDecl<'a>) {
    fun_scope.add_fun_decl(&MATCHES_REGEX);
}

pub(crate) struct MethodLookup {}

impl MethodLookup {
    pub(crate) fn new() -> MethodLookup {
        MethodLookup {}
    }

    pub(crate) fn get(&self, ty: &ValueTy, name: &FunctionName) -> Option<FuncDecl> {
        unimplemented!("method lookup {} {}", ty, name)
    }
}
