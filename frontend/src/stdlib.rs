//! This module contains the Lola standard library.

use crate::analysis::naming::ScopedDecl;
use crate::ast::{BinOp, FunctionName, UnOp};
use crate::ty::{FloatTy, TypeConstraint, ValueTy};
use lazy_static::lazy_static;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Generic {
    pub constraint: ValueTy,
}

/// A (possibly generic) function declaration
#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub name: FunctionName,
    pub generics: Vec<Generic>,
    pub parameters: Vec<ValueTy>,
    pub return_type: ValueTy,
}

impl FuncDecl {
    /// Given the instantiation of the generic parameters, this function returns the instantiated types of the arguments and return type.
    /// For example `sqrt<T>(_: T) -> T` is `sqrt<T>(_: Float32) -> Float32` when `T` is instantiated by `Float32`.
    pub(crate) fn get_types_for_args_and_ret(&self, generics: &[ValueTy]) -> (Vec<ValueTy>, ValueTy) {
        let args = self.parameters.iter().map(|ty| ty.replace_params_with_ty(generics)).collect();
        (args, self.return_type.replace_params_with_ty(generics))
    }
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

pub(crate) fn import_math_method(lookup: &mut MethodLookup) {
    lookup.add(ValueTy::Float(FloatTy::F32), &SQRT);
    lookup.add(ValueTy::Float(FloatTy::F32), &COS);
    lookup.add(ValueTy::Float(FloatTy::F32), &SIN);
    lookup.add(ValueTy::Float(FloatTy::F32), &ABS);
    lookup.add(ValueTy::Float(FloatTy::F32), &ARCTAN);

    lookup.add(ValueTy::Float(FloatTy::F64), &SQRT);
    lookup.add(ValueTy::Float(FloatTy::F64), &COS);
    lookup.add(ValueTy::Float(FloatTy::F64), &SIN);
    lookup.add(ValueTy::Float(FloatTy::F64), &ABS);
    lookup.add(ValueTy::Float(FloatTy::F64), &ARCTAN);
}

pub(crate) fn import_regex_method(lookup: &mut MethodLookup) {
    lookup.add(ValueTy::String, &MATCHES_REGEX);
}

pub(crate) struct MethodLookup {
    lookup_table: HashMap<ValueTy, HashMap<String, FuncDecl>>,
}

impl MethodLookup {
    pub(crate) fn new() -> MethodLookup {
        MethodLookup { lookup_table: HashMap::new() }
    }

    pub(crate) fn add(&mut self, ty: ValueTy, decl: &FuncDecl) {
        let entry = self.lookup_table.entry(ty).or_insert_with(HashMap::new);
        let mut name = decl.name.clone();
        assert!(name.arg_names[0] == None);
        name.arg_names.remove(0);
        entry.insert(name.to_string(), decl.clone());
    }

    pub(crate) fn get(&self, ty: &ValueTy, name: &FunctionName) -> Option<FuncDecl> {
        self.lookup_table.get(ty).and_then(|func_decls| func_decls.get(&name.to_string())).map(|decl| decl.clone())
    }
}
