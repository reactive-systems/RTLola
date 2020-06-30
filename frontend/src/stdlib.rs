//! This module contains the Lola standard library.

use crate::analysis::naming::ScopedDecl;
use crate::ast::{BinOp, FunctionName, UnOp};
use crate::ty::{FloatTy, IntTy, TypeConstraint, UIntTy, ValueTy};
use lazy_static::lazy_static;
use std::collections::HashMap;

/// A (possibly generic) function declaration
#[derive(Debug, Clone)]
pub(crate) struct FuncDecl {
    pub(crate) name: FunctionName,
    pub(crate) generics: Vec<ValueTy>,
    pub(crate) parameters: Vec<ValueTy>,
    pub(crate) return_type: ValueTy,
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
                generics: vec![ValueTy::Constr(TypeConstraint::Numeric)],
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
                generics: vec![ValueTy::Constr(TypeConstraint::Equatable)],
                parameters: vec![ValueTy::Param(0, "T".to_string()), ValueTy::Param(0, "T".to_string())],
                return_type: ValueTy::Bool,
            },
            Lt | Le | Ge | Gt => FuncDecl {
                name: FunctionName::new(format!("{}", self), &[None, None]),
                generics: vec![ValueTy::Constr(TypeConstraint::Comparable)],
                parameters: vec![ValueTy::Param(0, "T".to_string()), ValueTy::Param(0, "T".to_string())],
                return_type: ValueTy::Bool,
            },
            BitAnd | BitOr | BitXor => FuncDecl {
                name: FunctionName::new(format!("{}", self), &[None, None]),
                generics: vec![ValueTy::Constr(TypeConstraint::Integer)],
                parameters: vec![ValueTy::Param(0, "T".to_string()), ValueTy::Param(0, "T".to_string())],
                return_type: ValueTy::Param(0, "T".to_string()),
            },
            Shl | Shr => FuncDecl {
                name: FunctionName::new(format!("{}", self), &[None, None]),
                generics: vec![
                    ValueTy::Constr(TypeConstraint::Integer),
                    ValueTy::Constr(TypeConstraint::UnsignedInteger),
                ],
                parameters: vec![ValueTy::Param(0, "T".to_string()), ValueTy::Param(0, "U".to_string())],
                return_type: ValueTy::Param(0, "T".to_string()),
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
                generics: vec![ValueTy::Constr(TypeConstraint::Numeric)],
                parameters: vec![ValueTy::Param(0, "T".to_string())],
                return_type: ValueTy::Param(0, "T".to_string()),
            },
            BitNot => FuncDecl {
                name: FunctionName::new(format!("{}", self), &[None]),
                generics: vec![ValueTy::Constr(TypeConstraint::Integer)],
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
        generics: vec![ValueTy::Constr(TypeConstraint::FloatingPoint),
        ],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
    // fn min<T: Numeric>(T, T) -> T
    static ref MIN: FuncDecl = FuncDecl {
        name: FunctionName::new("min".to_string(), &[None, None]),
        generics: vec![ValueTy::Constr(TypeConstraint::Numeric)],
        parameters: vec![ValueTy::Param(0, "T".to_string()), ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
    // fn max<T: Numeric>(T, T) -> T
    static ref MAX: FuncDecl = FuncDecl {
        name: FunctionName::new("max".to_string(), &[None, None]),
        generics: vec![ValueTy::Constr(TypeConstraint::Numeric)],
        parameters: vec![ValueTy::Param(0, "T".to_string()), ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
    // fn cos<T: FloatingPoint>(T) -> T
    static ref COS: FuncDecl = FuncDecl {
        name: FunctionName::new("cos".to_string(), &[None]),
        generics: vec![ValueTy::Constr(TypeConstraint::FloatingPoint),
        ],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
    // fn sin<T: FloatingPoint>(T) -> T
    static ref SIN: FuncDecl = FuncDecl {
        name: FunctionName::new("sin".to_string(), &[None]),
        generics: vec![ValueTy::Constr(TypeConstraint::FloatingPoint),
        ],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
     // fn arctan<T: FloatingPoint>(T) -> T
    static ref ARCTAN: FuncDecl = FuncDecl {
        name: FunctionName::new("arctan".to_string(), &[None]),
        generics: vec![ ValueTy::Constr(TypeConstraint::FloatingPoint),
        ],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };
    // fn abs<T: Numeric>(T) -> T
    static ref ABS: FuncDecl = FuncDecl {
        name: FunctionName::new("abs".to_string(), &[None]),
        generics: vec![ValueTy::Constr(TypeConstraint::Numeric),
        ],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(0, "T".to_string()),
    };

    // fn matches(String, regex: String) -> Bool
    static ref MATCHES_STRING_REGEX: FuncDecl = FuncDecl {
        name: FunctionName::new("matches".to_string(), &[None, Some("regex".to_string())]),
        generics: vec![],
        parameters: vec![ValueTy::String, ValueTy::String],
        return_type: ValueTy::Bool,
    };

    // fn matches(Bytes, regex: String) -> Bool
    static ref MATCHES_BYTES_REGEX: FuncDecl = FuncDecl {
        name: FunctionName::new("matches".to_string(), &[None, Some("regex".to_string())]),
        generics: vec![],
        parameters: vec![ValueTy::Bytes, ValueTy::String],
        return_type: ValueTy::Bool,
    };

    /// fn cast<T: Numeric, U: Numeric>(T) -> U
    /// allows for arbitrary conversion of numeric types T -> U
    static ref CAST: FuncDecl = FuncDecl {
        name: FunctionName::new("cast".to_string(), &[None]),
        generics: vec![ValueTy::Constr(TypeConstraint::Numeric),
             ValueTy::Constr(TypeConstraint::Numeric),
        ],
        parameters: vec![ValueTy::Param(0, "T".to_string())],
        return_type: ValueTy::Param(1, "U".to_string()),
    };

    /// access index of byte array
    static ref BYTES_AT: FuncDecl = FuncDecl {
        name: FunctionName::new("at".to_string(), &[None, Some("index".to_string())]),
        generics: vec![],
        parameters: vec![ValueTy::Bytes, ValueTy::UInt(UIntTy::U64)],
        return_type: ValueTy::Option( ValueTy::UInt(UIntTy::U8).into() ),
    };
}

pub(crate) fn import_implicit_module(fun_scope: &mut ScopedDecl) {
    fun_scope.add_fun_decl(&CAST);
}

pub(crate) fn import_implicit_method(lookup: &mut MethodLookup) {
    lookup.add(ValueTy::Bytes, &BYTES_AT);
}

pub(crate) fn import_math_module(fun_scope: &mut ScopedDecl) {
    fun_scope.add_fun_decl(&SQRT);
    fun_scope.add_fun_decl(&COS);
    fun_scope.add_fun_decl(&SIN);
    fun_scope.add_fun_decl(&ABS);
    fun_scope.add_fun_decl(&ARCTAN);
    fun_scope.add_fun_decl(&MIN);
    fun_scope.add_fun_decl(&MAX);
}

pub(crate) fn import_regex_module(fun_scope: &mut ScopedDecl) {
    fun_scope.add_fun_decl(&MATCHES_STRING_REGEX);
}

pub(crate) fn import_math_method(lookup: &mut MethodLookup) {
    lookup.add(ValueTy::Float(FloatTy::F16), &SQRT);
    lookup.add(ValueTy::Float(FloatTy::F16), &COS);
    lookup.add(ValueTy::Float(FloatTy::F16), &SIN);
    lookup.add(ValueTy::Float(FloatTy::F16), &ABS);
    lookup.add(ValueTy::Float(FloatTy::F16), &ARCTAN);

    lookup.add(ValueTy::Float(FloatTy::F32), &SQRT);
    lookup.add(ValueTy::Float(FloatTy::F32), &COS);
    lookup.add(ValueTy::Float(FloatTy::F32), &SIN);
    lookup.add(ValueTy::Float(FloatTy::F32), &ABS);
    lookup.add(ValueTy::Float(FloatTy::F32), &ARCTAN);
    lookup.add(ValueTy::Float(FloatTy::F32), &MIN);
    lookup.add(ValueTy::Float(FloatTy::F32), &MAX);

    lookup.add(ValueTy::Float(FloatTy::F64), &SQRT);
    lookup.add(ValueTy::Float(FloatTy::F64), &COS);
    lookup.add(ValueTy::Float(FloatTy::F64), &SIN);
    lookup.add(ValueTy::Float(FloatTy::F64), &ABS);
    lookup.add(ValueTy::Float(FloatTy::F64), &ARCTAN);
    lookup.add(ValueTy::Float(FloatTy::F64), &MIN);
    lookup.add(ValueTy::Float(FloatTy::F64), &MAX);

    lookup.add(ValueTy::Int(IntTy::I8), &MIN);
    lookup.add(ValueTy::Int(IntTy::I8), &MAX);
    lookup.add(ValueTy::Int(IntTy::I16), &MIN);
    lookup.add(ValueTy::Int(IntTy::I16), &MAX);
    lookup.add(ValueTy::Int(IntTy::I32), &MIN);
    lookup.add(ValueTy::Int(IntTy::I32), &MAX);
    lookup.add(ValueTy::Int(IntTy::I64), &MIN);
    lookup.add(ValueTy::Int(IntTy::I64), &MAX);

    lookup.add(ValueTy::UInt(UIntTy::U8), &MIN);
    lookup.add(ValueTy::UInt(UIntTy::U8), &MAX);
    lookup.add(ValueTy::UInt(UIntTy::U16), &MIN);
    lookup.add(ValueTy::UInt(UIntTy::U16), &MAX);
    lookup.add(ValueTy::UInt(UIntTy::U32), &MIN);
    lookup.add(ValueTy::UInt(UIntTy::U32), &MAX);
    lookup.add(ValueTy::UInt(UIntTy::U64), &MIN);
    lookup.add(ValueTy::UInt(UIntTy::U64), &MAX);
}

pub(crate) fn import_regex_method(lookup: &mut MethodLookup) {
    lookup.add(ValueTy::String, &MATCHES_STRING_REGEX);
    lookup.add(ValueTy::Bytes, &MATCHES_BYTES_REGEX);
}

pub(crate) struct MethodLookup<'a> {
    lookup_table: HashMap<ValueTy, HashMap<String, &'a FuncDecl>>,
}

impl<'a> MethodLookup<'a> {
    pub(crate) fn new() -> Self {
        MethodLookup { lookup_table: HashMap::new() }
    }

    pub(crate) fn add(&mut self, ty: ValueTy, decl: &'a FuncDecl) {
        let entry = self.lookup_table.entry(ty).or_insert_with(HashMap::new);
        let mut name = decl.name.clone();
        assert!(name.arg_names[0] == None);
        name.arg_names.remove(0);
        let key = name.to_string();
        assert!(!entry.contains_key(&key));
        entry.insert(key, decl);
    }

    pub(crate) fn get(&self, ty: &ValueTy, name: &FunctionName) -> Option<&'a FuncDecl> {
        self.lookup_table.get(ty).and_then(|func_decls| func_decls.get(&name.to_string())).cloned()
        // cloned for dereferencing once && -> &
    }
}
