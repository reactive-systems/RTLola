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
}

pub(crate) fn import_math_module<'a>(scope: &mut ScopedDecl<'a>) {
    scope.add_decl_for(&SQRT.name, Declaration::Func(&SQRT));
    scope.add_decl_for(&COS.name, Declaration::Func(&COS));
    scope.add_decl_for(&SIN.name, Declaration::Func(&SIN));
}

pub(crate) struct MethodLookup {}

impl MethodLookup {
    pub(crate) fn new() -> MethodLookup {
        MethodLookup {}
    }

    pub(crate) fn get(&self, ty: &Ty, name: &str) -> Option<FuncDecl> {
        match (ty, name) {
            // offset<T,I:Integer>(EventStream<T, ()>, I) -> Option<T>
            (Ty::EventStream(_, params), "offset") if params.is_empty() => Some(FuncDecl {
                name: "offset".to_string(),
                generics: vec![
                    Generic {
                        constraint: Ty::Constr(TypeConstraint::Unconstrained),
                    },
                    Generic {
                        constraint: Ty::Constr(TypeConstraint::Integer),
                    },
                ],
                parameters: vec![
                    Ty::EventStream(Box::new(Ty::Param(0, "T".to_string())), Vec::new()),
                    Ty::Param(1, "I".to_string()),
                ],
                return_type: Ty::Option(Box::new(Ty::Param(0, "T".to_string()))),
            }),
            // window<D: Duration, T>(EventStream<T, ()>, D) -> Window<T, D>
            (Ty::EventStream(_, params), "window") if params.is_empty() => Some(FuncDecl {
                name: "window".to_string(),
                generics: vec![
                    Generic {
                        constraint: Ty::Constr(TypeConstraint::Duration),
                    },
                    Generic {
                        constraint: Ty::Constr(TypeConstraint::Unconstrained),
                    },
                ],
                parameters: vec![Ty::EventStream(
                    Box::new(Ty::Param(1, "T".to_string())),
                    Vec::new(),
                )],
                return_type: Ty::Window(
                    Box::new(Ty::Param(1, "T".to_string())),
                    Box::new(Ty::Param(0, "D".to_string())),
                ),
            }),
            // default<T>(Option<T>, T) -> T
            (Ty::Option(_), "default") => Some(FuncDecl {
                name: "default".to_string(),
                generics: vec![Generic {
                    constraint: Ty::Constr(TypeConstraint::Unconstrained),
                }],
                parameters: vec![
                    Ty::Option(Box::new(Ty::Param(0, "T".to_string()))),
                    Ty::Param(0, "T".to_string()),
                ],
                return_type: Ty::Param(0, "T".to_string()),
            }),
            // sum<T:Numeric, D: Duration>(Window<T,D>) -> T
            (Ty::Window(_, _), "sum") => Some(FuncDecl {
                name: "sum".to_string(),
                generics: vec![
                    Generic {
                        constraint: Ty::Constr(TypeConstraint::Numeric),
                    },
                    Generic {
                        constraint: Ty::Constr(TypeConstraint::Duration),
                    },
                ],
                parameters: vec![Ty::Window(
                    Box::new(Ty::Param(0, "T".to_string())),
                    Box::new(Ty::Param(1, "D".to_string())),
                )],
                return_type: Ty::Param(0, "T".to_string()),
            }),
            // avg<T:Numeric, D: Duration, F: FloatingPoint>(Window<T,D>) -> Option<F>
            (Ty::Window(_, _), "avg") => Some(FuncDecl {
                name: "avg".to_string(),
                generics: vec![
                    Generic {
                        constraint: Ty::Constr(TypeConstraint::Numeric),
                    },
                    Generic {
                        constraint: Ty::Constr(TypeConstraint::Duration),
                    },
                    Generic {
                        constraint: Ty::Constr(TypeConstraint::FloatingPoint),
                    },
                ],
                parameters: vec![Ty::Window(
                    Box::new(Ty::Param(0, "T".to_string())),
                    Box::new(Ty::Param(1, "D".to_string())),
                )],
                return_type: Ty::Option(Box::new(Ty::Param(2, "F".to_string()))),
            }),
            _ => unimplemented!("{} for {}", name, ty),
        }
    }
}
