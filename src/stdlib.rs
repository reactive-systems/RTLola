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
                parameters: vec![
                    Ty::EventStream(Ty::Param(0, "T".to_string()).into(), vec![]),
                    Ty::EventStream(Ty::Param(0, "T".to_string()).into(), vec![]),
                ],
                return_type: Ty::EventStream(Ty::Param(0, "T".to_string()).into(), vec![]),
            },
            And | Or => FuncDecl {
                name: format!("{}", self),
                generics: vec![],
                parameters: vec![
                    Ty::EventStream(Ty::Bool.into(), vec![]),
                    Ty::EventStream(Ty::Bool.into(), vec![]),
                ],
                return_type: Ty::EventStream(Ty::Bool.into(), vec![]),
            },
            Eq | Ne => FuncDecl {
                name: format!("{}", self),
                generics: vec![Generic {
                    constraint: Ty::Constr(TypeConstraint::Equatable),
                }],
                parameters: vec![
                    Ty::EventStream(Ty::Param(0, "T".to_string()).into(), vec![]),
                    Ty::EventStream(Ty::Param(0, "T".to_string()).into(), vec![]),
                ],
                return_type: Ty::EventStream(Ty::Bool.into(), vec![]),
            },
            Lt | Le | Ge | Gt => FuncDecl {
                name: format!("{}", self),
                generics: vec![Generic {
                    constraint: Ty::Constr(TypeConstraint::Comparable),
                }],
                parameters: vec![
                    Ty::EventStream(Ty::Param(0, "T".to_string()).into(), vec![]),
                    Ty::EventStream(Ty::Param(0, "T".to_string()).into(), vec![]),
                ],
                return_type: Ty::EventStream(Ty::Bool.into(), vec![]),
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
                parameters: vec![Ty::EventStream(Ty::Bool.into(), vec![])],
                return_type: Ty::EventStream(Ty::Bool.into(), vec![]),
            },
            Neg => FuncDecl {
                name: format!("{}", self),
                generics: vec![Generic {
                    constraint: Ty::Constr(TypeConstraint::Numeric),
                }],
                parameters: vec![Ty::EventStream(
                    Ty::Param(0, "T".to_string()).into(),
                    vec![],
                )],
                return_type: Ty::EventStream(Ty::Param(0, "T".to_string()).into(), vec![]),
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
        parameters: vec![Ty::EventStream(
            Ty::Param(0, "T".to_string()).into(),
            vec![],
        )],
        return_type: Ty::EventStream(
            Ty::Param(0, "T".to_string()).into(),
            vec![],
        ),
    };
    // fn cos<T: FloatingPoint>(T) -> T
    static ref COS: FuncDecl = FuncDecl {
        name: "cos".to_string(),
        generics: vec![Generic {
            constraint: Ty::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![Ty::EventStream(
            Ty::Param(0, "T".to_string()).into(),
            vec![],
        )],
        return_type: Ty::EventStream(
            Ty::Param(0, "T".to_string()).into(),
            vec![],
        ),
    };
    // fn sin<T: FloatingPoint>(T) -> T
    static ref SIN: FuncDecl = FuncDecl {
        name: "sin".to_string(),
        generics: vec![Generic {
            constraint: Ty::Constr(TypeConstraint::FloatingPoint),
        }],
        parameters: vec![Ty::EventStream(
            Ty::Param(0, "T".to_string()).into(),
            vec![],
        )],
        return_type: Ty::EventStream(
            Ty::Param(0, "T".to_string()).into(),
            vec![],
        ),
    };

    // fn matches_regexp(String, String) -> Bool
    static ref MATCHES_REGEX: FuncDecl = FuncDecl {
        name: "matches_regex".to_string(),
        generics: vec![],
        parameters: vec![Ty::EventStream(Ty::String.into(), vec![]), Ty::EventStream(Ty::String.into(), vec![])],
        return_type: Ty::EventStream(Ty::Bool.into(), vec![]),
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

    #[allow(unreachable_code)]
    #[allow(unused_variables)]
    pub(crate) fn get(&self, ty: &Ty, name: &str) -> Option<FuncDecl> {
        #[cfg(not(feature = "methods"))]
        {
            return None;
        }

        match ty {
            Ty::EventStream(inner, params) => {
                match (inner.as_ref(), params, name) {
                    // offset<T,I:Integer>(EventStream<T, ()>, I) -> EventStream<Option<T>>
                    (_, params, "offset") if params.is_empty() => Some(FuncDecl {
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
                        return_type: Ty::EventStream(
                            Ty::Option(Box::new(Ty::Param(0, "T".to_string()))).into(),
                            vec![],
                        ),
                    }),
                    // default<T>(EventStream<Option<T>>, EventStream<T>) -> EventStream<T>
                    (Ty::Option(_), params, "default") if params.is_empty() => Some(FuncDecl {
                        name: "default".to_string(),
                        generics: vec![Generic {
                            constraint: Ty::Constr(TypeConstraint::Unconstrained),
                        }],
                        parameters: vec![
                            Ty::EventStream(
                                Ty::Option(Box::new(Ty::Param(0, "T".to_string()))).into(),
                                vec![],
                            ),
                            Ty::EventStream(Ty::Param(0, "T".to_string()).into(), vec![]),
                        ],
                        return_type: Ty::EventStream(Ty::Param(0, "T".to_string()).into(), vec![]),
                    }),
                    _ => None,
                }
            }
            _ => unimplemented!("{} for {}", name, ty),
        }
    }
}
