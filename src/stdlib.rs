//! This module contains the Lola standard library.

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
            // offset<T,I:Integer>(EventStream<T>, I) -> Option<T>
            (Ty::EventStream(_), "offset") => Some(FuncDecl {
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
                    Ty::EventStream(Box::new(Ty::Param(0, "T".to_string()))),
                    Ty::Param(1, "I".to_string()),
                ],
                return_type: Ty::Option(Box::new(Ty::Param(0, "T".to_string()))),
            }),
            // window<T,D: Duration>(EventStream<T>, D) -> Window<T, D>
            (Ty::EventStream(_), "window") => Some(FuncDecl {
                name: "window".to_string(),
                generics: vec![
                    Generic {
                        constraint: Ty::Constr(TypeConstraint::Unconstrained),
                    },
                    Generic {
                        constraint: Ty::Constr(TypeConstraint::Duration),
                    },
                ],
                parameters: vec![Ty::EventStream(Box::new(Ty::Param(0, "T".to_string())))],
                return_type: Ty::Window(
                    Box::new(Ty::Param(0, "T".to_string())),
                    Box::new(Ty::Param(1, "D".to_string())),
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
            _ => unimplemented!("{} for {}", name, ty),
        }
    }
}
