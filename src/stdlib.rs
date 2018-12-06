//! This module contains the Lola standard library.

trait TypeDecl {}

struct Generic {
    //constraint: Candidates,
}

enum Parameter {
    Ty(Box<dyn TypeDecl>),
    Generic(u8),
}

/// A (possibly generic) function declaration
pub struct FuncDecl {
    name: String,
    generics: Vec<Generic>,
    parameters: Vec<Parameter>,
    return_type: Parameter,
}

/// A definition of a Lola Module, that consists of types and functions
struct LolaModule {
    types: Vec<Box<dyn TypeDecl>>,
    functions: Vec<FuncDecl>,
}

// the core module
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum PrimitiveType {
    Int(u8),
    UInt(u8),
    Float(u8),
    String,
    Bool,
}

impl TypeDecl for PrimitiveType {}

enum HyperType {
    InvocationTime,
}

impl TypeDecl for HyperType {}

trait SubType {
    fn is_subtype<T: TypeDecl>(&self, rhs: &T) -> bool;
}

impl SubType for PrimitiveType {
    fn is_subtype<T: TypeDecl>(&self, rhs: &T) -> bool {
        false
    }

    fn is_subtype<T: TypeDecl>(&self, rhs: &T) -> bool {
        false
    }
}

fn x() {
    let a = PrimitiveType::UInt(8);
    let b = HyperType::InvocationTime;
    a.is_subtype(&b);
}

/*#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn unknown_types_are_reported() {}
}*/
