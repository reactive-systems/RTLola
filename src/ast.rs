//! This module contains the AST data structures for the Lola Language.

use super::parse::{Ident, Span, Symbol, SymbolTable};

/// The root Lola specification
#[derive(Debug)]
pub struct LolaSpec {
    pub language: Option<LanguageSpec>,
    pub constants: Vec<Constant>,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub trigger: Vec<Trigger>,
    pub(crate) symbols: SymbolTable,
}

impl LolaSpec {
    pub fn new() -> LolaSpec {
        LolaSpec {
            language: None,
            constants: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            trigger: Vec::new(),
            symbols: SymbolTable::new(),
        }
    }
}

/// Versions and Extensions of the Lola language
#[derive(Debug)]
pub enum LanguageSpec {
    /// The original Lola specification language,
    /// see ``LOLA: Runtime Monitoring of Synchronous Systems'' (https://ieeexplore.ieee.org/document/1443364)
    Classic,

    /// Extension of Lola to parameterized streams,
    /// see ``A Stream-based Specification Language for Network Monitoring'' (https://link.springer.com/chapter/10.1007%2F978-3-319-46982-9_10)
    Lola2,

    /// Extension of Lola to real-time specifications,
    /// see ``Real-time Stream-based Monitoring'' (https://arxiv.org/abs/1711.03829)
    RTLola,
}

impl<'a> From<&'a str> for LanguageSpec {
    fn from(string: &str) -> Self {
        unimplemented!();
    }
}

/// A declaration of a constant (stream)
#[derive(Debug)]
pub struct Constant {
    pub name: Ident,
    pub ty: Type,
    pub literal: Literal,
    pub span: Span,
}

/// A declaration of an input stream
#[derive(Debug)]
pub struct Input {
    pub name: Ident,
    pub ty: Type,
    pub span: Span,
}

/// A declaration of an output stream
#[derive(Debug)]
pub struct Output {
    pub name: Ident,
    pub ty: Type,
    pub expression: Expression,
    pub span: Span,
}

/// A declaration of a trigger
#[derive(Debug)]
pub struct Trigger {
    pub name: Option<Ident>,
    pub expression: Expression,
    pub message: Option<Symbol>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Type {
    kind: TypeKind,
    span: Span,
}

impl Type {
    pub fn new_simple(name: Symbol, span: Span) -> Type {
        Type {
            kind: TypeKind::Simple(name),
            span,
        }
    }

    pub fn new_tuple(tuple: Vec<Box<Type>>, span: Span) -> Type {
        Type {
            kind: TypeKind::Tuple(tuple),
            span,
        }
    }
}

#[derive(Debug)]
pub enum TypeKind {
    /// A tuple type, e.g., (Int, Float)
    Tuple(Vec<Box<Type>>),
    /// A simple type, e.g., Int
    Simple(Symbol),
}

/// An expression
///
/// inspired from https://doc.rust-lang.org/nightly/nightly-rustc/src/syntax/ast.rs.html
#[derive(Debug)]
pub struct Expression {
    kind: ExpressionKind,
    span: Span,
}

impl Expression {
    pub fn new(kind: ExpressionKind, span: Span) -> Expression {
        Expression { kind, span }
    }
}

#[derive(Debug)]
pub enum ExpressionKind {
    /// A literal, e.g., `1`, `"foo"`
    Lit(Literal),
    /// An identifier, e.g., `foo`
    Ident(Ident),
    /// A default expression, e.g., ` a ? 0 `
    Default(Box<Expression>, Box<Literal>),
    /// A stream lookup with offset
    Lookup(Literal, Offset),
    /// A tuple (`(a, b, c ,d)`)
    Tup(Vec<Box<Expression>>),
    /// A binary operation (For example: `a + b`, `a * b`)
    Binary(BinOp, Box<Expression>, Box<Expression>),
    /// A unary operation (For example: `!x`, `*x`)
    Unary(UnOp, Box<Expression>),
    /// A function call
    Call(Literal, Vec<Box<Expression>>),
    /// An if-then-else expression
    Ite(Box<Expression>, Box<Expression>, Box<Expression>),
}

#[derive(Debug)]
pub struct Literal {
    kind: LitKind,
    span: Span,
}

impl Literal {
    pub fn new_bool(val: bool, span: Span) -> Literal {
        Literal {
            kind: LitKind::Bool(val),
            span,
        }
    }

    pub fn new_int(val: i128, span: Span) -> Literal {
        Literal {
            kind: LitKind::Int(val),
            span,
        }
    }
}

#[derive(Debug)]
pub enum LitKind {
    /// A string literal (`"foo"`)
    Str(Symbol),
    /// An integer literal (`1`)
    Int(i128),
    /// A float literal (`1f64` or `1E10f64`)
    Float(Symbol),
    /// A boolean literal
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    /// The `+` operator (addition)
    Add,
    /// The `-` operator (subtraction)
    Sub,
    /// The `*` operator (multiplication)
    Mul,
    /// The `/` operator (division)
    Div,
    /// The `%` operator (modulus)
    Rem,
    /// The `**` operator (power)
    Pow,
    /// The `&&` operator (logical and)
    And,
    /// The `||` operator (logical or)
    Or,
    /*
    /// The `^` operator (bitwise xor)
    BitXor,
    /// The `&` operator (bitwise and)
    BitAnd,
    /// The `|` operator (bitwise or)
    BitOr,
    /// The `<<` operator (shift left)
    Shl,
    /// The `>>` operator (shift right)
    Shr,
    */
    /// The `==` operator (equality)
    Eq,
    /// The `<` operator (less than)
    Lt,
    /// The `<=` operator (less than or equal to)
    Le,
    /// The `!=` operator (not equal to)
    Ne,
    /// The `>=` operator (greater than or equal to)
    Ge,
    /// The `>` operator (greater than)
    Gt,
}

#[derive(Debug, Clone, Copy)]
pub enum UnOp {
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
}

/// Offset used in the lookup expression
#[derive(Debug, Clone, Copy)]
pub enum Offset {
    /// A constant offset, e.g., `0`, `-4`, or `42`
    Constant(i32),
}
