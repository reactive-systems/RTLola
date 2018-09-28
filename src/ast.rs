//! This module contains the AST data structures for the Lola Language.

use super::parse::{Symbol, SymbolTable};

/// The root Lola specification
#[derive(Debug)]
pub struct LolaSpec {
    language: Option<LanguageSpec>,
    constants: Vec<ConstantDecl>,
    inputs: Vec<InputDecl>,
    outputs: Vec<OutputDecl>,
    trigger: Vec<TriggerDecl>,
    symbols: SymbolTable,
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

/// A declaration of a constant (stream)
#[derive(Debug)]
pub struct ConstantDecl {
    symbol: Symbol,
    ty: Type,
    expression: Expression,
}

/// A declaration of an input stream
#[derive(Debug)]
pub struct InputDecl {
    symbol: Symbol,
    ty: Type,
    expression: Expression,
}

/// A declaration of an output stream
#[derive(Debug)]
pub struct OutputDecl {
    symbol: Symbol,
    ty: Type,
    expression: Expression,
}

/// A declaration of a trigger
#[derive(Debug)]
pub struct TriggerDecl {
    symbol: Option<Symbol>,
    expression: Expression,
    message: Option<Symbol>,
}

#[derive(Debug)]
pub struct Type(Symbol);

/// An expression
///
/// inspired from https://doc.rust-lang.org/nightly/nightly-rustc/src/syntax/ast.rs.html
#[derive(Debug)]
pub struct Expression {
    kind: ExpressionKind,
}

#[derive(Debug)]
pub enum ExpressionKind {
    /// A literal (For example: `1`, `"foo"`)
    Lit(Literal),
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
pub struct Literal(Symbol);

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
