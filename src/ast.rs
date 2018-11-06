//! This module contains the AST data structures for the Lola Language.

use super::parse::{Ident, Span};

/// The root Lola specification
#[derive(Debug)]
pub struct LolaSpec {
    pub language: Option<LanguageSpec>,
    pub constants: Vec<Constant>,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub trigger: Vec<Trigger>,
}

impl LolaSpec {
    pub fn new() -> LolaSpec {
        LolaSpec {
            language: None,
            constants: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            trigger: Vec::new(),
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

/// Every node in the AST gets a unique id, represented by a 32bit unsiged integer.
/// They are used in the later analysis phases to store information about AST nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u32);

impl NodeId {
    pub fn new(x: usize) -> NodeId {
        assert!(x < (std::u32::MAX as usize));
        NodeId(x as u32)
    }

    pub fn from_u32(x: u32) -> NodeId {
        NodeId(x)
    }

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }

    /// When parsing, we initially give all AST nodes this AST node id.
    /// Then later, in the renumber pass, we renumber them to have small, positive ids.
    pub const DUMMY: NodeId = NodeId(!0);
}

/// A declaration of a constant (stream)
#[derive(Debug)]
pub struct Constant {
    pub(crate) id: NodeId,
    pub name: Ident,
    pub ty: Option<Type>,
    pub literal: Literal,
    pub span: Span,
}

/// A declaration of an input stream
#[derive(Debug)]
pub struct Input {
    pub(crate) id: NodeId,
    pub name: Ident,
    pub ty: Type,
    pub span: Span,
}

/// A declaration of an output stream
#[derive(Debug)]
pub struct Output {
    pub(crate) id: NodeId,
    pub name: Ident,
    pub ty: Option<Type>,
    pub expression: Expression,
    pub span: Span,
}

/// A declaration of a trigger
#[derive(Debug)]
pub struct Trigger {
    pub(crate) id: NodeId,
    pub name: Option<Ident>,
    pub expression: Expression,
    pub message: Option<String>,
    pub span: Span,
}

#[derive(Debug)]
pub struct TypeDeclaration {
    pub name: Option<Ident>,
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Debug)]
pub struct StreamInstance {
    pub stream_identifier: Ident,
    pub arguments: Vec<Box<Expression>>,
}

#[derive(Debug)]
pub struct Type {
    pub kind: TypeKind,
    pub span: Span,
}

impl Type {
    pub fn new_simple(name: String, span: Span) -> Type {
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
pub struct Parenthesis {
    pub span: Span,
}

impl Parenthesis {
    pub fn new(span: Span) -> Parenthesis {
        Parenthesis { span }
    }
}

#[derive(Debug)]
pub enum TypeKind {
    /// A tuple type, e.g., (Int, Float)
    Tuple(Vec<Box<Type>>),
    /// A simple type, e.g., Int
    Simple(String),
    /// Malformed type, e.g, `mis$ing`
    Malformed(String),
}

/// An expression
///
/// inspired by https://doc.rust-lang.org/nightly/nightly-rustc/src/syntax/ast.rs.html
#[derive(Debug)]
pub struct Expression {
    pub(crate) id: NodeId,
    pub(crate) kind: ExpressionKind,
    span: Span,
}

impl Expression {
    pub fn new(kind: ExpressionKind, span: Span) -> Expression {
        Expression { id: NodeId::DUMMY, kind, span }
    }
}

#[derive(Debug)]
pub enum ExpressionKind {
    /// A literal, e.g., `1`, `"foo"`
    Lit(Literal),
    /// An identifier, e.g., `foo`
    Ident(Ident),
    /// A default expression, e.g., ` a ? 0 `
    Default(Box<Expression>, Box<Expression>),
    /// A stream lookup with offset
    Lookup(StreamInstance, Offset, Option<WindowOperation>),
    /// A binary operation (For example: `a + b`, `a * b`)
    Binary(BinOp, Box<Expression>, Box<Expression>),
    /// A unary operation (For example: `!x`, `*x`)
    Unary(UnOp, Box<Expression>),
    /// An if-then-else expression
    Ite(Box<Expression>, Box<Expression>, Box<Expression>),
    /// An expression enveloped in parentheses
    ParenthesizedExpression(
        Option<Box<Parenthesis>>,
        Box<Expression>,
        Option<Box<Parenthesis>>,
    ),
    /// An expression was expected, e.g., after an operator like `*`
    MissingExpression(),
    /// A tuple expression
    Tuple(Vec<Box<Expression>>),
    /// A function call
    Function(FunctionKind, Vec<Box<Expression>>),
}

#[derive(Debug, PartialEq, Eq)]
pub enum FunctionKind {
    NthRoot,
    Sqrt,
    Projection,
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
    Exp,
    Floor,
    Ceil,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum WindowOperation {
    Sum,
    Product,
    Average,
    Count,
    Integral,
}

#[derive(Debug, Clone)]
pub struct Literal {
    pub kind: LitKind,
    pub span: Span,
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

    pub fn new_tuple(vals: &Vec<Literal>, span: Span) -> Literal {
        Literal {
            kind: LitKind::Tuple(Box::new(vals.to_vec())),
            span,
        }
    }

    pub fn new_float(val: f64, span: Span) -> Literal {
        Literal {
            kind: LitKind::Float(val),
            span,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LitKind {
    /// A string literal (`"foo"`)
    Str(String),
    /// An integer literal (`1`)
    Int(i128),
    /// A float literal (`1f64` or `1E10f64`)
    Float(f64),
    /// A boolean literal (`true`)
    Bool(bool),
    // A tuple literal (`(1, 2f64, "34as", 99)`)
    Tuple(Box<Vec<Literal>>),
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
#[derive(Debug)]
pub enum Offset {
    /// A discrete offset, e.g., `0`, `-4`, or `42`
    DiscreteOffset(Box<Expression>),
    /// A real-time offset, e.g., `3ms`, `4min`, `2.3h`
    RealTimeOffset(Box<Expression>, TimeUnit),
}

/// Supported time unit for real time expressions
#[derive(Debug, Clone, Copy)]
pub enum TimeUnit {
    NanoSecond,
    MicroSecond,
    MilliSecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    /// Note: A year is always, *always*, 365 days long.
    Year,
}

