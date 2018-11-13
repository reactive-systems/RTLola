//! This module contains the AST data structures for the Lola Language.

use super::parse::Ident;
use ast_node::NodeId;
use ast_node::Span;

/// The root Lola specification
#[derive(Debug, Default)]
pub struct LolaSpec {
    pub language: Option<LanguageSpec>,
    pub constants: Vec<Constant>,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub trigger: Vec<Trigger>,
    pub type_declarations: Vec<TypeDeclaration>,
}

impl LolaSpec {
    pub fn new() -> LolaSpec {
        LolaSpec {
            language: None,
            constants: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            trigger: Vec::new(),
            type_declarations: Vec::new(),
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
    fn from(_string: &str) -> Self {
        unimplemented!();
    }
}

/// A declaration of a constant (stream)
#[derive(AstNode, Debug)]
pub struct Constant {
    pub name: Ident,
    pub ty: Option<Type>,
    pub literal: Literal,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

/// A declaration of an input stream
#[derive(AstNode, Debug)]
pub struct Input {
    pub name: Ident,
    pub ty: Type,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

/// A declaration of an output stream
#[derive(AstNode, Debug)]
pub struct Output {
    pub name: Ident,
    pub ty: Option<Type>,
    pub params: Vec<Parameter>,
    pub template_spec: Option<TemplateSpec>,
    pub expression: Expression,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug)]
pub struct Parameter {
    pub name: Ident,
    pub ty: Option<Type>,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug)]
pub struct TemplateSpec {
    pub inv: Option<InvokeSpec>,
    pub ext: Option<ExtendSpec>,
    pub ter: Option<TerminateSpec>,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug)]
pub struct InvokeSpec {
    pub target: Expression,
    pub condition: Option<Expression>,
    pub is_if: bool,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug)]
pub struct ExtendSpec {
    pub target: Option<Expression>,
    pub freq: Option<ExtendRate>,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(Debug)]
pub enum ExtendRate {
    Frequency(Box<Expression>, FreqUnit),
    Duration(Box<Expression>, TimeUnit),
}

#[derive(AstNode, Debug)]
pub struct TerminateSpec {
    pub target: Expression,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

/// A declaration of a trigger
#[derive(AstNode, Debug)]
pub struct Trigger {
    pub name: Option<Ident>,
    pub expression: Expression,
    pub message: Option<String>,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug)]
pub struct TypeDeclaration {
    pub name: Option<Ident>,
    pub fields: Vec<Box<TypeDeclField>>,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug)]
pub struct TypeDeclField {
    pub ty: Type,
    pub name: String,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug)]
pub struct StreamInstance {
    pub stream_identifier: Ident,
    pub arguments: Vec<Box<Expression>>,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug)]
pub struct Type {
    pub kind: TypeKind,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

impl Type {
    pub fn new_simple(name: String, span: Span) -> Type {
        Type {
            _id: NodeId::DUMMY,
            kind: TypeKind::Simple(name),
            _span: span,
        }
    }

    pub fn new_tuple(tuple: Vec<Box<Type>>, span: Span) -> Type {
        Type {
            _id: NodeId::DUMMY,
            kind: TypeKind::Tuple(tuple),
            _span: span,
        }
    }
}

#[derive(AstNode, Debug)]
pub struct Parenthesis {
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

impl Parenthesis {
    pub fn new(span: Span) -> Parenthesis {
        Parenthesis {
            _id: NodeId::DUMMY,
            _span: span,
        }
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
#[derive(AstNode, Debug)]
pub struct Expression {
    pub(crate) kind: ExpressionKind,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

impl Expression {
    pub fn new(kind: ExpressionKind, span: Span) -> Expression {
        Expression {
            _id: NodeId::DUMMY,
            kind,
            _span: span,
        }
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

#[derive(AstNode, Debug, Clone)]
pub struct Literal {
    pub kind: LitKind,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

impl Literal {
    pub fn new_bool(val: bool, span: Span) -> Literal {
        Literal {
            _id: NodeId::DUMMY,
            kind: LitKind::Bool(val),
            _span: span,
        }
    }

    pub fn new_int(val: i128, span: Span) -> Literal {
        Literal {
            _id: NodeId::DUMMY,
            kind: LitKind::Int(val),
            _span: span,
        }
    }

    pub fn new_tuple(vals: &[Literal], span: Span) -> Literal {
        Literal {
            _id: NodeId::DUMMY,
            kind: LitKind::Tuple(vals.to_vec()),
            _span: span,
        }
    }

    pub fn new_float(val: f64, span: Span) -> Literal {
        Literal {
            _id: NodeId::DUMMY,
            kind: LitKind::Float(val),
            _span: span,
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
    Tuple(Vec<Literal>),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Supported frequencies for sliding windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreqUnit {
    MicroHertz, // ~11 Days
    MilliHertz, // ~16 Minutes
    Hertz,
    KiloHertz,
    MegaHertz,
    GigaHertz,
}
