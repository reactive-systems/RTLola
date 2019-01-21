//! This module contains the AST data structures for the Lola Language.

use super::parse::Ident;
use ast_node::NodeId;
use ast_node::Span;
use std::time::Duration;

/// The root Lola specification
#[derive(Debug, Default, Clone)]
pub struct LolaSpec {
    pub language: Option<LanguageSpec>,
    pub imports: Vec<Import>,
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
            imports: Vec::new(),
            constants: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            trigger: Vec::new(),
            type_declarations: Vec::new(),
        }
    }
}

/// Versions and Extensions of the Lola language
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
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

#[derive(Debug, Clone)]
pub struct Import {
    pub name: Ident,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

/// A declaration of a constant (stream)
#[derive(AstNode, Debug, Clone)]
pub struct Constant {
    pub name: Ident,
    pub ty: Option<Type>,
    pub literal: Literal,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

/// A declaration of an input stream
#[derive(AstNode, Debug, Clone)]
pub struct Input {
    pub name: Ident,
    pub ty: Type,
    pub params: Vec<Parameter>,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

/// A declaration of an output stream
#[derive(AstNode, Debug, Clone)]
pub struct Output {
    pub name: Ident,
    pub ty: Type,
    pub params: Vec<Parameter>,
    pub template_spec: Option<TemplateSpec>,
    pub expression: Expression,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug, Clone)]
pub struct Parameter {
    pub name: Ident,
    pub ty: Type,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug, Clone)]
pub struct TemplateSpec {
    pub inv: Option<InvokeSpec>,
    pub ext: Option<ExtendSpec>,
    pub ter: Option<TerminateSpec>,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug, Clone)]
pub struct InvokeSpec {
    pub target: Expression,
    pub condition: Option<Expression>,
    pub is_if: bool,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug, Clone)]
pub struct ExtendSpec {
    pub target: Option<Expression>,
    pub freq: Option<ExtendRate>,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(Debug, Clone)]
pub enum ExtendRate {
    Frequency(Box<Expression>, FreqUnit),
    Duration(Box<Expression>, TimeUnit),
}

impl Into<Duration> for &ExtendRate {
    fn into(self) -> Duration {
        let (expr, factor) = match &self {
            ExtendRate::Duration(expr, unit) => {
                (
                    expr,
                    match unit {
                        TimeUnit::NanoSecond => 1u64,
                        TimeUnit::MicroSecond => 10u64.pow(3),
                        TimeUnit::MilliSecond => 10u64.pow(6),
                        TimeUnit::Second => 10u64.pow(9),
                        TimeUnit::Minute => 10u64.pow(9) * 60,
                        TimeUnit::Hour => 10u64.pow(9) * 60 * 60,
                        TimeUnit::Day => 10u64.pow(9) * 60 * 60 * 24,
                        TimeUnit::Week => 10u64.pow(9) * 60 * 24 * 24 * 7,
                        TimeUnit::Year => 10u64.pow(9) * 60 * 24 * 24 * 7 * 365, // fits in u57
                    },
                )
            }
            ExtendRate::Frequency(expr, unit) => {
                (
                    expr,
                    match unit {
                        FreqUnit::MicroHertz => 10u64.pow(15), // fits in u50,
                        FreqUnit::MilliHertz => 10u64.pow(12),
                        FreqUnit::Hertz => 10u64.pow(9),
                        FreqUnit::KiloHertz => 10u64.pow(6),
                        FreqUnit::MegaHertz => 10u64.pow(3),
                        FreqUnit::GigaHertz => 1u64,
                    },
                )
            }
        };
        match &expr.kind {
            ExpressionKind::Lit(l) => {
                match l.kind {
                    LitKind::Int(i) => {
                        // TODO: Improve: Robust against overflows.
                        let value = i as u128 * u128::from(factor); // Multiplication might fail.
                        let secs = (value / 10u128.pow(9)) as u64; // Cast might fail.
                        let nanos = (value % 10u128.pow(9)) as u32; // Perfectly safe cast to u32.
                        Duration::new(secs, nanos)
                    }
                    LitKind::Float(f) => {
                        // TODO: Improve: Robust against overflows and inaccuracies.
                        let value = f * factor as f64;
                        let secs = (value / 1_000_000_000f64) as u64;
                        let nanos = (value % 1_000_000_000f64) as u32;
                        Duration::new(secs, nanos)
                    }
                    _ => panic!(),
                }
            }

            _ => panic!(),
        }
    }
}

#[derive(AstNode, Debug, Clone)]
pub struct TerminateSpec {
    pub target: Expression,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

/// A declaration of a trigger
#[derive(AstNode, Debug, Clone)]
pub struct Trigger {
    pub name: Option<Ident>,
    pub expression: Expression,
    pub message: Option<String>,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug, Clone)]
pub struct TypeDeclaration {
    pub name: Option<Ident>,
    pub fields: Vec<Box<TypeDeclField>>,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug, Clone)]
pub struct TypeDeclField {
    pub ty: Type,
    pub name: String,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug, Clone)]
pub struct StreamInstance {
    pub stream_identifier: Ident,
    pub arguments: Vec<Box<Expression>>,
    pub(crate) _id: NodeId,
    pub(crate) _span: Span,
}

#[derive(AstNode, Debug, Clone)]
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

#[derive(AstNode, Debug, Clone)]
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

    pub fn new_tuple(tuple: Vec<Type>, span: Span) -> Type {
        Type {
            _id: NodeId::DUMMY,
            kind: TypeKind::Tuple(tuple),
            _span: span,
        }
    }

    pub fn new_inferred() -> Type {
        Type {
            _id: NodeId::DUMMY,
            kind: TypeKind::Inferred,
            _span: Span::unknown(),
        }
    }

    pub fn new_duration(val: u32, unit: TimeUnit, span: Span) -> Type {
        Type {
            _id: NodeId::DUMMY,
            kind: TypeKind::Duration(val, unit),
            _span: span,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TypeKind {
    /// A simple type, e.g., Int
    Simple(String),
    /// A tuple type, e.g., (Int32, Float32)
    Tuple(Vec<Type>),
    /// A duration, e.g., `22s`
    Duration(u32, TimeUnit),
    /// Should be inferred, i.e., is not annotated
    Inferred,
    /// Malformed type, e.g, `mis$ing`
    Malformed(String),
}

/// An expression
///
/// inspired by https://doc.rust-lang.org/nightly/nightly-rustc/src/syntax/ast.rs.html
#[derive(AstNode, Debug, Clone)]
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

#[derive(Debug, Clone)]
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
    /// Access of a named (`obj.foo`) or unnamed (`obj.0`) struct field
    Field(Box<Expression>, Ident),
    /// A method call, e.g., `foo.offset(-1)`
    Method(Box<Expression>, Ident, Vec<Type>, Vec<Box<Expression>>),
    /// A function call
    Function(Ident, Vec<Type>, Vec<Box<Expression>>),
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

    pub fn new_float(val: f64, span: Span) -> Literal {
        Literal {
            _id: NodeId::DUMMY,
            kind: LitKind::Float(val),
            _span: span,
        }
    }

    pub(crate) fn new_str(val: &str, span: Span) -> Literal {
        Literal {
            _id: NodeId::DUMMY,
            kind: LitKind::Str(val.to_string()),
            _span: span,
        }
    }

    pub(crate) fn new_raw_str(val: &str, span: Span) -> Literal {
        Literal {
            _id: NodeId::DUMMY,
            kind: LitKind::RawStr(val.to_string()),
            _span: span,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LitKind {
    /// A string literal (`"foo"`)
    Str(String),
    /// A raw string literal (`r#" x " a \ff "#`)
    RawStr(String),
    /// An integer literal (`1`)
    Int(i128),
    /// A float literal (`1f64` or `1E10f64`)
    Float(f64),
    /// A boolean literal (`true`)
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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum UnOp {
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
}

/// Offset used in the lookup expression
#[derive(Debug, Clone)]
pub enum Offset {
    /// A discrete offset, e.g., `0`, `-4`, or `42`
    DiscreteOffset(Box<Expression>),
    /// A real-time offset, e.g., `3ms`, `4min`, `2.3h`
    RealTimeOffset(Box<Expression>, TimeUnit),
}

/// Supported time unit for real time expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
