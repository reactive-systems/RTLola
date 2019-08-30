//! This module contains the AST data structures for the Lola Language.

pub(crate) mod conversion;
pub(crate) mod print;
pub(crate) mod verify;

use super::parse::Ident;
use crate::parse::NodeId;
use crate::parse::Span;
use num::rational::Rational64 as Rational;
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
#[derive(Debug, Clone)]
pub struct Constant {
    pub name: Ident,
    pub ty: Option<Type>,
    pub literal: Literal,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

/// A declaration of an input stream
#[derive(Debug, Clone)]
pub struct Input {
    pub name: Ident,
    pub ty: Type,
    pub params: Vec<Parameter>,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

/// A declaration of an output stream
#[derive(Debug, Clone)]
pub struct Output {
    pub name: Ident,
    pub ty: Type,
    pub extend: ActivationCondition,
    pub params: Vec<Parameter>,
    pub template_spec: Option<TemplateSpec>,
    pub termination: Option<Expression>,
    pub expression: Expression,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: Ident,
    pub ty: Type,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

#[derive(Debug, Clone)]
pub struct ActivationCondition {
    pub expr: Option<Expression>,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

#[derive(Debug, Clone)]
pub struct TemplateSpec {
    pub inv: Option<InvokeSpec>,
    pub ext: Option<ExtendSpec>,
    pub ter: Option<TerminateSpec>,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

#[derive(Debug, Clone)]
pub struct InvokeSpec {
    pub target: Expression,
    pub condition: Option<Expression>,
    pub is_if: bool,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

#[derive(Debug, Clone)]
pub struct ExtendSpec {
    pub target: Expression,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

pub(crate) type Signum = i8;

#[derive(Debug, Clone)]
pub struct TimeSpec {
    pub signum: Signum,
    pub period: Duration,
    pub exact_period: Rational,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

#[derive(Debug, Clone)]
pub struct TerminateSpec {
    pub target: Expression,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

/// A declaration of a trigger
#[derive(Debug, Clone)]
pub struct Trigger {
    pub name: Option<Ident>,
    pub expression: Expression,
    pub message: Option<String>,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

#[allow(clippy::vec_box)]
#[derive(Debug, Clone)]
pub struct TypeDeclaration {
    pub name: Option<Ident>,
    pub fields: Vec<Box<TypeDeclField>>,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

#[derive(Debug, Clone)]
pub struct TypeDeclField {
    pub ty: Type,
    pub name: String,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

#[derive(Debug, Clone)]
pub struct Parenthesis {
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

impl Parenthesis {
    pub fn new(span: Span) -> Parenthesis {
        Parenthesis { id: NodeId::DUMMY, span }
    }
}

#[derive(Debug, Clone)]
pub struct Type {
    pub kind: TypeKind,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

impl Type {
    pub fn new_simple(name: String, span: Span) -> Type {
        Type { id: NodeId::DUMMY, kind: TypeKind::Simple(name), span }
    }

    pub fn new_tuple(tuple: Vec<Type>, span: Span) -> Type {
        Type { id: NodeId::DUMMY, kind: TypeKind::Tuple(tuple), span }
    }

    pub fn new_optional(name: Type, span: Span) -> Type {
        Type { id: NodeId::DUMMY, kind: TypeKind::Optional(name.into()), span }
    }

    pub fn new_inferred() -> Type {
        Type { id: NodeId::DUMMY, kind: TypeKind::Inferred, span: Span::unknown() }
    }
}

#[derive(Debug, Clone)]
pub enum TypeKind {
    /// A simple type, e.g., `Int`
    Simple(String),
    /// A tuple type, e.g., `(Int32, Float32)`
    Tuple(Vec<Type>),
    /// An optional type, e.g., `Int?`
    Optional(Box<Type>),
    /// Should be inferred, i.e., is not annotated
    Inferred,
}

/// An expression
///
/// inspired by <https://doc.rust-lang.org/nightly/nightly-rustc/src/syntax/ast.rs.html>
#[derive(Debug, Clone)]
pub struct Expression {
    pub(crate) kind: ExpressionKind,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

impl Expression {
    pub fn new(kind: ExpressionKind, span: Span) -> Expression {
        Expression { id: NodeId::DUMMY, kind, span }
    }
}

#[allow(clippy::large_enum_variant, clippy::vec_box)]
#[derive(Debug, Clone)]
pub enum ExpressionKind {
    /// A literal, e.g., `1`, `"foo"`
    Lit(Literal),
    /// An identifier, e.g., `foo`
    Ident(Ident),
    /// Accessing a stream
    StreamAccess(Box<Expression>, StreamAccessKind),
    /// A default expression, e.g., `a.defaults(to: 0) `
    Default(Box<Expression>, Box<Expression>),
    /// An offset expression, e.g., `a.offset(by: -1)`
    Offset(Box<Expression>, Offset),
    /// A sliding window with duration `duration` and aggregation function `aggregation`
    SlidingWindowAggregation {
        expr: Box<Expression>,
        duration: Box<Expression>,
        wait: bool,
        aggregation: WindowOperation,
    },
    /// A binary operation (For example: `a + b`, `a * b`)
    Binary(BinOp, Box<Expression>, Box<Expression>),
    /// A unary operation (For example: `!x`, `*x`)
    Unary(UnOp, Box<Expression>),
    /// An if-then-else expression
    Ite(Box<Expression>, Box<Expression>, Box<Expression>),
    /// An expression enveloped in parentheses
    ParenthesizedExpression(Option<Box<Parenthesis>>, Box<Expression>, Option<Box<Parenthesis>>),
    /// An expression was expected, e.g., after an operator like `*`
    MissingExpression,
    /// A tuple expression
    Tuple(Vec<Box<Expression>>),
    /// Access of a named (`obj.foo`) or unnamed (`obj.0`) struct field
    Field(Box<Expression>, Ident),
    /// A method call, e.g., `foo.bar(-1)`
    Method(Box<Expression>, FunctionName, Vec<Type>, Vec<Box<Expression>>),
    /// A function call
    Function(FunctionName, Vec<Type>, Vec<Box<Expression>>),
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum WindowOperation {
    Count,
    Min,
    Max,
    Sum,
    Product,
    Average,
    Integral,
}

/// Describes the operation used to access a stream
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum StreamAccessKind {
    /// Synchronous access
    Sync,
    /// Hold access for *incompatible* stream types, returns previous known value
    Hold,
    /// Optional access, returns value if it exists
    Optional,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Offset {
    Discrete(i16),
    RealTime(Rational, TimeUnit),
}

/// Supported time unit for real time expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    Nanosecond,
    Microsecond,
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    /// Note: A year is always, *always*, 365 days long.
    Year,
}

#[derive(Debug, Clone)]
pub struct Literal {
    pub kind: LitKind,
    pub(crate) id: NodeId,
    pub(crate) span: Span,
}

impl Literal {
    pub fn new_bool(val: bool, span: Span) -> Literal {
        Literal { id: NodeId::DUMMY, kind: LitKind::Bool(val), span }
    }

    pub fn new_numeric(val: &str, unit: Option<String>, span: Span) -> Literal {
        Literal { id: NodeId::DUMMY, kind: LitKind::Numeric(val.to_string(), unit), span }
    }

    pub(crate) fn new_str(val: &str, span: Span) -> Literal {
        Literal { id: NodeId::DUMMY, kind: LitKind::Str(val.to_string()), span }
    }

    pub(crate) fn new_raw_str(val: &str, span: Span) -> Literal {
        Literal { id: NodeId::DUMMY, kind: LitKind::RawStr(val.to_string()), span }
    }
}

#[derive(Debug, Clone)]
pub enum LitKind {
    /// A string literal (`"foo"`)
    Str(String),
    /// A raw string literal (`r#" x " a \ff "#`)
    RawStr(String),
    /// A numeric value with optional postfix part (`42`, `1.3`, `1Hz`, `100sec`)
    /// Strored as a string to have lossless representation
    Numeric(String, Option<String>),
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
    /// The `~` operator for one's complement
    BitNot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionName {
    pub name: Ident,
    pub arg_names: Vec<Option<Ident>>,
}

impl FunctionName {
    pub(crate) fn new(name: String, arg_names: &[Option<String>]) -> Self {
        Self {
            name: Ident::new(name, Span::unknown()),
            arg_names: arg_names.iter().map(|o| o.clone().map(|s| Ident::new(s, Span::unknown()))).collect(),
        }
    }
}
