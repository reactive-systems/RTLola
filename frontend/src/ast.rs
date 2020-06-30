/*! This module contains the AST data structures for the RTLola Language.

Every node in the abstract syntax tree is assigned a unique id and stores the matching span in the specification.
*/

pub(crate) mod conversion;
pub(crate) mod print;
pub(crate) mod verify;

use super::parse::Ident;
use crate::parse::NodeId;
use crate::parse::Span;
use num::rational::Rational64 as Rational;
use std::rc::Rc;
use std::time::Duration;

/// The root of a RTLola specification
#[derive(Debug, Default, Clone)]
pub struct RTLolaAst {
    pub imports: Vec<Import>,
    pub constants: Vec<Rc<Constant>>,
    pub inputs: Vec<Rc<Input>>,
    pub outputs: Vec<Rc<Output>>,
    pub trigger: Vec<Rc<Trigger>>,
    pub type_declarations: Vec<TypeDeclaration>,
}

impl RTLolaAst {
    pub(crate) fn new() -> RTLolaAst {
        RTLolaAst {
            imports: Vec::new(),
            constants: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            trigger: Vec::new(),
            type_declarations: Vec::new(),
        }
    }
}

/**
An AST node representing the import of a module.
*/
#[derive(Debug, Clone)]
pub struct Import {
    pub name: Ident,
    pub id: NodeId,
    pub span: Span,
}

/**
An AST node representing the declaration of a constant.
*/
#[derive(Debug, Clone)]
pub struct Constant {
    pub name: Ident,
    pub ty: Option<Type>,
    pub literal: Literal,
    pub id: NodeId,
    pub span: Span,
}

/**
An AST node representing the declaration of an input stream.
*/
#[derive(Debug, Clone)]
pub struct Input {
    pub name: Ident,
    pub ty: Type,
    pub params: Vec<Rc<Parameter>>,
    pub id: NodeId,
    pub span: Span,
}

/**
An AST node representing the declaration of an output stream.
*/
#[derive(Debug, Clone)]
pub struct Output {
    pub name: Ident,
    pub ty: Type,
    pub extend: ActivationCondition,
    pub params: Vec<Rc<Parameter>>,
    pub template_spec: Option<TemplateSpec>,
    pub termination: Option<Expression>,
    pub expression: Expression,
    pub id: NodeId,
    pub span: Span,
}

/**
An AST node representing the declaration of a parameter of a stream.
*/
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: Ident,
    pub ty: Type,
    pub id: NodeId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ActivationCondition {
    pub expr: Option<Expression>,
    pub id: NodeId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct TemplateSpec {
    pub inv: Option<InvokeSpec>,
    pub ext: Option<ExtendSpec>,
    pub ter: Option<TerminateSpec>,
    pub id: NodeId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct InvokeSpec {
    pub target: Expression,
    pub condition: Option<Expression>,
    pub is_if: bool,
    pub id: NodeId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ExtendSpec {
    pub target: Expression,
    pub id: NodeId,
    pub span: Span,
}

pub type Signum = i8;

#[derive(Debug, Clone, Copy)]
pub struct TimeSpec {
    pub signum: Signum,
    pub period: Duration,
    pub exact_period: Rational,
    pub id: NodeId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct TerminateSpec {
    pub target: Expression,
    pub id: NodeId,
    pub span: Span,
}

/**
An AST node representing the declaration of a trigger.
*/
#[derive(Debug, Clone)]
pub struct Trigger {
    pub name: Option<Ident>,
    pub expression: Expression,
    pub message: Option<String>,
    pub id: NodeId,
    pub span: Span,
}

/**
An AST node representing the declaration of a user-defined type.
*/
#[allow(clippy::vec_box)]
#[derive(Debug, Clone)]
pub struct TypeDeclaration {
    pub name: Option<Ident>,
    pub fields: Vec<Box<TypeDeclField>>,
    pub id: NodeId,
    pub span: Span,
}

/**
An AST node representing the declaration of a field of a user-defined type.
*/
#[derive(Debug, Clone)]
pub struct TypeDeclField {
    pub ty: Type,
    pub name: String,
    pub id: NodeId,
    pub span: Span,
}

/**
An AST node representing an opening or closing parenthesis.
*/
#[derive(Debug, Clone, Copy)]
pub struct Parenthesis {
    pub id: NodeId,
    pub span: Span,
}

impl Parenthesis {
    pub(crate) fn new(id: NodeId, span: Span) -> Parenthesis {
        Parenthesis { id, span }
    }
}

#[derive(Debug, Clone)]
pub struct Type {
    pub kind: TypeKind,
    pub id: NodeId,
    pub span: Span,
}

impl Type {
    pub(crate) fn new_simple(id: NodeId, name: String, span: Span) -> Type {
        Type { id, kind: TypeKind::Simple(name), span }
    }

    pub(crate) fn new_tuple(id: NodeId, tuple: Vec<Type>, span: Span) -> Type {
        Type { id, kind: TypeKind::Tuple(tuple), span }
    }

    pub(crate) fn new_optional(id: NodeId, name: Type, span: Span) -> Type {
        Type { id, kind: TypeKind::Optional(name.into()), span }
    }

    pub(crate) fn new_inferred(id: NodeId) -> Type {
        Type { id, kind: TypeKind::Inferred, span: Span::unknown() }
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

/**
An expression

inspired by <https://doc.rust-lang.org/nightly/nightly-rustc/src/syntax/ast.rs.html>
*/
#[derive(Debug, Clone)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub id: NodeId,
    pub span: Span,
}

impl Expression {
    pub(crate) fn new(id: NodeId, kind: ExpressionKind, span: Span) -> Expression {
        Expression { id, kind, span }
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
    //TODO remove
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
    Conjunction,
    Disjunction,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    pub id: NodeId,
    pub span: Span,
}

impl Literal {
    pub(crate) fn new_bool(id: NodeId, val: bool, span: Span) -> Literal {
        Literal { id, kind: LitKind::Bool(val), span }
    }

    pub(crate) fn new_numeric(id: NodeId, val: &str, unit: Option<String>, span: Span) -> Literal {
        Literal { id, kind: LitKind::Numeric(val.to_string(), unit), span }
    }

    pub(crate) fn new_str(id: NodeId, val: &str, span: Span) -> Literal {
        Literal { id, kind: LitKind::Str(val.to_string()), span }
    }

    pub(crate) fn new_raw_str(id: NodeId, val: &str, span: Span) -> Literal {
        Literal { id, kind: LitKind::RawStr(val.to_string()), span }
    }
}

#[derive(Debug, Clone)]
pub enum LitKind {
    /// A string literal (`"foo"`)
    Str(String),
    /// A raw string literal (`r#" x " a \ff "#`)
    RawStr(String),
    /// A numeric value with optional postfix part (`42`, `1.3`, `1Hz`, `100sec`)
    /// Stores as a string to have lossless representation
    Numeric(String, Option<String>),
    /// A boolean literal (`true`)
    Bool(bool),
}

/**
An AST node representing a binary operator.
*/
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

/**
An AST node representing an unary operator.
*/
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum UnOp {
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
    /// The `~` operator for one's complement
    BitNot,
}

/**
An AST node representing the name of a called function and also the names of the arguments.
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionName {
    /**
    The name of the called function.
    */
    pub name: Ident,
    /**
    A list containing an element for each argument, containing the name if it is a named argument or else `None`.
    */
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
