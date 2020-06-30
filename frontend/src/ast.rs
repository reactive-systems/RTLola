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

/// The root of a RTLola specification, consisting of stream and trigger declarations.
/// Each declaration contains the id of the AST node, the span in the input specification file, and declaration specific components.
#[derive(Debug, Default, Clone)]
pub struct RTLolaAst {
    /// The imports of additional modules
    pub imports: Vec<Import>,
    /// The constant stream declarations
    pub constants: Vec<Rc<Constant>>,
    /// The input stream declarations
    pub inputs: Vec<Rc<Input>>,
    /// The output stream declarations
    pub outputs: Vec<Rc<Output>>,
    /// The trigger declarations
    pub trigger: Vec<Rc<Trigger>>,
    /// The user-defined type declarations
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
An AST node representing the import of a module, which brings additional implemented functionality to a specification.
The 'math' module, for example, adds pre-defined mathematical functions as the sine or cosine function.
*/
#[derive(Debug, Clone)]
pub struct Import {
    /// The name of the module
    pub name: Ident,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the module
    pub span: Span,
}

/**
An AST node representing the declaration of a constant.
*/
#[derive(Debug, Clone)]
pub struct Constant {
    /// The name of the constant stream
    pub name: Ident,
    /// The value type of the constant stream
    pub ty: Option<Type>,
    /// The literal defining the constant
    pub literal: Literal,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the constant stream
    pub span: Span,
}

/**
An AST node representing the declaration of an input stream.
*/
#[derive(Debug, Clone)]
pub struct Input {
    /// The name of the input stream
    pub name: Ident,
    ///  The value type of the input stream
    pub ty: Type,
    /// The parameters of a parameterized input stream; The vector is empty in non-parametrized streams.
    pub params: Vec<Rc<Parameter>>,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the input stream
    pub span: Span,
}

/**
An AST node representing the declaration of an output stream.
*/
#[derive(Debug, Clone)]
pub struct Output {
    /// The name of the output stream
    pub name: Ident,
    /// The value type of the output stream
    pub ty: Type,
    /// The activation condition, which defines when a new value of a stream is computed. In periodic streams, the condition is 'None'
    pub extend: ActivationCondition,
    /// The parameters of a parameterized output stream; The vector is empty in non-parametrized streams
    pub params: Vec<Rc<Parameter>>,
    /// The declaration of the stream template for parametrized streams, e.g., the invoke declaration.
    pub template_spec: Option<TemplateSpec>,
    ///  The termination condition of parametrized streams
    pub termination: Option<Expression>,
    /// The stream expression of a output stream, e.g., a + b.offset(by: -1).defaults(to: 0)
    pub expression: Expression,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the output stream
    pub span: Span,
}

/**
An AST node representing the declaration of a parameter of a parametrized stream.
*/
#[derive(Debug, Clone)]
pub struct Parameter {
    /// The name of the parameter
    pub name: Ident,
    /// The value type of the parameter
    pub ty: Type,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the parameter
    pub span: Span,
}

/**
An AST node representing the declaration of a activation condition of a stream.
*/
#[derive(Debug, Clone)]
pub struct ActivationCondition {
    /// The boolean expression representing the activation condition. For periodic streams this component is assigned to 'None'
    pub expr: Option<Expression>,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the activation condition
    pub span: Span,
}

/**
An AST node representing the declaration of a template of a parametrized stream.
*/
#[derive(Debug, Clone)]
pub struct TemplateSpec {
    /// The invoke condition of the parametrized stream.
    pub inv: Option<InvokeSpec>,
    /// The extend condition of the parametrized stream.
    pub ext: Option<ExtendSpec>,
    /// The termination condition of the parametrized stream.
    pub ter: Option<TerminateSpec>,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the template
    pub span: Span,
}

/**
An AST node representing the declaration of an invoke condition of a parametrized stream
*/
#[derive(Debug, Clone)]
pub struct InvokeSpec {
    /// The expression defining the parameter instances. If the stream has more than one parameter, the expression needs to return a tuple, with one element for each parameter
    pub target: Expression,
    /// An additional condition for the creation of an instance, i.e., an instance is only created if the condition is true If 'is_true' is false, this component is assigned to 'None'
    pub condition: Option<Expression>,
    /// A flag to describe if the invoke declaration contains an additional condition
    pub is_if: bool,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the invoke declaration
    pub span: Span,
}

/**
An AST node representing the declaration of an extend condition of a parametrized stream
*/
#[derive(Debug, Clone)]
pub struct ExtendSpec {
    /// The boolean expression defining the condition, if a parameterized instance is evaluated.
    pub target: Expression,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
    pub span: Span,
}

/**
An AST node representing the declaration of a termination condition of a parametrized stream
*/
#[derive(Debug, Clone)]
pub struct TerminateSpec {
    /// The boolean expression defining the condition, if a parameterized instance is terminated.
    pub target: Expression,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
    pub span: Span,
}

/**
An AST node representing the declaration of a trigger
*/
#[derive(Debug, Clone)]
pub struct Trigger {
    /// The optional name of a trigger
    pub name: Option<Ident>,
    /// The boolean expression of a trigger
    pub expression: Expression,
    /// The optional trigger message, which is printed if the monitor raises the trigger
    pub message: Option<String>,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
    pub span: Span,
}

/**
An AST node representing the declaration of a user-defined type.
*/
#[allow(clippy::vec_box)]
#[derive(Debug, Clone)]
pub struct TypeDeclaration {
    /// The name of the new type.
    pub name: Option<Ident>,
    /// The components of the new type, e.g. a GPS type might consist of a type for the latitude and for the longitude
    pub fields: Vec<Box<TypeDeclField>>,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the type declaration
    pub span: Span,
}

/**
An AST node representing the declaration of a field of a user-defined type.
*/
#[derive(Debug, Clone)]
pub struct TypeDeclField {
    /// The type of a field of a user-defined type
    pub ty: Type,
    /// The name of a field of a user-defined type
    pub name: String,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the type declaration
    pub span: Span,
}

/**
An AST node representing an opening or closing parenthesis.
*/
#[derive(Debug, Clone, Copy)]
pub struct Parenthesis {
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
    pub span: Span,
}

impl Parenthesis {
    pub(crate) fn new(id: NodeId, span: Span) -> Parenthesis {
        Parenthesis { id, span }
    }
}

/**
An AST node representing the declaration of a value type

*/
#[derive(Debug, Clone)]
pub struct Type {
    /// The kind of the Type, e.g., a tuple
    pub kind: TypeKind,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
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

/// AST representation of the value type of a stream
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
The AST representation of a stream expression
inspired by <https://doc.rust-lang.org/nightly/nightly-rustc/src/syntax/ast.rs.html>
*/
#[derive(Debug, Clone)]
pub struct Expression {
    /// The kind of the root expression, e.g., stream access
    pub kind: ExpressionKind,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
    pub span: Span,
}

impl Expression {
    pub(crate) fn new(id: NodeId, kind: ExpressionKind, span: Span) -> Expression {
        Expression { id, kind, span }
    }
}

#[allow(clippy::large_enum_variant, clippy::vec_box)]
#[derive(Debug, Clone)]
/// The AST representation of a single expression
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
        /// The accesses stream
        expr: Box<Expression>,
        /// The duration of the window
        duration: Box<Expression>,
        /// Flag to mark that the window returns only a value if the complete duration has passed
        wait: bool,
        /// The aggregation function
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
/// The AST representation of the different aggregation functions
pub enum WindowOperation {
    /// Aggregation function to count the number of updated values on the accessed stream
    Count,
    /// Aggregation function to return the minimum
    Min,
    /// Aggregation function to return the minimum
    Max,
    /// Aggregation function to return the addition
    Sum,
    /// Aggregation function to return the product
    Product,
    /// Aggregation function to return the average
    Average,
    /// Aggregation function to return the integral
    Integral,
    /// Aggregation function to return the conjunction, i.e., the sliding window returns true iff ALL values on the accessed stream inside a window are assigned to true
    Conjunction,
    /// Aggregation function to return the disjunction, i.e., the sliding window returns true iff AT LEAST ONE value on the accessed stream inside a window is assigned to true
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

/// Describes the operation used to access a stream with a offset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Offset {
    /// Discrete offset
    Discrete(i16),
    /// Real-time offset
    RealTime(Rational, TimeUnit),
}

/// Supported time unit for real time expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
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
/**
An AST node representing the declaration of a literal
*/
pub struct Literal {
    /// The kind of the literal, e.g., boolean, string, numeric, ...
    pub kind: LitKind,
    /// The ID of the node in the AST
    pub id: NodeId,
    /// The span in the specification declaring the extend declaration
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
/// The AST representation of literals
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
