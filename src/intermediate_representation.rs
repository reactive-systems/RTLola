#[derive(Debug)]
pub struct IntermediateRepresentation {
    /// All input streams.
    pub inputs: Vec<InputStream>,
    /// All event-triggered output streams and triggers. See `OutputStream` for more information.
    pub event_outputs: Vec<OutputStream>,
    /// All time-triggered output streams and triggers. See `OutputStream` for more information.
    pub time_outputs: Vec<OutputStream>,
    /// A collection of all sliding windows.
    pub sliding_windows: Vec<SlidingWindow>,
    /// A collection of flags representing features the specification requires.
    pub feature_flags: Vec<FeatureFlag>,
}

/// Represents an atomic type, i.e. a type that is not composed of other types.
#[derive(Debug, Clone, Copy)]
pub enum AtomicType {
    Int(u8),
    UInt(u8),
    Float(u8),
    String,
    Bool,
}

/// Represents a type that is either atomic or a composed of other types, such as a tuple.
/// Allows for computing the required memory to store one value of this type.
#[derive(Debug, Clone)]
pub enum Type {
    Atomic(AtomicType),
    Tuple(Vec<AtomicType>),
}

/// Represents an input stream of a Lola specification.
#[derive(Debug)]
pub struct InputStream {
    pub name: String,
    pub ty: Type,
}

/// Represents a parameter, i.e. a name and a type.
#[derive(Debug)]
pub struct Parameter {
    pub name: String,
    pub ty: Type,
}

/// Represents a constant value of a certain kind.
#[derive(Debug)]
pub enum Constant {
    Str(String),
    Bool(bool),
    Int(i128),
    Float(f64),
}

/// Represents a single instance of a stream. The stream template is accessible by the reference,
/// the specific instance by the arguments.
#[derive(Debug)]
pub struct StreamInstance {
    pub reference: StreamReference,
    pub arguments: Vec<Box<Expression>>,
}

/// Offset used in the lookup expression
#[derive(Debug)]
pub enum Offset {
    /// A discrete offset, e.g., `0`, `-4`, or `42`
    DiscreteOffset(Box<Expression>),
    /// A real-time offset, e.g., `3ms`, `4min`, `2.3h`
    RealTimeOffset(Duration),
}

#[derive(Debug)]
pub enum WindowOperation {
    Sum,
    Product,
    Average,
    Count,
    Integral,
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

#[derive(Debug, PartialEq, Eq)]
pub enum FunctionKind {
    NthRoot,
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

/// Specifies a duration as well as a rate, normalized to ms.
#[derive(Debug)]
pub struct Duration {
    pub expr: Box<Expression>,
    pub factor: f64,
}

/// Specifies the kind of an expression and contains all information specific to this kind.
#[derive(Debug)]
pub enum ExpressionKind {
    // I chose to call it `Constant` rather than `Literal`, because it might be the result of
    // constant folding or similar, and not originally a literal.
    Const(Constant),
    /// An identifier, e.g., `foo`
    // TODO: This can only refer to a parameter. Use indices for accessing params, store index here.
    // TODO: Irrelevant until parametrization is introduced.
    Ident(String),
    /// A default expression, e.g., ` a ? 0 `
    Default(Box<Expression>, Box<Expression>),
    /// A stream lookup with offset
    StreamLookup(StreamInstance, Offset),
    /// A window expression over a duration
    WindowLookup(WindowReference),
    /// A binary operation (For example: `a + b`, `a * b`)
    Binary(BinOp, Box<Expression>, Box<Expression>),
    /// A unary operation (For example: `!x`, `*x`)
    Unary(UnOp, Box<Expression>),
    /// An if-then-else expression
    Ite(Box<Expression>, Box<Expression>, Box<Expression>),
    /// A tuple expression
    Tuple(Vec<Box<Expression>>),
    /// A function call
    Function(FunctionKind, Vec<Box<Expression>>),
    // /// An aggregation such as `count`, `exists` or `forall`. Note: This is not a window aggregation
    // TODO: Does not exist, yet.
    //    Aggregation(AggregationKind, Parameter, Box<Expression>),
}

/// Represents an arbitrary expression. Specific information is available in the `kind`.
#[derive(Debug)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub ty: Type,
}

/// Represents an output stream in a Lola specification. The optional `message` is supposed to be
/// printed when any instance of this stream produces a `true` output.
#[derive(Debug)]
pub struct OutputStream {
    pub name: String,
    pub ty: Type,
    pub params: Vec<Parameter>,
    pub expr: Expression,
    pub message: Option<String>,
    // TODO: Check in constructor that message is only available when type is bool.
}

/// Represents an instance of a sliding window.
#[derive(Debug)]
pub struct SlidingWindow {
    pub target: StreamReference,
    pub duration: Duration,
    pub op: WindowOperation,
}

/// Each flag represents a certain feature of Lola not necessarily available in all version of the
/// language or for all functions of the front-end.
#[derive(Clone, Copy, Debug)]
pub enum FeatureFlag {
    DiscreteFutureOffset,
    RealTimeOffset,
    RealTimeFutureOffset,
    SlidingWindows,
    DiscreteWindows,
}

/////// Referencing Structures ///////

/// Allows for referencing a window instance.
#[derive(Debug, Clone, Copy)]
pub struct WindowReference {
    pub ix: usize,
}

/// Allows for referencing a stream within the specification.
#[derive(Debug, Clone, Copy)]
pub enum StreamReference {
    InRef(usize),
    OutRef { timed: bool, ix: usize },
}

/// A super-type for any kind of stream.
#[derive(Debug)]
pub enum Stream<'a> {
    In(&'a InputStream),
    Out(&'a OutputStream),
}

////////// Implementations //////////

impl<'a> IntermediateRepresentation {
    pub fn outputs(&'a self) -> Vec<Stream<'a>> {
        self.event_outputs
            .iter()
            .chain(self.time_outputs.iter())
            .map(|s| Stream::Out(s))
            .collect()
    }
    pub fn get(&'a self, reference: StreamReference) -> Stream<'a> {
        match reference {
            StreamReference::InRef(ix) => Stream::In(&self.inputs[ix]),
            StreamReference::OutRef { timed: true, ix } => Stream::Out(&self.time_outputs[ix]),
            StreamReference::OutRef { timed: false, ix } => Stream::Out(&self.event_outputs[ix]),
        }
    }
}

impl AtomicType {
    fn size(self) -> ValSize {
        match self {
            AtomicType::Int(w) | AtomicType::UInt(w) | AtomicType::Float(w) => u32::from(*w),
            AtomicType::String => unimplemented!(), // String handling is not clear, yet.
            AtomicType::Bool => 1,
        }
    }
}

/// The size of a specific value in bytes.
pub type ValSize = u32; // Needs to be reasonable large for compound types.

impl Type {
    fn size(&self) -> ValSize {
        match self {
            Type::Atomic(a) => a.size(),
            Type::Tuple(v) => v.iter().map(|x| x.size()).sum(),
        }
    }
}
