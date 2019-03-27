use crate::ty::ValueTy;
pub use crate::ty::{FloatTy, IntTy, UIntTy}; // Re-export needed for IR
use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub struct LolaIR {
    /// All input streams.
    pub inputs: Vec<InputStream>,
    /// All output streams with the bare minimum of information.
    pub outputs: Vec<OutputStream>,
    /// References to all time-driven streams.
    pub time_driven: Vec<TimeDrivenStream>,
    /// References to all event-driven streams.
    pub event_driven: Vec<EventDrivenStream>,
    /// References to all parametrized streams.
    pub parametrized: Vec<ParametrizedStream>,
    /// A collection of all sliding windows.
    pub sliding_windows: Vec<SlidingWindow>,
    /// A collection of triggers
    pub triggers: Vec<Trigger>,
    /// A collection of flags representing features the specification requires.
    pub feature_flags: Vec<FeatureFlag>,
}

/// Represents a type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Bool,
    Int(IntTy),
    UInt(UIntTy),
    Float(FloatTy),
    // an abstract data type, e.g., structs, enums, etc.
    //Adt(AdtDef),
    String,
    Tuple(Vec<Type>),
    /// an optional value type, e.g., resulting from accessing a stream with offset -1
    Option(Box<Type>),
}

impl From<&ValueTy> for Type {
    fn from(ty: &ValueTy) -> Type {
        match ty {
            ValueTy::Bool => Type::Bool,
            ValueTy::Int(i) => Type::Int(*i),
            ValueTy::UInt(u) => Type::UInt(*u),
            ValueTy::Float(f) => Type::Float(*f),
            ValueTy::String => Type::String,
            ValueTy::Tuple(t) => Type::Tuple(t.iter().map(|e| e.into()).collect()),
            ValueTy::Option(o) => Type::Option(Box::new(o.as_ref().into())),
            _ => unreachable!("cannot lower `ValueTy` {}", ty),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MemorizationBound {
    Unbounded,
    Bounded(u16),
}

impl PartialOrd for MemorizationBound {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        use MemorizationBound::*;
        match (self, other) {
            (Unbounded, Unbounded) => None,
            (Bounded(_), Unbounded) => Some(Ordering::Less),
            (Unbounded, Bounded(_)) => Some(Ordering::Greater),
            (Bounded(b1), Bounded(b2)) => Some(b1.cmp(&b2)),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Tracking {
    /// Need to store every single value of a stream
    All(StreamReference),
    /// Need to store `num` values of `trackee`, evicting/add a value every `rate` time units.
    Bounded { trackee: StreamReference, num: u128, rate: Duration },
}

/// Represents an input stream of a Lola specification.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct InputStream {
    pub name: String,
    pub ty: Type,
    pub dependent_streams: Vec<Tracking>,
    pub dependent_windows: Vec<WindowReference>,
    pub(crate) layer: u32,
    pub(crate) memory_bound: MemorizationBound,
    pub reference: StreamReference,
}

/// Represents an output stream in a Lola specification.
#[derive(Debug, PartialEq, Clone)]
pub struct OutputStream {
    pub name: String,
    pub ty: Type,
    pub expr: Expression,
    pub input_dependencies: Vec<StreamReference>,
    pub outgoing_dependencies: Vec<Dependency>,
    pub dependent_streams: Vec<Tracking>,
    pub dependent_windows: Vec<WindowReference>,
    pub(crate) memory_bound: MemorizationBound,
    pub(crate) layer: u32,
    pub reference: StreamReference,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct TimeDrivenStream {
    pub reference: StreamReference,
    pub extend_rate: Duration,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct EventDrivenStream {
    pub reference: StreamReference,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ParametrizedStream {
    pub reference: StreamReference,
    pub params: Vec<Parameter>,
    pub invoke: Option<StreamReference>,
    pub extend: Option<StreamReference>,
    pub terminate: Option<StreamReference>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Trigger {
    pub message: Option<String>,
    pub reference: StreamReference,
}

/// Represents a parameter, i.e. a name and a type.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Parameter {
    pub name: String,
    pub ty: Type,
}

/// An expression in the IR is a list of executable statements
#[derive(Debug, PartialEq, Clone)]
pub struct Expression {
    /// A list of statements where the last statement represents the result of the expression
    pub stmts: Vec<Statement>,
    /// A list of temporary values, used in the statements
    pub temporaries: Vec<Type>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Temporary(pub usize);

impl From<u32> for Temporary {
    fn from(u: u32) -> Temporary {
        Temporary(u as usize)
    }
}

/// A statement is of the form `target = op <arguments>`
#[derive(Debug, PartialEq, Clone)]
pub struct Statement {
    /// the name of the temporary
    pub target: Temporary,
    pub op: Op,
    pub args: Vec<Temporary>,
}

/// the operations (instruction set) of the IR
#[derive(Debug, PartialEq, Clone)]
pub enum Op {
    /// Loading a constant
    /// 1st argument -> Constant.
    LoadConstant(Constant),
    /// Applying arithmetic or logic operation
    /// Unary: 1st argument -> operand
    /// Binary: 1st argument -> lhs, 2nd argument -> rhs
    /// n-ary: kth argument -> kth operand
    ArithLog(ArithLogOp),
    /// Accessing another stream
    /// 1st argument -> default
    /// StreamInstance contains further arguments.
    StreamLookup { instance: StreamInstance, offset: Offset },
    /// Accessing another stream under sample and hold semantics
    /// 1st argument -> default
    /// StreamInstance contains further arguments.
    SampleAndHoldStreamLookup { instance: StreamInstance, offset: Offset },
    /// Accessing another stream synchronously
    /// No arguments; StreamInstance contains further arguments.
    SyncStreamLookup(StreamInstance),
    /// A window expression over a duration
    WindowLookup(WindowReference),
    /// An if-then-else expression
    /// One argument: The register with the condition.
    Ite { consequence: Vec<Statement>, alternative: Vec<Statement> },
    /// A tuple expression
    /// Arguments: values of the tuple.
    Tuple,
    /// A function call
    /// Arguments in statement in proper order.
    Function(String),
    /// Converts a value to a different type.
    /// One arguments containing the value to convert.
    Convert,
    /// Moves the value of a temporary to a different one.
    /// One argument, the source temporary.
    Move,
}

/// Represents a constant value of a certain kind.
#[derive(Debug, PartialEq, Clone)]
pub enum Constant {
    Str(String),
    Bool(bool),
    Int(i128),
    Float(f64),
}

/// Represents a single instance of a stream. The stream template is accessible by the reference,
/// the specific instance by the arguments.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct StreamInstance {
    pub reference: StreamReference,
    pub arguments: Vec<Temporary>,
}

///TODO
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Dependency {
    pub stream: StreamReference,
    pub offsets: Vec<Offset>,
}

/// Offset used in the lookup expression
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Offset {
    /// A positive discrete offset, e.g., `4`, or `42`
    FutureDiscreteOffset(u128),
    /// A non-positive discrete offset, e.g., `0`, `-4`, or `-42`
    PastDiscreteOffset(u128),
    /// A positive real-time offset, e.g., `-3ms`, `-4min`, `-2.3h`
    FutureRealTimeOffset(Duration),
    /// A non-positive real-time offset, e.g., `0`, `4min`, `2.3h`
    PastRealTimeOffset(Duration),
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum WindowOperation {
    Sum,
    Product,
    Average,
    Count,
    Integral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithLogOp {
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
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

/// Represents an instance of a sliding window.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SlidingWindow {
    pub target: StreamReference,
    pub duration: Duration,
    pub op: WindowOperation,
    pub reference: WindowReference,
    pub ty: Type,
}

/// Each flag represents a certain feature of Lola not necessarily available in all version of the
/// language or for all functions of the front-end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureFlag {
    DiscreteFutureOffset,
    RealTimeOffset,
    RealTimeFutureOffset,
    SlidingWindows,
    DiscreteWindows,
    UnboundedMemory,
}

/////// Referencing Structures ///////

/// Allows for referencing a window instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowReference {
    pub ix: usize,
}

/// Allows for referencing a stream within the specification.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum StreamReference {
    InRef(usize),
    OutRef(usize),
}

impl StreamReference {
    pub fn out_ix(&self) -> usize {
        match self {
            StreamReference::InRef(_) => panic!(),
            StreamReference::OutRef(ix) => *ix,
        }
    }

    pub fn in_ix(&self) -> usize {
        match self {
            StreamReference::OutRef(_) => panic!(),
            StreamReference::InRef(ix) => *ix,
        }
    }

    pub fn ix_unchecked(&self) -> usize {
        match self {
            StreamReference::InRef(ix) | StreamReference::OutRef(ix) => *ix,
        }
    }
}

/// A trait for any kind of stream.
pub trait Stream {
    fn eval_layer(&self) -> u32;
    fn is_input(&self) -> bool;
    fn values_to_memorize(&self) -> MemorizationBound;
    fn as_stream_ref(&self) -> StreamReference;
}

////////// Implementations //////////

impl MemorizationBound {
    pub fn unwrap(self) -> u16 {
        match self {
            MemorizationBound::Bounded(b) => b,
            MemorizationBound::Unbounded => panic!("Called `MemorizationBound::unwrap()` on an `Unbounded` value."),
        }
    }
    pub fn unwrap_or(self, dft: u16) -> u16 {
        match self {
            MemorizationBound::Bounded(b) => b,
            MemorizationBound::Unbounded => dft,
        }
    }
    pub fn as_opt(self) -> Option<u16> {
        match self {
            MemorizationBound::Bounded(b) => Some(b),
            MemorizationBound::Unbounded => None,
        }
    }
}

impl Stream for OutputStream {
    fn eval_layer(&self) -> u32 {
        self.layer
    }
    fn is_input(&self) -> bool {
        false
    }
    fn values_to_memorize(&self) -> MemorizationBound {
        self.memory_bound
    }
    fn as_stream_ref(&self) -> StreamReference {
        self.reference
    }
}

impl Stream for InputStream {
    fn eval_layer(&self) -> u32 {
        self.layer
    }
    fn is_input(&self) -> bool {
        true
    }
    fn values_to_memorize(&self) -> MemorizationBound {
        self.memory_bound
    }
    fn as_stream_ref(&self) -> StreamReference {
        self.reference
    }
}

impl LolaIR {
    pub fn input_refs(&self) -> Vec<StreamReference> {
        self.inputs.iter().map(|s| (s as &Stream).as_stream_ref()).collect()
    }

    pub fn output_refs(&self) -> Vec<StreamReference> {
        self.outputs.iter().map(|s| (s as &Stream).as_stream_ref()).collect()
    }

    pub(crate) fn get_in_mut(&mut self, reference: StreamReference) -> &mut InputStream {
        match reference {
            StreamReference::InRef(ix) => &mut self.inputs[ix],
            StreamReference::OutRef(_) => panic!("Called `LolaIR::get_in` with a `StreamReference::OutRef`."),
        }
    }

    pub fn get_in(&self, reference: StreamReference) -> &InputStream {
        match reference {
            StreamReference::InRef(ix) => &self.inputs[ix],
            StreamReference::OutRef(_) => panic!("Called `LolaIR::get_in` with a `StreamReference::OutRef`."),
        }
    }

    pub(crate) fn get_out_mut(&mut self, reference: StreamReference) -> &mut OutputStream {
        match reference {
            StreamReference::InRef(_) => panic!("Called `LolaIR::get_out` with a `StreamReference::InRef`."),
            StreamReference::OutRef(ix) => &mut self.outputs[ix],
        }
    }

    pub fn get_out(&self, reference: StreamReference) -> &OutputStream {
        match reference {
            StreamReference::InRef(_) => panic!("Called `LolaIR::get_out` with a `StreamReference::InRef`."),
            StreamReference::OutRef(ix) => &self.outputs[ix],
        }
    }

    pub fn all_streams(&self) -> Vec<StreamReference> {
        self.input_refs().iter().chain(self.output_refs().iter()).cloned().collect()
    }

    pub fn get_triggers(&self) -> Vec<&OutputStream> {
        self.triggers.iter().map(|t| self.get_out(t.reference)).collect()
    }

    pub fn get_event_driven(&self) -> Vec<&OutputStream> {
        self.event_driven.iter().map(|t| self.get_out(t.reference)).collect()
    }

    pub fn get_time_driven(&self) -> Vec<&OutputStream> {
        self.time_driven.iter().map(|t| self.get_out(t.reference)).collect()
    }

    pub fn get_parametrized(&self) -> Vec<&OutputStream> {
        self.parametrized.iter().map(|t| self.get_out(t.reference)).collect()
    }

    pub fn get_window(&self, window: WindowReference) -> &SlidingWindow {
        &self.sliding_windows[window.ix]
    }
}

/// The size of a specific value in bytes.
#[derive(Debug, Clone, Copy)]
pub struct ValSize(pub u32); // Needs to be reasonable large for compound types.

impl From<u8> for ValSize {
    fn from(val: u8) -> ValSize {
        ValSize(u32::from(val))
    }
}

impl std::ops::Add for ValSize {
    type Output = ValSize;
    fn add(self, rhs: ValSize) -> ValSize {
        ValSize(self.0 + rhs.0)
    }
}

impl Type {
    pub fn size(&self) -> Option<ValSize> {
        match self {
            Type::Bool => Some(ValSize(1)),
            Type::Int(IntTy::I8) => Some(ValSize(1)),
            Type::Int(IntTy::I16) => Some(ValSize(2)),
            Type::Int(IntTy::I32) => Some(ValSize(4)),
            Type::Int(IntTy::I64) => Some(ValSize(8)),
            Type::UInt(UIntTy::U8) => Some(ValSize(1)),
            Type::UInt(UIntTy::U16) => Some(ValSize(2)),
            Type::UInt(UIntTy::U32) => Some(ValSize(4)),
            Type::UInt(UIntTy::U64) => Some(ValSize(8)),
            Type::Float(FloatTy::F32) => Some(ValSize(4)),
            Type::Float(FloatTy::F64) => Some(ValSize(8)),
            Type::Option(_) => panic!("Should not be used directly!"),
            Type::Tuple(t) => {
                let size = t.iter().map(|t| Type::size(t).unwrap().0).sum();
                Some(ValSize(size))
            }
            Type::String => panic!("Strings are not properly handled yet."),
        }
    }
}
