/*!
This module describes the intermediate representation of a specification.
*/

pub(crate) mod lowering;
mod print;
mod schedule;

pub use crate::ast::StreamAccessKind;
pub use crate::ast::WindowOperation;
pub use crate::ir::schedule::{Deadline, Schedule};
pub use crate::ty::{Activation, FloatTy, IntTy, UIntTy, ValueTy}; // Re-export needed for IR

use std::time::Duration;
use uom::si::rational64::Frequency as UOM_Frequency;
use uom::si::rational64::Time as UOM_Time;

/// Intermediate representation of an RTLola specification.
/// Contains all relevant information found in the underlying specification and is enriched with information collected in semantic analyses.
#[derive(Debug, Clone, PartialEq)]
pub struct RTLolaIR {
    /// All input streams.
    pub inputs: Vec<InputStream>,
    /// All output streams with the bare minimum of information.
    pub outputs: Vec<OutputStream>,
    /// References to all time-driven streams.
    pub time_driven: Vec<TimeDrivenStream>,
    /// References to all event-driven streams.
    pub event_driven: Vec<EventDrivenStream>,
    /// A collection of all sliding windows.
    pub sliding_windows: Vec<SlidingWindow>,
    /// A collection of triggers
    pub triggers: Vec<Trigger>,
}

/// Represents a value type. Stream types are no longer relevant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// A binary type
    Bool,
    /// An integer type containing an enum stating its bit-width.
    Int(IntTy),
    /// An unsigned integer type containing an enum stating its bit-width.
    UInt(UIntTy),
    /// An floating point number type containing an enum stating its bit-width.
    Float(FloatTy),
    /// A unicode string
    String,
    /// A sequence of 8bit bytes
    Bytes,
    /// A n-ary tuples where n is the length of the contained vector.
    Tuple(Vec<Type>),
    /// An optional value type, e.g., resulting from accessing a stream with offset -1
    Option(Box<Type>),
    /// A type describing a function containing its argument types and return type. Resolve ambiguities in polymorphic functions and operations.
    Function(Vec<Type>, Box<Type>),
}

impl From<&ValueTy> for Type {
    fn from(ty: &ValueTy) -> Type {
        match ty {
            ValueTy::Bool => Type::Bool,
            ValueTy::Int(i) => Type::Int(*i),
            ValueTy::UInt(u) => Type::UInt(*u),
            ValueTy::Float(f) => Type::Float(*f),
            ValueTy::String => Type::String,
            ValueTy::Bytes => Type::Bytes,
            ValueTy::Tuple(t) => Type::Tuple(t.iter().map(|e| e.into()).collect()),
            ValueTy::Option(o) => Type::Option(Box::new(o.as_ref().into())),
            _ => unreachable!("cannot lower `ValueTy` {}", ty),
        }
    }
}

/// This enum indicates how much memory is required to store a stream.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MemorizationBound {
    /// The required memory might exceed any bound.
    Unbounded,
    /// No less then the contained amount of stream entries does ever need to be stored.
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

/// This data type provides information regarding how much data a stream needs to have access to from another stream.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Tracking {
    /// Need to store every single value of a stream
    All(StreamReference),
    /// Need to store `num` values of `trackee`, evicting/add a value every `rate` time units.
    Bounded {
        /// The stream that will be tracked.
        trackee: StreamReference,
        /// The number of values that will be accessed.
        num: u128,
        /// The duration in which values might be accessed.
        rate: Duration,
    },
}

/// Represents an input stream in an RTLola specification.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct InputStream {
    /// The name of the stream.
    pub name: String,
    /// The type of the stream.
    pub ty: Type,
    /// What streams depend, i.e., access values of this stream.
    pub dependent_streams: Vec<Tracking>,
    /// Which sliding windows aggregate values of this stream.
    pub dependent_windows: Vec<WindowReference>,
    /// Indicates in which evaluation layer the stream is.  
    pub layer: u32,
    /// The amount of memory required for this stream.
    pub memory_bound: MemorizationBound,
    /// The reference pointing to this stream.
    pub reference: StreamReference,
}

/// Represents an output stream in an RTLola specification.
#[derive(Debug, PartialEq, Clone)]
pub struct OutputStream {
    /// The name of the stream.
    pub name: String,
    /// The type of the stream.
    pub ty: Type,
    /// The stream expression
    pub expr: Expression,
    /// The input streams on which this stream depends.
    pub input_dependencies: Vec<StreamReference>,
    /// The output streams on which this stream depends.
    pub outgoing_dependencies: Vec<Dependency>,
    /// The Tracking of all streams that depend on this stream.
    pub dependent_streams: Vec<Tracking>,
    /// The sliding windows depending on this stream.
    pub dependent_windows: Vec<WindowReference>,
    /// The amount of memory required for this stream.
    pub memory_bound: MemorizationBound,
    /// Indicates in which evaluation layer the stream is.  
    pub layer: u32,
    /// The reference pointing to this stream.
    pub reference: StreamReference,
    /// The activation condition, which indicates when this stream needs to be evaluated.  Will be empty if the stream has a fixed frequency.
    pub ac: Option<Activation<StreamReference>>,
}

/// Wrapper for output streams providing additional information specific to timedriven streams.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct TimeDrivenStream {
    /// A reference to the stream that is specified.
    pub reference: StreamReference,
    /// The evaluation frequency of the stream.
    pub frequency: UOM_Frequency,
    /// The duration between two evaluation cycles.
    pub extend_rate: Duration,
    /// The period of the stream.
    pub period: UOM_Time,
}

/// Wrapper for output streams providing additional information specific to event-based streams.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct EventDrivenStream {
    /// A reference to the stream that is specified.
    pub reference: StreamReference,
}

/// Wrapper for output streams that are actually triggers.  Provides additional information specific to triggers.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Trigger {
    /// The trigger message that is supposed to be conveyed to the user if the trigger reports a violation.
    pub message: String,
    /// A reference to the output stream representing the trigger.
    pub reference: StreamReference,
    /// The index of the trigger.
    pub trigger_idx: usize,
}

/// Represents an expression.
#[derive(Debug, PartialEq, Clone)]
pub struct Expression {
    /// The kind of expression.
    pub kind: ExpressionKind,
    /// The type of the expression.
    pub ty: Type,
}

/// The expressions of the IR.
#[derive(Debug, PartialEq, Clone)]
pub enum ExpressionKind {
    /// Loading a constant
    LoadConstant(Constant),
    /// Applying arithmetic or logic operation and its monomorphic type
    /// Arguments never need to be coerced, @see `Expression::Convert`.
    /// Unary: 1st argument -> operand
    /// Binary: 1st argument -> lhs, 2nd argument -> rhs
    /// n-ary: kth argument -> kth operand
    ArithLog(ArithLogOp, Vec<Expression>, Type),
    /// Accessing another stream with a potentially 0 offset
    /// 1st argument -> default
    OffsetLookup {
        /// The target of the lookup.
        target: StreamReference,
        /// The offset of the lookup.
        offset: Offset,
    },
    /// Accessing another stream
    StreamAccess(StreamReference, StreamAccessKind),
    /// A window expression over a duration
    WindowLookup(WindowReference),
    /// An if-then-else expression
    Ite {
        #[allow(missing_docs)]
        condition: Box<Expression>,
        #[allow(missing_docs)]
        consequence: Box<Expression>,
        #[allow(missing_docs)]
        alternative: Box<Expression>,
    },
    /// A tuple expression
    Tuple(Vec<Expression>),
    /// Represents an access to a specific tuple element.  The second argument indicates the index of the accessed element while the first produces the accessed tuple.
    TupleAccess(Box<Expression>, usize),
    /// A function call with its monomorphic type
    /// Argumentes never need to be coerced, @see `Expression::Convert`.
    Function(String, Vec<Expression>, Type),
    /// Converting a value to a different type
    Convert {
        /// The original type
        from: Type,
        /// The target type
        to: Type,
        /// The expression that produces a value of type `from` which should be converted to `to`.
        expr: Box<Expression>,
    },
    /// Transforms an optional value into a "normal" one
    Default {
        /// The expression that results in an optional value.
        expr: Box<Expression>,
        /// An infallible expression providing a default value of `expr` evaluates to `None`.
        default: Box<Expression>,
    },
}

/// Represents a constant value of a certain kind.
#[derive(Debug, PartialEq, Clone)]
pub enum Constant {
    #[allow(missing_docs)]
    Str(String),
    #[allow(missing_docs)]
    Bool(bool),
    #[allow(missing_docs)]
    UInt(u64),
    #[allow(missing_docs)]
    Int(i64),
    #[allow(missing_docs)]
    Float(f64),
}

/// Contains information regarding the dependency between two streams which occurs due to a lookup expression.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Dependency {
    /// The target of the lookup.
    pub stream: StreamReference,
    /// The offset of the lookup.
    pub offsets: Vec<Offset>,
}

/// Offset used in the lookup expression
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Offset {
    /// A strictly positive discrete offset, e.g., `4`, or `42`
    FutureDiscreteOffset(u32),
    /// A non-negative discrete offset, e.g., `0`, `-4`, or `-42`
    PastDiscreteOffset(u32),
    /// A positive real-time offset, e.g., `-3ms`, `-4min`, `-2.3h`
    FutureRealTimeOffset(Duration),
    /// A non-negative real-time offset, e.g., `0`, `4min`, `2.3h`
    PastRealTimeOffset(Duration),
}

/// Contains all arithmetical and logical operations.
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
    /// The `^` operator (bitwise xor)
    BitXor,
    /// The `&` operator (bitwise and)
    BitAnd,
    /// The `|` operator (bitwise or)
    BitOr,
    /// The `~` operator for one's complement
    BitNot,
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

/// Represents an instance of a sliding window.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SlidingWindow {
    /// The stream whose values will be aggregated.
    pub target: StreamReference,
    /// The duration over which the window aggregates.
    pub duration: Duration,
    /// Indicates whether or not the first aggregated value will be produced immediately or whether the window waits until `duration` has passed at least once.
    pub wait: bool,
    /// The aggregation operation.
    pub op: WindowOperation,
    /// A reference to this sliding window.
    pub reference: WindowReference,
    /// The type of value the window produces.
    pub ty: Type,
}

/////// Referencing Structures ///////

/// Allows for referencing a window instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowReference(usize);

impl WindowReference {
    /// Provides access to the index inside the reference.
    pub fn idx(self) -> usize {
        self.0
    }
}

/// Allows for referencing an input stream within the specification.
pub type InputReference = usize;
/// Allows for referencing an output stream within the specification.
pub type OutputReference = usize;

/// Allows for referencing a stream within the specification.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum StreamReference {
    /// References an input stream.
    InRef(InputReference),
    /// References an output stream.
    OutRef(OutputReference),
}

impl StreamReference {
    /// Returns the index inside the reference if it is an output reference.  Panics otherwise.
    pub fn out_ix(&self) -> usize {
        match self {
            StreamReference::InRef(_) => unreachable!(),
            StreamReference::OutRef(ix) => *ix,
        }
    }

    /// Returns the index inside the reference if it is an input reference.  Panics otherwise.
    pub fn in_ix(&self) -> usize {
        match self {
            StreamReference::OutRef(_) => unreachable!(),
            StreamReference::InRef(ix) => *ix,
        }
    }

    /// Returns the index inside the reference disregarding whether it is an input or output reference.
    pub fn ix_unchecked(&self) -> usize {
        match self {
            StreamReference::InRef(ix) | StreamReference::OutRef(ix) => *ix,
        }
    }
}

/// A trait for any kind of stream.
pub trait Stream {
    /// Returns the evaluation laying in which the stream resides.
    fn eval_layer(&self) -> u32;
    /// Indicates whether or not the stream is an input stream.
    fn is_input(&self) -> bool;
    /// Indicates how many values need to be memorized.
    fn values_to_memorize(&self) -> MemorizationBound;
    /// Produces a stream references referring to the stream.
    fn as_stream_ref(&self) -> StreamReference;
}

////////// Implementations //////////

impl MemorizationBound {
    /// Produces the memory bound.  Panics if it is unbounded.
    pub fn unwrap(self) -> u16 {
        match self {
            MemorizationBound::Bounded(b) => b,
            MemorizationBound::Unbounded => {
                unreachable!("Called `MemorizationBound::unwrap()` on an `Unbounded` value.")
            }
        }
    }

    /// Produces the memory bound.  If it is unbounded, the default value will be returned.
    pub fn unwrap_or(self, dft: u16) -> u16 {
        match self {
            MemorizationBound::Bounded(b) => b,
            MemorizationBound::Unbounded => dft,
        }
    }
    /// Produces `Some(v)` if the memory bound is finite and `v` and `None` if it is unbounded.
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

impl Expression {
    fn new(kind: ExpressionKind, ty: Type) -> Self {
        Self { kind, ty }
    }
}

impl PartialOrd for Offset {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        use Offset::*;
        match (self, other) {
            (PastDiscreteOffset(_), FutureDiscreteOffset(_))
            | (PastRealTimeOffset(_), FutureRealTimeOffset(_))
            | (PastDiscreteOffset(_), FutureRealTimeOffset(_))
            | (PastRealTimeOffset(_), FutureDiscreteOffset(_)) => Some(Ordering::Less),

            (FutureDiscreteOffset(_), PastDiscreteOffset(_))
            | (FutureDiscreteOffset(_), PastRealTimeOffset(_))
            | (FutureRealTimeOffset(_), PastDiscreteOffset(_))
            | (FutureRealTimeOffset(_), PastRealTimeOffset(_)) => Some(Ordering::Greater),

            (FutureDiscreteOffset(a), FutureDiscreteOffset(b)) => Some(a.cmp(b)),
            (PastDiscreteOffset(a), PastDiscreteOffset(b)) => Some(b.cmp(a)),

            (_, _) => unimplemented!(),
        }
    }
}

impl Ord for Offset {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl RTLolaIR {
    /// Returns a `Vec` containing a reference for each input stream in the specification.
    pub fn input_refs(&self) -> Vec<InputReference> {
        (0..self.inputs.len()).collect()
    }

    /// Returns a `Vec` containing a reference for each output stream in the specification.
    pub fn output_refs(&self) -> Vec<OutputReference> {
        (0..self.outputs.len()).collect()
    }

    /// Provides mutable access to an input stream.
    pub fn get_in_mut(&mut self, reference: StreamReference) -> &mut InputStream {
        match reference {
            StreamReference::InRef(ix) => &mut self.inputs[ix],
            StreamReference::OutRef(_) => unreachable!("Called `LolaIR::get_in` with a `StreamReference::OutRef`."),
        }
    }

    /// Provides immutable access to an input stream.
    pub fn get_in(&self, reference: StreamReference) -> &InputStream {
        match reference {
            StreamReference::InRef(ix) => &self.inputs[ix],
            StreamReference::OutRef(_) => unreachable!("Called `LolaIR::get_in` with a `StreamReference::OutRef`."),
        }
    }

    /// Provides mutable access to an output stream.
    pub fn get_out_mut(&mut self, reference: StreamReference) -> &mut OutputStream {
        match reference {
            StreamReference::InRef(_) => unreachable!("Called `LolaIR::get_out` with a `StreamReference::InRef`."),
            StreamReference::OutRef(ix) => &mut self.outputs[ix],
        }
    }

    /// Provides immutable access to an output stream.
    pub fn get_out(&self, reference: StreamReference) -> &OutputStream {
        match reference {
            StreamReference::InRef(_) => unreachable!("Called `LolaIR::get_out` with a `StreamReference::InRef`."),
            StreamReference::OutRef(ix) => &self.outputs[ix],
        }
    }

    /// Returns a `Vec` containing a reference for each stream in the specification.
    pub fn all_streams(&self) -> Vec<StreamReference> {
        self.input_refs()
            .iter()
            .map(|ix| StreamReference::InRef(*ix))
            .chain(self.output_refs().iter().map(|ix| StreamReference::OutRef(*ix)))
            .collect()
    }

    /// Returns a `Vec` containing a reference to an output stream representing a trigger in the specification.
    pub fn get_triggers(&self) -> Vec<&OutputStream> {
        self.triggers.iter().map(|t| self.get_out(t.reference)).collect()
    }

    /// Returns a `Vec` containing a reference for each event-driven output stream in the specification.
    pub fn get_event_driven(&self) -> Vec<&OutputStream> {
        self.event_driven.iter().map(|t| self.get_out(t.reference)).collect()
    }

    /// Returns a `Vec` containing a reference for each time-driven output stream in the specification.
    pub fn get_time_driven(&self) -> Vec<&OutputStream> {
        self.time_driven.iter().map(|t| self.get_out(t.reference)).collect()
    }

    /// Returns a `Vec` containing a reference for each sliding window in the specification.
    pub fn get_window(&self, window: WindowReference) -> &SlidingWindow {
        &self.sliding_windows[window.0]
    }

    /// Provides a representation for the evaluation layers of all event-driven output streams.  Each element of the outer `Vec` represents a layer, each element of the inner `Vec` a stream in the layer.
    pub fn get_event_driven_layers(&self) -> Vec<Vec<OutputReference>> {
        if self.event_driven.is_empty() {
            return vec![];
        }

        // Zip eval layer with stream reference.
        let streams_with_layers: Vec<(usize, OutputReference)> = self
            .event_driven
            .iter()
            .map(|s| s.reference)
            .map(|r| (self.get_out(r).eval_layer() as usize, r.out_ix()))
            .collect();

        // Streams are annotated with an evaluation layer. The layer is not minimal, so there might be
        // layers without entries and more layers than streams.
        // Minimization works as follows:
        // a) Find the greatest layer
        // b) For each potential layer...
        // c) Find streams that would be in it.
        // d) If there is none, skip this layer
        // e) If there are some, add them as layer.

        // a) Find the greatest layer. Maximum must exist because vec cannot be empty.
        let max_layer = streams_with_layers.iter().max_by_key(|(layer, _)| layer).unwrap().0;

        let mut layers = Vec::new();
        // b) For each potential layer
        for i in 0..=max_layer {
            // c) Find streams that would be in it.
            let in_layer_i: Vec<OutputReference> =
                streams_with_layers.iter().filter_map(|(l, r)| if *l == i { Some(*r) } else { None }).collect();
            if in_layer_i.is_empty() {
                // d) If there is none, skip this layer
                continue;
            } else {
                // e) If there are some, add them as layer.
                layers.push(in_layer_i);
            }
        }
        layers
    }

    /// Computes a schedule for all time-driven streams.
    pub fn compute_schedule(&self) -> Result<Schedule, String> {
        Schedule::from(self)
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
    /// Indicates how many bytes a type requires to be stored in memory.
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
            Type::Float(FloatTy::F16) => Some(ValSize(2)),
            Type::Float(FloatTy::F32) => Some(ValSize(4)),
            Type::Float(FloatTy::F64) => Some(ValSize(8)),
            Type::Option(_) => unimplemented!("Size of option not determined, yet."),
            Type::Tuple(t) => {
                let size = t.iter().map(|t| Type::size(t).unwrap().0).sum();
                Some(ValSize(size))
            }
            Type::String | Type::Bytes => unimplemented!("Size of Strings not determined, yet."),
            Type::Function(_, _) => None,
        }
    }
}
