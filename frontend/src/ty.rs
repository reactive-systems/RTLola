//! This module contains the basic definition of types
//!
//! It is inspired by <https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/index.html>

pub(crate) mod check;
pub(crate) mod unifier;

use crate::parse::NodeId;
use lazy_static::lazy_static;
use num::rational::Rational64 as Rational;
use num::{CheckedDiv, Integer};
use unifier::ValueVar;
use uom::si::frequency::hertz;
use uom::si::rational64::Frequency as UOM_Frequency;

/**
Configuration of the type system in the frontend.
*/
#[derive(Debug, Clone, Copy, Default)]
pub struct TypeConfig {
    /// allow only 64bit types, i.e., `Int64`, `UInt64`, and `Float64`
    pub use_64bit_only: bool,
    /// include type aliases `Int` -> `Int64`, `UInt` -> `UInt64`, and `Float` -> `Float64`
    pub type_aliases: bool,
}

/// The type of an expression consists of both, a value type (`Bool`, `String`, etc.) and
/// a stream type (periodic or event-based).
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub struct Ty {
    value: ValueTy,
    stream: StreamTy,
}

/**
The possible stream types describing the temporal behavior of a stream.
*/
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub enum StreamTy {
    /// An event stream with the given dependencies
    Event(Activation<NodeId>),
    /// A real-time stream with given frequency
    RealTime(Freq),
    /// **INTERNAL USE**: The type of the stream should be inferred as the conjunction of the given `StreamTy`
    Infer(Vec<NodeId>),
}

/// The `value` type, storing information about the stored values (`Bool`, `UInt8`, etc.)
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub enum ValueTy {
    /// The boolean value type.
    Bool,
    /// A signed integer value type. See `IntTy` for more information.
    Int(IntTy),
    /// An unsigned integer value type. See `UIntTy` for more information.
    UInt(UIntTy),
    /// A floating-point value type. See `FloatTy` for more information.
    Float(FloatTy),
    // an abstract data type, e.g., structs, enums, etc.
    //Adt(AdtDef),
    /// A utf-8 encoded string type.
    String,
    /// A byte string type.
    Bytes,
    /// A tuple of value types.
    Tuple(Vec<ValueTy>),
    /// an optional value type, e.g., resulting from accessing a stream with offset -1
    Option(Box<ValueTy>),
    /// Used during type inference
    Infer(ValueVar),
    /// Constraint used during type inference
    Constr(TypeConstraint),
    /// A reference to a generic parameter in a function declaration, e.g. `T` in `a<T>(x:T) -> T`
    Param(u8, String),
    /**
     **INTERNAL USE**: A type error.
     */
    Error,
}

/**
The possible signed integer value types.
*/
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum IntTy {
    /**
    Signed 8-bit integer value type.
    */
    I8,
    /**
    Signed 16-bit integer value type.
    */
    I16,
    /**
    Signed 32-bit integer value type.
    */
    I32,
    /**
    Signed 64-bit integer value type.
    */
    I64,
}
use self::IntTy::*;

/**
The possible unsigned integer value types.
*/
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum UIntTy {
    /**
    Unsigned 8-bit integer value type.
    */
    U8,
    /**
    Unsigned 16-bit integer value type.
    */
    U16,
    /**
    Unsigned 32-bit integer value type.
    */
    U32,
    /**
    Unsigned 64-bit integer value type.
    */
    U64,
}
use self::UIntTy::*;

/**
The possible floating-point value types.
*/
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum FloatTy {
    /**
    16-bit floating-point value type.
    */
    F16,
    /**
    32-bit floating-point value type.
    */
    F32,
    /**
    64-bit floating-point value type.
    */
    F64,
}
use self::FloatTy::*;

/**
The frequency with which a stream gets executed.
*/
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct Freq {
    /**
    The frequency stored in the format provided by the `uom` crate.
    */
    pub freq: UOM_Frequency,
}

/**
The activation condition describes when an event-based stream produces a new value.
*/
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum Activation<Var> {
    /**
    When all of the activation conditions is true.
    */
    Conjunction(Vec<Self>),
    /**
    When one of the activation conditions is true.
    */
    Disjunction(Vec<Self>),
    /**
    Whenever the specified stream produces a new value.
    */
    Stream(Var),
    /**
    Whenever an event-based stream produces a new value.
    */
    True,
}

impl std::fmt::Display for Freq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.freq.clone().into_format_args(hertz, uom::fmt::DisplayStyle::Abbreviation))
    }
}

impl StreamTy {
    pub(crate) fn new_event(activation: Activation<NodeId>) -> StreamTy {
        StreamTy::Event(activation)
    }

    pub(crate) fn new_periodic(freq: Freq) -> StreamTy {
        StreamTy::RealTime(freq)
    }

    pub(crate) fn is_valid(&self, right: &StreamTy) -> Result<bool, String> {
        // RealTime<freq_self> -> RealTime<freq_right> if freq_left is multiple of freq_right
        match (&self, &right) {
            (StreamTy::RealTime(target), StreamTy::RealTime(other)) => {
                // coercion is only valid if `other` is a multiple of `target`,
                // for example, `target = 3Hz` and `other = 12Hz`
                other.is_multiple_of(target)
            }
            (StreamTy::Event(target), StreamTy::Event(other)) => {
                // coercion is only valid if the implication `target -> other` is valid,
                // for example, `target = a && b` and `other = b` is valid while
                //              `target = a || b` and `other = b` is invalid.
                Ok(target.implies_valid(other))
            }
            _ => Ok(false),
        }
    }

    pub(crate) fn simplify(&mut self) {
        let ac = match self {
            StreamTy::Event(ac) => ac,
            _ => return,
        };
        match ac {
            Activation::Conjunction(args) if args.is_empty() => *ac = Activation::True,
            Activation::Conjunction(args) | Activation::Disjunction(args) => {
                args.sort();
                args.dedup();
            }
            _ => {}
        }
    }
}

impl Activation<NodeId> {
    /// Checks whether `self -> other` is valid
    pub(crate) fn implies_valid(&self, other: &Self) -> bool {
        if self == other {
            return true;
        }
        match (self, other) {
            (Activation::Conjunction(left), Activation::Conjunction(right)) => {
                right.iter().all(|cond| left.contains(cond))
            }
            (Activation::Conjunction(left), _) => left.contains(other),
            (_, Activation::True) => true,
            _ => {
                // there are possible many more cases that we want to look at in order to make analysis more precise
                false
            }
        }
    }

    pub(crate) fn conjunction(&self, other: &Self) -> Self {
        use Activation::*;
        match (self, other) {
            (True, _) => other.clone(),
            (Conjunction(c_l), Conjunction(c_r)) => {
                let mut con = c_l.clone();
                con.extend(c_r.iter().cloned());
                Conjunction(con)
            }
            (Conjunction(c), other) | (other, Conjunction(c)) => {
                let mut con = c.clone();
                con.push(other.clone());
                Conjunction(con)
            }
            (_, _) => Conjunction(vec![self.clone(), other.clone()]),
        }
    }
}

impl Freq {
    pub(crate) fn new(freq: UOM_Frequency) -> Self {
        Freq { freq }
    }

    pub(crate) fn is_multiple_of(&self, other: &Freq) -> Result<bool, String> {
        let lhs = self.freq.get::<hertz>();
        let rhs = other.freq.get::<hertz>();
        if lhs < rhs {
            return Ok(false);
        }
        match lhs.checked_div(&rhs) {
            Some(q) => Ok(q.is_integer()),
            None => Err(format!("division of frequencies `{:?}`/`{:?}` failed", &self.freq, &other.freq)),
        }
    }

    pub(crate) fn conjunction(&self, other: &Freq) -> Freq {
        let numer_left = *self.freq.get::<hertz>().numer();
        let numer_right = *other.freq.get::<hertz>().numer();
        let denom_left = *self.freq.get::<hertz>().denom();
        let denom_right = *other.freq.get::<hertz>().denom();
        // gcd(self, other) = gcd(numer_left, numer_right) / lcm(denom_left, denom_right)
        // only works if rational numbers are reduced, which ist the default for `Rational`
        Freq {
            freq: UOM_Frequency::new::<hertz>(Rational::new(
                numer_left.gcd(&numer_right),
                denom_left.lcm(&denom_right),
            )),
        }
    }
}

lazy_static! {
    static ref PRIMITIVE_TYPES: Vec<(&'static str, &'static ValueTy)> = vec![
        ("Bool", &ValueTy::Bool),
        ("Int8", &ValueTy::Int(I8)),
        ("Int16", &ValueTy::Int(I16)),
        ("Int32", &ValueTy::Int(I32)),
        ("Int64", &ValueTy::Int(I64)),
        ("UInt8", &ValueTy::UInt(U8)),
        ("UInt16", &ValueTy::UInt(U16)),
        ("UInt32", &ValueTy::UInt(U32)),
        ("UInt64", &ValueTy::UInt(U64)),
        ("Float16", &ValueTy::Float(F16)),
        ("Float32", &ValueTy::Float(F32)),
        ("Float64", &ValueTy::Float(F64)),
        ("String", &ValueTy::String),
        ("Bytes", &ValueTy::Bytes),
    ];
    static ref REDUCED_PRIMITIVE_TYPES: Vec<(&'static str, &'static ValueTy)> = vec![
        ("Bool", &ValueTy::Bool),
        ("Int64", &ValueTy::Int(I64)),
        ("UInt64", &ValueTy::UInt(U64)),
        ("Float64", &ValueTy::Float(F64)),
        ("String", &ValueTy::String),
        ("Bytes", &ValueTy::Bytes),
    ];
    static ref PRIMITIVE_TYPES_ALIASES: Vec<(&'static str, &'static ValueTy)> =
        vec![("Int", &ValueTy::Int(I64)), ("UInt", &ValueTy::UInt(U64)), ("Float", &ValueTy::Float(F64)),];
}

impl ValueTy {
    pub(crate) fn primitive_types(config: TypeConfig) -> Vec<(&'static str, &'static ValueTy)> {
        let mut types = vec![];
        if config.use_64bit_only {
            types.extend_from_slice(&REDUCED_PRIMITIVE_TYPES)
        } else {
            types.extend_from_slice(&PRIMITIVE_TYPES)
        }
        if config.type_aliases {
            types.extend_from_slice(&PRIMITIVE_TYPES_ALIASES)
        }
        types
    }

    pub(crate) fn satisfies(&self, constraint: &TypeConstraint) -> bool {
        use self::TypeConstraint::*;
        use self::ValueTy::*;
        match constraint {
            Unconstrained => true,
            Comparable | Equatable => self.is_primitive(),
            Numeric => self.satisfies(&Integer) || self.satisfies(&FloatingPoint),
            FloatingPoint => match self {
                Float(_) => true,
                _ => false,
            },
            Integer => self.satisfies(&SignedInteger) || self.satisfies(&UnsignedInteger),
            SignedInteger => match self {
                Int(_) => true,
                _ => false,
            },
            UnsignedInteger => match self {
                UInt(_) => true,
                _ => false,
            },
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_error(&self) -> bool {
        use self::ValueTy::*;
        match self {
            Error => true,
            Tuple(args) => args.iter().any(|el| el.is_error()),
            Option(ty) => ty.is_error(),
            _ => false,
        }
    }

    /**
    Returns whether this type is a primitive type.

    Primitive types are the built-in types except compound types such as tuples.
    */
    pub fn is_primitive(&self) -> bool {
        use self::ValueTy::*;
        match self {
            Bool | Int(_) | UInt(_) | Float(_) | String | Bytes => true,
            _ => false,
        }
    }

    /// Replaces parameters by the given list
    pub(crate) fn replace_params(&self, infer_vars: &[ValueVar]) -> ValueTy {
        match self {
            &ValueTy::Param(id, _) => ValueTy::Infer(infer_vars[id as usize]),
            ValueTy::Option(t) => ValueTy::Option(t.replace_params(infer_vars).into()),
            ValueTy::Infer(_) | ValueTy::Constr(_) => self.clone(),
            _ if self.is_primitive() => self.clone(),
            _ => unreachable!("replace_param for {}", self),
        }
    }

    /// Replaces parameters by the given list
    pub(crate) fn replace_params_with_ty(&self, generics: &[ValueTy]) -> ValueTy {
        match self {
            &ValueTy::Param(id, _) => generics[id as usize].clone(),
            ValueTy::Option(t) => ValueTy::Option(t.replace_params_with_ty(generics).into()),
            ValueTy::Infer(_) | ValueTy::Constr(_) => self.clone(),
            _ if self.is_primitive() => self.clone(),
            _ => unreachable!("replace_param for {}", self),
        }
    }

    /// Replaces constraints by default values
    pub(crate) fn replace_constr(&self) -> ValueTy {
        match &self {
            ValueTy::Tuple(t) => ValueTy::Tuple(t.iter().map(|el| el.replace_constr()).collect()),
            ValueTy::Option(ty) => ValueTy::Option(ty.replace_constr().into()),
            ValueTy::Constr(c) => match c.has_default() {
                Some(d) => d,
                None => ValueTy::Error,
            },
            ValueTy::Param(_, _) => self.clone(),
            _ if self.is_primitive() => self.clone(),
            _ => unreachable!("cannot replace_constr for {}", self),
        }
    }
}

impl std::fmt::Display for ValueTy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ValueTy::Bool => write!(f, "Bool"),
            ValueTy::Int(I8) => write!(f, "Int8"),
            ValueTy::Int(I16) => write!(f, "Int16"),
            ValueTy::Int(I32) => write!(f, "Int32"),
            ValueTy::Int(I64) => write!(f, "Int64"),
            ValueTy::UInt(U8) => write!(f, "UInt8"),
            ValueTy::UInt(U16) => write!(f, "UInt16"),
            ValueTy::UInt(U32) => write!(f, "UInt32"),
            ValueTy::UInt(U64) => write!(f, "UInt64"),
            ValueTy::Float(F16) => write!(f, "Float16"),
            ValueTy::Float(F32) => write!(f, "Float32"),
            ValueTy::Float(F64) => write!(f, "Float64"),
            ValueTy::String => write!(f, "String"),
            ValueTy::Bytes => write!(f, "Bytes"),
            ValueTy::Option(ty) => write!(f, "{}?", ty),
            ValueTy::Tuple(inner) => {
                let joined: Vec<String> = inner.iter().map(|e| format!("{}", e)).collect();
                write!(f, "({})", joined.join(", "))
            }
            ValueTy::Infer(id) => write!(f, "?{}", id),
            ValueTy::Constr(constr) => write!(f, "{{{}}}", constr),
            ValueTy::Param(_, name) => write!(f, "{}", name),
            ValueTy::Error => write!(f, "Error"),
        }
    }
}

impl std::fmt::Display for StreamTy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            StreamTy::Event(activation) => write!(f, "EventStream({})", activation),
            StreamTy::RealTime(freq) => write!(f, "PeriodicStream({})", freq),
            StreamTy::Infer(vars) => write!(f, "InferedStream({:?})", vars),
        }
    }
}

impl std::fmt::Display for Activation<NodeId> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use crate::ast::print::write_delim_list;
        match self {
            Activation::Conjunction(con) => write_delim_list(f, &con, "(", ")", " && "),
            Activation::Disjunction(dis) => write_delim_list(f, &dis, "(", ")", " | "),
            Activation::Stream(v) => write!(f, "{}", v),
            Activation::True => write!(f, "true"),
        }
    }
}

/**
Type constraint used during type checking and type inference.

**FOR INERNAL USE**
*/
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Copy)]
pub enum TypeConstraint {
    /**
    The type must be a signed integer type.
    */
    SignedInteger,
    /**
    The type must be an unsigned integer type.
    */
    UnsignedInteger,
    /**
    The type must be a floating-point type.
    */
    FloatingPoint,
    /// signed + unsigned integer
    Integer,
    /// integer + floating point
    Numeric,
    /// Types that can be compared, i.e., implement `==`
    Equatable,
    /// Types that can be ordered, i.e., implement `<`, `>`,
    Comparable,
    /**
    The type is unconstrained.
    */
    Unconstrained,
}

impl TypeConstraint {
    pub(crate) fn has_default(&self) -> Option<ValueTy> {
        use self::TypeConstraint::*;
        match self {
            Integer | SignedInteger | Numeric => Some(ValueTy::Int(I64)),
            UnsignedInteger => Some(ValueTy::UInt(U64)),
            FloatingPoint => Some(ValueTy::Float(F64)),
            _ => None,
        }
    }

    pub(crate) fn conjunction<'a>(&'a self, other: &'a TypeConstraint) -> Option<&'a TypeConstraint> {
        use self::TypeConstraint::*;
        if self > other {
            return other.conjunction(self);
        }
        if self == other {
            return Some(self);
        }
        assert!(self < other);
        match other {
            Unconstrained | Comparable | Equatable | Numeric => Some(self),
            Integer => match self {
                FloatingPoint => None,
                _ => Some(self),
            },
            FloatingPoint | SignedInteger | UnsignedInteger => None,
        }
    }
}

impl std::fmt::Display for TypeConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::TypeConstraint::*;
        match self {
            SignedInteger => write!(f, "signed integer"),
            UnsignedInteger => write!(f, "unsigned integer"),
            Integer => write!(f, "integer"),
            FloatingPoint => write!(f, "floating point"),
            Numeric => write!(f, "numeric type"),
            Equatable => write!(f, "equatable type"),
            Comparable => write!(f, "comparable type"),
            Unconstrained => write!(f, "unconstrained type"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::traits::cast::FromPrimitive;

    #[test]
    fn test_freq_conjunction() {
        let a = Freq::new(UOM_Frequency::new::<hertz>(Rational::from_i64(6).unwrap()));
        let b = Freq::new(UOM_Frequency::new::<hertz>(Rational::from_i64(4).unwrap()));
        let c = Freq::new(UOM_Frequency::new::<hertz>(Rational::from_i64(2).unwrap()));
        assert_eq!(a.conjunction(&b), c)
    }
}
