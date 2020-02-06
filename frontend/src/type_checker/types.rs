//! Every type needs to implement `type_checker::AbstractType` and thus provide an unconstrained value.  As such, every
//! conjunctive type (aka `struct`) can be constructed arbitrarily provided its constituents are valid
//! `type_check::AbstractType`s.  Every disjunctive type (aka `enum`) needs to provide a variant representing the
//! unconstrained version of itself where the disjunction is not resolved, yet.

use uom::si::rational64::Frequency as UOM_Frequency;
use uom::si::frequency::hertz;
use lazy_static::lazy_static;
use num::{CheckedDiv, Integer};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub(in crate::type_checker) struct Key(pub(in crate::type_checker) u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnificationError {
    IncompatibleValueTypes(ValueTy, ValueTy),
    MixedEventPeriodic(EventTy, PeriodicTy),
    IncompatibleFrequencies(Freq, Freq),
    Other(String),
}

/// The type of an expression consists of both, a value type (`Bool`, `String`, etc.) and
/// an evaluation type.
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub struct Ty {
    pub(in crate::type_checker) value: ValueTy,
    pub(in crate::type_checker) eval: EvaluationTy,
}

/// Contains information on when a stream is created evaluated.
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub struct EvaluationTy {
    pub(in crate::type_checker) pacing: PacingTy,
    pub(in crate::type_checker) filter: ExpressionTy,
    pub(in crate::type_checker) spawn: ExpressionTy,
}

/// The expression represents a data-dependent condition on the current monitor state.
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub enum ExpressionTy {
    Infer,
    Node(NodeId), // TODO: This is probably a bad idea.
}

/// The pacing type dictates when a stream is evaluated.
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub enum PacingTy {
    /// An event stream is extended when its activation condition is satisfied.
    Event(ActivationCondition<NodeId>),
    /// A real-time stream is extended periodically.
    Periodic(Freq),
    /// An undetermined type that can be unified into either of the other options.
    Infer,
}


/// The `value` type, storing information about the stored values (`Bool`, `UInt8`, etc.)
/// # Inference
/// Generally, a of `Infer` with `T` results in `T` and a meet of `Error` with `T` results in `Error`.  Combining `T`
/// with `T` yields `T`.  All types are annotated with their inference capability _apart from the rules outlined above_.
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Hash)]
pub enum ValueTy {
    /// The abstract version of a bool.
    /// # Inference
    /// Results in an error.
    /// Is `ValueTy::Equatable`.
    Bool,
    /// An abstract integer type.
    /// # Inference
    /// Widening is permitted.
    /// Is `ValueTy::Integer`.
    Int(IntTy),
    /// An abstract unsigned integer type.
    /// # Inference
    /// Widening is permitted.
    /// Is `ValueTy::Integer`.
    UInt(UIntTy),
    /// A floating point number.
    /// # Inference
    /// Widening is permitted.
    /// Is `ValueTy::Numeric`.
    Float(FloatTy),
    /// A string.  Is equatable.
    /// # Inference
    /// Results in an error.
    /// Is `ValueTy::Equatable`.
    String,
    /// Conjunction of several types.
    /// # Inference
    /// The union of two non-identical tuples is a point-wise unification of its constituents.
    /// Is `ValueTy::Equatable` if its constituents are equatable.
    Tuple(Vec<ValueTy>),
    /// An optional type indicating that a value may or may not be present.
    /// # Inference
    /// Unification is possible if the inner types can be unified.
    /// Is `ValueTy::Equatable` if its inner type is `ValueTy::Equatable`.
    Option(Box<ValueTy>),
    /// An unconstrained type.
    /// # Inference
    /// Unification is always possible and will yield the other operand.
    Infer,
    /// Represents either a signed or an unsigned integer of unconstrained size.
    /// # Inference
    /// Unification with either `ValueTy::UInt` or `ValueTy::Int` will result in the other operand.
    /// Is `ValueTy::Equatable` and `ValueTy::Numeric`.
    Integer,
    /// Represents either a `ValueTy::Integer` or a floating point number of unconstrained size.
    /// # Inference
    /// Is `ValueTy::Comparable`.
    Numeric,
    /// Represents any type that can be equated with `=`.
    /// # Inference
    /// See annotations of each type.
    Equatable,
    /// Types that can be ordered, i.e., implement `<`, `>`,
    /// # Inference
    /// See annotations of each type.
    Comparable,
//    /// A reference to a generic parameter in a function declaration, e.g. `T` in `a<T>(x:T) -> T`
//    Generic(u8, String),
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
}
use self::IntTy::*;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum UIntTy {
    U8,
    U16,
    U32,
    U64,
}
use self::UIntTy::*;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum FloatTy {
    F32,
    F64,
}
use self::FloatTy::*;
use crate::ast::Expression;
use crate::parse::NodeId;
use uom::num_rational::{Rational, Ratio};
use crate::TypeConfig;
use std::convert::TryFrom;
use std::collections::HashSet;
use std::hash::Hash;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct Freq(pub(crate) UOM_Frequency);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum ActivationCondition<Var> {
    Conjunction(Vec<Self>),
    Disjunction(Vec<Self>),
    Stream(Var),
    True,
}

impl std::fmt::Display for Freq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.0.clone().into_format_args(uom::si::frequency::hertz, uom::fmt::DisplayStyle::Abbreviation)
        )
    }
}

impl Freq {

    pub(crate) fn is_multiple_of(&self, other: &Freq) -> Result<bool, String> {
        let lhs = self.0.get::<hertz>();
        let rhs = other.0.get::<hertz>();
        if lhs < rhs {
            return Ok(false);
        }
        match lhs.checked_div(&rhs) {
            Some(q) => Ok(q.is_integer()),
            None => Err(format!("division of frequencies `{:?}`/`{:?}` failed", &self.0, &other.0)),
        }
    }

    pub(crate) fn conjunction(&self, other: &Freq) -> Freq {
        let numer_left = self.0.get::<hertz>().numer().clone();
        let numer_right = other.0.get::<hertz>().numer().clone();
        let denom_left = self.0.get::<hertz>().denom().clone();
        let denom_right = other.0.get::<hertz>().denom().clone();
        // gcd(self, other) = gcd(numer_left, numer_right) / lcm(denom_left, denom_right)
        // only works if rational numbers are reduced, which ist the default for `Rational`
        let r1: i64 = i64::try_from(numer_left.gcd(&numer_right)).unwrap();
        let r2: i64 = i64::try_from(denom_left.gcd(&denom_right)).unwrap();
        let r : Ratio<i64> = Ratio::new(r1, r2);
        Freq(UOM_Frequency::new::<hertz>(r))
    }
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

    pub(crate) fn is_primitive(&self) -> bool {
        use self::ValueTy::*;
        match self {
            Bool | Int(_) | UInt(_) | Float(_) | String => true,
            Infer | Tuple(_) | Option(_) | Numeric | Comparable | Equatable | Integer => false,
        }
    }

//    /// Replaces parameters by the given list
//    pub(crate) fn replace_params(&self, infer_vars: &[TypeKey]) -> ValueTy {
//        match self {
//            &ValueTy::Param(id, _) => ValueTy::Infer(infer_vars[id as usize]),
//            ValueTy::Option(t) => ValueTy::Option(t.replace_params(infer_vars).into()),
//            ValueTy::Infer(_) | ValueTy::Constr(_) => self.clone(),
//            _ if self.is_primitive() => self.clone(),
//            _ => unreachable!("replace_param for {}", self),
//        }
//    }
//
//    /// Replaces parameters by the given list
//    pub(crate) fn replace_params_with_ty(&self, generics: &[ValueTy]) -> ValueTy {
//        match self {
//            &ValueTy::Param(id, _) => generics[id as usize].clone(),
//            ValueTy::Option(t) => ValueTy::Option(t.replace_params_with_ty(generics).into()),
//            ValueTy::Infer(_) | ValueTy::Constr(_) => self.clone(),
//            _ if self.is_primitive() => self.clone(),
//            _ => unreachable!("replace_param for {}", self),
//        }
//    }
}

impl<Var: Hash> ActivationCondition<Var> {
    fn normalize(&mut self) -> bool {
        fn norm(ac: &mut ActivationCondition<Var>, stream_vars: &mut ) -> {
            match ac {
                Activation::Conjunction(args) | Activation::Disjunction(args) => {
                    args.iter_mut().all(|arg| arg.normalize(stream_vars))
                }
                Activation::Stream(var) => {
                    if stream_vars.contains(var) {
                        return true;
                    }
                    stream_vars.insert(*var);
                    *ac = match &self.stream_ty[var] {
                        // make stream types default to event stream with empty conjunction, i.e., true
                        StreamTy::Infer(_) => unreachable!(),
                        StreamTy::Event(Activation::Stream(v)) if *var == *v => {
                            return true;
                        }
                        StreamTy::Event(ac) => ac.clone(),
                        StreamTy::RealTime(_) => {
                            self.handler.error("real-time streams cannot be used in activation conditions");
                            return false;
                        }
                    };
                    self.normalize_activation_condition(ac, stream_vars)
                }
                Activation::True => true,
            }
        }
        norm(self, self.collect_vars());
    }

    fn collect_vars(&self) -> HashSet<Var> {
        use ActivationCondition::*;
        match self {
            Conjunction(args) | Disjunction(args) => args.iter().flat_map(ActivationCondition::collect_vars).collect(),
            Stream(v) => {let mut s = HashSet::new(); s.insert(v); s}
            True => HashSet::new(),
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
        ("Float32", &ValueTy::Float(F32)),
        ("Float64", &ValueTy::Float(F64)),
        ("String", &ValueTy::String),
    ];
    static ref REDUCED_PRIMITIVE_TYPES: Vec<(&'static str, &'static ValueTy)> = vec![
        ("Bool", &ValueTy::Bool),
        ("Int64", &ValueTy::Int(I64)),
        ("UInt64", &ValueTy::UInt(U64)),
        ("Float64", &ValueTy::Float(F64)),
        ("String", &ValueTy::String),
    ];
    static ref PRIMITIVE_TYPES_ALIASES: Vec<(&'static str, &'static ValueTy)> =
        vec![("Int", &ValueTy::Int(I64)), ("UInt", &ValueTy::UInt(U64)), ("Float", &ValueTy::Float(F64)),];
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
            ValueTy::Float(F32) => write!(f, "Float32"),
            ValueTy::Float(F64) => write!(f, "Float64"),
            ValueTy::String => write!(f, "String"),
            ValueTy::Option(ty) => write!(f, "{}?", ty),
            ValueTy::Tuple(inner) => {
                let joined: Vec<String> = inner.iter().map(|e| format!("{}", e)).collect();
                write!(f, "({})", joined.join(", "))
            }
            ValueTy::Infer => write!(f, "⊤"),
            ValueTy::Integer => write!(f, "{{Integer}}"),
            ValueTy::Numeric => write!(f, "{{Numeric}}"),
            ValueTy::Equatable => write!(f, "{{Equatable}}"),
            ValueTy::Comparable => write!(f, "{{Comparable}}"),
        }
    }
}