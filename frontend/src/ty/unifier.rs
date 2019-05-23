use super::{Activation, StreamTy, TypeConstraint, ValueTy};
use ena::unify::{
    EqUnifyValue, InPlace, InPlaceUnificationTable, Snapshot, UnificationStore, UnificationTable, UnifyKey, UnifyValue,
};
use log::{debug, trace};

/// Main data structure for the type unfication.
/// Implemented using a union-find data structure where the keys (`ValueVar`)
/// represent type variables. Those keys have associated values (`ValueVarVal`)
/// that represent whether the type variable is unbounded (`ValueVarVal::Unkown`)
/// or bounded (`ValueVarVal::Know(ty)` where `ty` has type `Ty`).
pub(crate) struct ValueUnifier<T: UnifiableTy> {
    /// union-find data structure representing the current state of the unification
    table: InPlaceUnificationTable<T::V>,
}

impl<T: UnifiableTy> ValueUnifier<T> {
    pub(crate) fn new() -> ValueUnifier<T> {
        ValueUnifier { table: UnificationTable::new() }
    }

    /// Returns type where every inference variable is substituted.
    /// If an infer variable remains, it is replaced by `ValueTy::Error`.
    pub(crate) fn get_normalized_type(&mut self, var: T::V) -> Option<T> {
        self.get_type(var).map(|t| t.normalize_ty(self))
    }
}

pub trait Unifier {
    type Var: UnifyKey;
    type Ty;
    type Store: UnificationStore;

    fn new_var(&mut self) -> Self::Var;
    fn get_type(&mut self, var: Self::Var) -> Option<Self::Ty>;
    fn unify_var_var(&mut self, left: Self::Var, right: Self::Var) -> InferResult;
    fn unify_var_ty(&mut self, var: Self::Var, ty: Self::Ty) -> InferResult;
    fn vars_equal(&mut self, l: Self::Var, r: Self::Var) -> bool;
    fn snapshot(&mut self) -> Snapshot<Self::Store>;
    fn commit(&mut self, ss: Snapshot<Self::Store>);
    fn rollback_to(&mut self, ss: Snapshot<Self::Store>);
}

impl<T: UnifiableTy> Unifier for ValueUnifier<T> {
    type Var = T::V;
    type Ty = T;
    type Store = InPlace<Self::Var>;

    fn new_var(&mut self) -> Self::Var {
        self.table.new_key(ValueVarVal::Unknown)
    }

    /// Unifies two variables.
    /// Cannot fail if one of them is unbounded.
    /// If both are bounded, we try to unify their types (recursively over the `Ty` type).
    /// If this fails as well, we try to coerce them, i.e., transform one type into the other.
    fn unify_var_var(&mut self, left: Self::Var, right: Self::Var) -> InferResult {
        debug!("unify var var {} {}", left, right);
        match (self.table.probe_value(left), self.table.probe_value(right)) {
            (ValueVarVal::Known(ty_l), ValueVarVal::Known(ty_r)) => {
                // if both variables have values, we try to unify them recursively
                if let Some(ty) = ty_l.equal_to(self, &ty_r) {
                    // proceed with unification
                    self.table
                        .unify_var_value(left, ValueVarVal::Concretize(ty.clone()))
                        .expect("overwrite cannot fail");
                    self.table.unify_var_value(right, ValueVarVal::Concretize(ty)).expect("overwrite cannot fail");
                } else if ty_l.coerces_with(self, &ty_r) {
                    return Ok(());
                } else {
                    return Err(ty_l.conflicts_with(ty_r));
                }
            }
            (ValueVarVal::Unknown, ValueVarVal::Known(ty)) => {
                if ty.contains_var(self, left) {
                    return Err(InferError::CyclicDependency);
                }
            }
            (ValueVarVal::Known(ty), ValueVarVal::Unknown) => {
                if ty.contains_var(self, right) {
                    return Err(InferError::CyclicDependency);
                }
            }
            _ => {}
        }

        self.table.unify_var_var(left, right)
    }

    /// Unifies a variable with a type.
    /// Cannot fail if the variable is unbounded.
    /// Prevents infinite recursion by checking if `var` appears in `ty`.
    /// Uses the same strategy to merge types as `unify_var_var` (in case `var` is bounded).
    fn unify_var_ty(&mut self, var: Self::Var, ty: Self::Ty) -> InferResult {
        debug!("unify var ty {} {}", var, ty);
        if let Some(other) = ty.is_inferred() {
            return self.unify_var_var(var, other);
        }
        assert!(ty.is_inferred().is_none(), "internal error: entered unreachable code");
        if ty.contains_var(self, var) {
            return Err(InferError::CyclicDependency);
        }
        if let ValueVarVal::Known(val) = self.table.probe_value(var) {
            if let Some(ty) = val.equal_to(self, &ty) {
                self.table.unify_var_value(var, ValueVarVal::Concretize(ty))
            } else if val.coerces_with(self, &ty) {
                return Ok(());
            } else {
                Err(val.conflicts_with(ty))
            }
        } else {
            self.table.unify_var_value(var, ValueVarVal::Known(ty))
        }
    }

    /// Returns current value of inference variable if it exists, `None` otherwise.
    fn get_type(&mut self, var: Self::Var) -> Option<Self::Ty> {
        if let ValueVarVal::Known(ty) = self.table.probe_value(var) {
            Some(ty)
        } else {
            None
        }
    }

    fn vars_equal(&mut self, l: Self::Var, r: Self::Var) -> bool {
        self.table.unioned(l, r)
    }

    fn snapshot(&mut self) -> Snapshot<Self::Store> {
        self.table.snapshot()
    }
    fn commit(&mut self, ss: Snapshot<Self::Store>) {
        self.table.commit(ss);
    }
    fn rollback_to(&mut self, ss: Snapshot<Self::Store>) {
        self.table.rollback_to(ss);
    }
}

pub trait UnifiableTy: std::marker::Sized + std::fmt::Display + Clone + std::cmp::PartialEq + std::fmt::Debug {
    type V: UnifyKey<Value = ValueVarVal<Self>> + std::fmt::Display;
    fn normalize_ty<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U) -> Self;
    fn coerces_with<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, right: &Self) -> bool;
    fn equal_to<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, right: &Self) -> Option<Self>;
    fn contains_var<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, var: Self::V) -> bool;
    fn is_inferred(&self) -> Option<Self::V>;
    fn conflicts_with(self, other: Self) -> InferError;
}

impl UnifiableTy for ValueTy {
    type V = ValueVar;

    /// Removes occurrences of inference variables by the inferred type.
    fn normalize_ty<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U) -> ValueTy {
        match self {
            ValueTy::Infer(var) => match unifier.get_type(*var) {
                None => self.clone(),
                Some(other_ty) => other_ty.normalize_ty(unifier),
            },
            ValueTy::Tuple(t) => ValueTy::Tuple(t.iter().map(|el| el.normalize_ty(unifier)).collect()),
            ValueTy::Option(ty) => ValueTy::Option(Box::new(ty.normalize_ty(unifier))),
            _ if self.is_primitive() => self.clone(),
            ValueTy::Constr(_) => self.clone(),
            ValueTy::Param(_, _) => self.clone(),
            _ => unreachable!("cannot normalize {}", self),
        }
    }

    /// Checks recursively if the `right` type can be transformed to match `self`.
    fn coerces_with<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, right: &ValueTy) -> bool {
        debug!("coerce {} {}", self, right);

        // Rule 1: Any type `T` can be coerced into `Option<T>`
        if let ValueTy::Option(ty) = self {
            // Take snapshot before checking equality to potentially rollback the side effects.
            let ss = unifier.snapshot();
            if ty.equal_to(unifier, right).is_some() {
                unifier.commit(ss);
                return true;
            } else {
                unifier.rollback_to(ss);
            }
        }

        // Rule 2: Bit-width increase is allowed
        match right {
            ValueTy::Int(lower) => match self {
                ValueTy::Int(upper) => lower <= upper,
                _ => false,
            },
            ValueTy::UInt(lower) => match self {
                ValueTy::UInt(upper) => lower <= upper,
                _ => false,
            },
            ValueTy::Float(lower) => match self {
                ValueTy::Float(upper) => lower <= upper,
                _ => false,
            },
            _ => false,
        }
    }

    /// Checks recursively if types are equal. Tries to unify type parameters if possible.
    /// Returns the unified, i.e., more concrete type if possible.
    fn equal_to<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, right: &ValueTy) -> Option<ValueTy> {
        trace!("comp {} {}", self, right);
        match (self, right) {
            (&ValueTy::Infer(l), &ValueTy::Infer(r)) => {
                if unifier.vars_equal(l, r) {
                    Some(ValueTy::Infer(l))
                } else {
                    // try to unify values
                    if unifier.unify_var_var(l, r).is_ok() {
                        Some(ValueTy::Infer(l))
                    } else {
                        None
                    }
                }
            }
            (&ValueTy::Infer(var), ty) | (ty, &ValueTy::Infer(var)) => {
                // try to unify
                if unifier.unify_var_ty(var, ty.clone()).is_ok() {
                    Some(ValueTy::Infer(var))
                } else {
                    None
                }
            }
            (ValueTy::Constr(constr_l), ValueTy::Constr(constr_r)) => {
                constr_l.conjunction(constr_r).map(|c| ValueTy::Constr(c.clone()))
            }
            (ValueTy::Constr(constr), other) => {
                if other.satisfies(constr) {
                    Some(other.clone())
                } else {
                    None
                }
            }
            (other, ValueTy::Constr(constr)) => {
                if other.satisfies(constr) {
                    Some(other.clone())
                } else {
                    None
                }
            }
            (ValueTy::Option(l), ValueTy::Option(r)) => l.equal_to(unifier, r).map(|ty| ValueTy::Option(ty.into())),
            (ValueTy::Tuple(l), ValueTy::Tuple(r)) => {
                if l.len() != r.len() {
                    return None;
                }
                let params: Vec<ValueTy> = l.iter().zip(r).flat_map(|(l, r)| l.equal_to(unifier, r)).collect();
                if params.len() != l.len() {
                    return None;
                }
                Some(ValueTy::Tuple(params))
            }
            (l, r) => {
                if l == r {
                    Some(l.clone())
                } else {
                    None
                }
            }
        }
    }

    /// Checks if `var` occurs in `self`
    fn contains_var<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, var: ValueVar) -> bool {
        trace!("check occurrence {} {}", var, self);
        match self {
            ValueTy::Infer(t) => unifier.vars_equal(var, *t),
            ValueTy::Tuple(t) => t.iter().any(|e| e.contains_var(unifier, var)),
            ValueTy::Option(t) => t.contains_var(unifier, var),
            _ => false,
        }
    }

    fn is_inferred(&self) -> Option<ValueVar> {
        if let ValueTy::Infer(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    fn conflicts_with(self, other: Self) -> InferError {
        InferError::ValueTypeMismatch(self, other)
    }
}

/// Representation of key for unification of `ValueTy`
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub struct ValueVar(u32);

// used in UnifcationTable
impl UnifyKey for ValueVar {
    type Value = ValueVarVal<ValueTy>;
    fn index(&self) -> u32 {
        self.0
    }
    fn from_index(u: u32) -> ValueVar {
        ValueVar(u)
    }
    fn tag() -> &'static str {
        "ValueVar"
    }
}

impl UnifiableTy for StreamTy {
    type V = StreamVar;

    fn normalize_ty<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U) -> Self {
        self.concretize(unifier).expect("unification table should be always consitent")
    }

    fn coerces_with<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, right: &Self) -> bool {
        debug!("coerce {} {}", self, right);

        // RealTime<freq_self> -> RealTime<freq_right> if freq_left is multiple of freq_right
        match (&self, &right) {
            (StreamTy::RealTime(target), StreamTy::RealTime(other)) => {
                // coercion is only valid if `other` is a multiple of `target`,
                // for example, `target = 3Hz` and `other = 12Hz`
                other.is_multiple_of(target)
            }
            (StreamTy::Event(target), StreamTy::Event(other)) => {
                // coercion is only valid if the implication `target -> other` is valid,
                // for example, `target = a & b` and `other = b` is valid while
                //              `target = a | b` and `other = b` is invalid.
                target.implies_valid(other, unifier)
            }
            _ => false,
        }
    }

    fn equal_to<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, right: &Self) -> Option<Self> {
        trace!("comp {} {}", self, right);
        match (self, right) {
            (StreamTy::Infer(vars_l), StreamTy::Infer(vars_r)) => {
                let mut vars = vars_l.clone();
                vars.extend(vars_r);
                Some(StreamTy::Infer(vars))
            }
            (StreamTy::Infer(vars), ty) | (ty, StreamTy::Infer(vars)) => {
                let mut vars = vars.clone();
                let new_var = unifier.new_var();
                unifier.unify_var_ty(new_var, ty.clone()).expect("unification with fresh var cannot fail");
                vars.push(new_var);
                let result = StreamTy::Infer(vars);
                if result.concretize(unifier).is_none() {
                    None
                } else {
                    Some(result)
                }
            }
            (StreamTy::Event(cond_l), StreamTy::Event(cond_r)) => {
                if cond_l == cond_r {
                    Some(self.clone())
                } else {
                    None
                }
            }
            (StreamTy::RealTime(f_l), StreamTy::RealTime(f_r)) => {
                if f_l == f_r {
                    Some(self.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn contains_var<U: Unifier<Var = Self::V, Ty = Self>>(&self, unifier: &mut U, var: Self::V) -> bool {
        trace!("check occurrence {} {}", var, self);
        match self {
            StreamTy::Infer(vars) => vars.iter().any(|v| unifier.vars_equal(var, *v)),
            _ => false,
        }
    }

    fn is_inferred(&self) -> Option<Self::V> {
        None
    }

    fn conflicts_with(self, other: Self) -> InferError {
        let hint = match (&self, &other) {
            (StreamTy::Event(_), StreamTy::Event(_)) => Some(String::from("try using `.hold()` or `.get()`")),
            _ => None,
        };
        InferError::StreamTypeMismatch(self, other, hint)
    }
}

impl StreamTy {
    /// Tries to concretize a type, that is, resolve the inferred variables.
    /// Returns `None` if this is no longer possible, that is, some streams are event-based and some periodic.
    fn concretize<U: Unifier<Var = StreamVar, Ty = Self>>(&self, unifier: &mut U) -> Option<StreamTy> {
        match self {
            StreamTy::RealTime(_) | StreamTy::Event(_) => Some(self.clone()),
            StreamTy::Infer(vars) => {
                if vars.is_empty() {
                    return Some(StreamTy::Event(Activation::Conjunction(Vec::new())));
                }
                let mut stream_ty = None;
                for &var in vars {
                    stream_ty = match (stream_ty.as_mut(), unifier.get_type(var).as_ref()) {
                        (None, Some(ty)) => Some(ty.clone()),
                        (_, None) => {
                            // if the right hand side cannot be inferred, we fail
                            // TODO: check if this is correct
                            return None;
                        }
                        (Some(StreamTy::Event(l)), Some(StreamTy::Event(r))) => Some(StreamTy::Event(l.conjunction(r))),
                        (Some(StreamTy::RealTime(l)), Some(StreamTy::RealTime(r))) => {
                            Some(StreamTy::RealTime(l.conjunction(r)))
                        }
                        (Some(_), Some(_)) => {
                            // Incompatible types
                            return None;
                        }
                    }
                }
                stream_ty
            }
        }
    }
}

impl Activation {
    /// Checks whether `self -> other` is valid
    pub(crate) fn implies_valid<U: Unifier<Var = StreamVar, Ty = StreamTy>>(
        &self,
        other: &Activation,
        _unifier: &mut U,
    ) -> bool {
        if self == other {
            return true;
        }
        match (self, other) {
            (Activation::Conjunction(left), Activation::Conjunction(right)) => {
                right.iter().all(|cond| left.contains(cond))
            }
            (Activation::Conjunction(left), _) => left.contains(other),
            _ => {
                // there are possible many more cases that we want to look at in order to make analysis more precise
                false
            }
        }
    }

    pub(crate) fn conjunction(&self, other: &Activation) -> Activation {
        use Activation::*;
        match (self, other) {
            (Conjunction(c_l), Conjunction(c_r)) => {
                let mut con = c_l.clone();
                con.extend(c_r.iter().cloned());
                Activation::Conjunction(con)
            }
            (Conjunction(c), other) | (other, Conjunction(c)) => {
                let mut con = c.clone();
                con.push(other.clone());
                Activation::Conjunction(con)
            }
            (_, _) => Activation::Conjunction(vec![self.clone(), other.clone()]),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ValueVarVal<Ty: UnifiableTy> {
    Known(Ty),
    Unknown,
    /// Used to overwrite known values
    Concretize(Ty),
}

/// Implements how the types are merged during unification
impl<Ty: UnifiableTy> UnifyValue for ValueVarVal<Ty> {
    type Error = InferError;

    /// the idea of the unification is to merge two types and always taking the more concrete value
    fn unify_values(left: &Self, right: &Self) -> Result<Self, InferError> {
        use self::ValueVarVal::*;
        match (left, right) {
            (Known(_), Unknown) => Ok(left.clone()),
            (Unknown, Known(_)) => Ok(right.clone()),
            (Unknown, Unknown) => Ok(Unknown),
            (Known(_), Concretize(ty)) => Ok(Known(ty.clone())),
            (Known(l), Known(r)) => {
                assert!(l == r);
                Ok(left.clone())
            }
            _ => unreachable!("unify values {:?} {:?}", left, right),
        }
    }
}

impl EqUnifyValue for ValueTy {}

impl std::fmt::Display for ValueVar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Representation of key for unification of `StreamTy`
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub struct StreamVar(u32);

/// Only used in tests to be able to use equality over `StreamTy`.
/// Stream vars are only created within unifier otherwise.
#[cfg(test)]
impl StreamVar {
    pub(crate) fn new(index: u32) -> Self {
        Self(index)
    }
}

impl UnifyKey for StreamVar {
    type Value = ValueVarVal<StreamTy>;
    fn index(&self) -> u32 {
        self.0
    }
    fn from_index(u: u32) -> Self {
        StreamVar(u)
    }
    fn tag() -> &'static str {
        "StreamVar"
    }
}

impl EqUnifyValue for StreamTy {}

impl std::fmt::Display for StreamVar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone)]
pub(crate) enum StreamVarOrTy {
    Var(StreamVar),
    Ty(StreamTy),
}

type InferResult = Result<(), InferError>;

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum InferError {
    ValueTypeMismatch(ValueTy, ValueTy),
    // the last element can be an optional hint
    StreamTypeMismatch(StreamTy, StreamTy, Option<String>),
    ConflictingConstraint(TypeConstraint, TypeConstraint),
    CyclicDependency,
}

impl InferError {
    pub(crate) fn normalize_types(&mut self, unifier: &mut ValueUnifier<ValueTy>) {
        if let InferError::ValueTypeMismatch(ref mut expected, ref mut found) = self {
            std::mem::replace(expected, expected.normalize_ty(unifier));
            std::mem::replace(found, found.normalize_ty(unifier));
        }
    }
}
