use crate::ty::Ty;
use std::convert::TryInto;

pub struct Lattice<E: LatticeElement + Eq + UpperBounded> { }

enum LatticeType<E: UpperBounded + Concretizable> {
    /// Represents an irrecoverable error in the type inference.
    Error,
    /// Represents any abstract type.  This type can be concretized via the `Concretizable` trait
    /// and create out of thin air via the `UpperBounded` trait.
    Abstract(E),
    Concrete(E),
}

impl<E> LatticeType<E> {
    fn is_error(&self) -> bool {
        self == LatticeType::Error
    }
}

impl<E: UpperBounded + Concretizable> LowerBounded for LatticeType<E> {
    fn bot() -> Self {
        LatticeType::Error
    }
}

impl<E: UpperBounded + Concretizable> UpperBounded for LatticeType<E> {
    fn top() -> Self {
        LatticeType::Abstract(<E as UpperBounded>::top())
    }
}

impl<E: UpperBounded + Concretizable> From<E> for LatticeType<E> {
    fn from(e: E) -> Self {
        LatticeType::Abstract(e)
    }
}

impl<E: UpperBounded + Concretizable> TryInto<E> for LatticeType<E> {
    type Error = ();

    fn try_into(self) -> Result<E, Self::Error> {
        match self {
            LatticeType::Error => Err(()),
            LatticeType::Abstract(a) => a.concretize().try_into(),
            LatticeType::Concrete(c) => Ok(c),
        }
    }
}

pub trait LowerBounded {
    fn bot() -> Self;
}

pub trait UpperBounded {
    fn top() -> Self;
}

pub trait Concretizable {
    fn concretize(&self) -> Self;
}

impl<E: LatticeElement + LowerBounded + UpperBounded> Lattice<E> {
    pub fn meet(left: LatticeType<E>, right: LatticeType<E>) -> LatticeType<E> {
        left.meet(right).unwrap_or(E::bot())
    }
    pub fn more_concrete(left: LatticeType<E>, right: LatticeType<E>) -> bool {
        Self::meet(left, right).is_error()
    }
    fn is_error(elem: LatticeType<E>) -> bool {
        elem.is_error()
    }
}

pub trait LatticeElement: Eq + LowerBounded + Sized {
    /// Unifies the element with another one if possible.
    /// Returns `None` if they are incompatible, otherwise `Some(e)` where `e` is the least concrete
    /// element that is more concrete than either operand.
    fn meet(&self, other: &Self) -> Option<Self>;
}

impl LatticeElement for Ty {
    fn meet(&self, other: &Self) -> Option<Self> {
        unimplemented!()
    }

    fn is_concrete(&self) -> bool {
        unimplemented!()
    }

    fn is_unconstrained(&self) -> bool {
        unimplemented!()
    }
}
