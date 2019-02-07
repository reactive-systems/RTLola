use crate::storage::{
    window::{WindowGeneric, WindowIV},
    Value,
};
use std::marker::PhantomData;
use std::ops::Add;

#[derive(Clone, Debug)]
pub(crate) struct SumIV<G: WindowGeneric> {
    v: Value,
    _marker: PhantomData<G>,
}

impl<G: WindowGeneric> WindowIV for SumIV<G> {}

impl<G: WindowGeneric> Add for SumIV<G> {
    type Output = SumIV<G>;
    fn add(self, other: SumIV<G>) -> SumIV<G> {
        (self.v + other.v).into()
    }
}

impl<G: WindowGeneric> Default for SumIV<G> {
    fn default() -> SumIV<G> {
        Self::from(G::from(Value::Unsigned(0)))
    }
}

impl<G: WindowGeneric> From<Value> for SumIV<G> {
    fn from(v: Value) -> SumIV<G> {
        SumIV { v: G::from(v), _marker: PhantomData }
    }
}

impl<G: WindowGeneric> Into<Value> for SumIV<G> {
    fn into(self) -> Value {
        self.v.clone()
    }
}
