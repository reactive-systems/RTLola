use crate::basics::Time;
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

impl<G: WindowGeneric> WindowIV for SumIV<G> {
    fn default(time: Time) -> SumIV<G> {
        let v = (G::from_value(Value::Unsigned(0)), time);
        Self::from(v)
    }
}

impl<G: WindowGeneric> Into<Value> for SumIV<G> {
    fn into(self) -> Value {
        self.v
    }
}

impl<G: WindowGeneric> Add for SumIV<G> {
    type Output = SumIV<G>;
    fn add(self, other: SumIV<G>) -> SumIV<G> {
        (self.v + other.v, Time::default()).into() // Timestamp will be discarded, anyway.
    }
}

impl<G: WindowGeneric> From<(Value, Time)> for SumIV<G> {
    fn from(v: (Value, Time)) -> SumIV<G> {
        SumIV { v: G::from_value(v.0), _marker: PhantomData }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConjIV {
    res: bool,
}

impl WindowIV for ConjIV {
    fn default(_ts: Time) -> Self {
        true.into()
    }
}

impl Into<Value> for ConjIV {
    fn into(self) -> Value {
        Value::Bool(self.res)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Add for ConjIV {
    type Output = ConjIV;
    fn add(self, other: ConjIV) -> ConjIV {
        (self.res && other.res).into()
    }
}

impl From<(Value, Time)> for ConjIV {
    fn from(v: (Value, Time)) -> ConjIV {
        match v.0 {
            Value::Bool(b) => b.into(),
            _ => unreachable!("Type error."),
        }
    }
}

impl From<bool> for ConjIV {
    fn from(v: bool) -> ConjIV {
        ConjIV { res: v }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DisjIV {
    res: bool,
}

impl WindowIV for DisjIV {
    fn default(_ts: Time) -> Self {
        false.into()
    }
}

impl Into<Value> for DisjIV {
    fn into(self) -> Value {
        Value::Bool(self.res)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Add for DisjIV {
    type Output = DisjIV;
    fn add(self, other: DisjIV) -> DisjIV {
        (self.res || other.res).into()
    }
}

impl From<(Value, Time)> for DisjIV {
    fn from(v: (Value, Time)) -> DisjIV {
        match v.0 {
            Value::Bool(b) => b.into(),
            _ => unreachable!("Type error."),
        }
    }
}

impl From<bool> for DisjIV {
    fn from(v: bool) -> DisjIV {
        DisjIV { res: v }
    }
}

// TODO: Generic for floats...
#[derive(Clone, Debug)]
pub(crate) struct AvgIV<G: WindowGeneric> {
    sum: Value,
    num: u64,
    _marker: PhantomData<G>,
}

impl<G: WindowGeneric> WindowIV for AvgIV<G> {
    fn default(_time: Time) -> AvgIV<G> {
        AvgIV { sum: Value::None, num: 0, _marker: PhantomData }
    }
}

impl<G: WindowGeneric> Into<Value> for AvgIV<G> {
    fn into(self) -> Value {
        match self.sum {
            Value::None => Value::None,
            Value::Unsigned(u) => Value::Unsigned(u / self.num),
            Value::Signed(u) => Value::Signed(u / self.num as i64),
            Value::Float(u) => Value::Float(u / self.num as f64),
            _ => unreachable!("Type error."),
        }
    }
}

impl<G: WindowGeneric> Add for AvgIV<G> {
    type Output = AvgIV<G>;
    fn add(self, other: AvgIV<G>) -> AvgIV<G> {
        match (&self.sum, &other.sum) {
            (Value::None, Value::None) => Self::default(Time::default()),
            (_, Value::None) => self,
            (Value::None, _) => other,
            _ => {
                let sum = self.sum + other.sum;
                let num = self.num + other.num;
                AvgIV { sum, num, _marker: PhantomData }
            }
        }
    }
}

impl<G: WindowGeneric> From<(Value, Time)> for AvgIV<G> {
    fn from(v: (Value, Time)) -> AvgIV<G> {
        AvgIV { sum: G::from_value(v.0), num: 1u64, _marker: PhantomData }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct IntegralIV {
    volume: f64,
    end_value: f64,
    end_time: Time,
    start_value: f64,
    start_time: Time,
    valid: bool,
}

impl WindowIV for IntegralIV {
    fn default(time: Time) -> IntegralIV {
        IntegralIV { volume: 0f64, end_value: 0f64, end_time: time, start_value: 0f64, start_time: time, valid: false }
    }
}

impl Into<Value> for IntegralIV {
    fn into(self) -> Value {
        Value::new_float(self.volume)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Add for IntegralIV {
    type Output = IntegralIV;
    fn add(self, other: IntegralIV) -> IntegralIV {
        match (self.valid, other.valid) {
            (false, false) => return self,
            (false, true) => return other,
            (true, false) => return self,
            (true, true) => {}
        }

        let start_volume = self.volume + other.volume;

        assert!(other.start_time >= self.end_time, "Time does not behave monotonically!");
        let time_diff = other.start_time - self.end_time;
        let time_diff_secs = (time_diff.as_secs() as f64) + (f64::from(time_diff.subsec_nanos())) / (100_000_000f64);
        let time_diff = time_diff_secs;
        let value_sum = other.start_value + self.end_value;

        let additional_volume = value_sum * time_diff / 2f64;

        let volume = start_volume + additional_volume;
        let end_value = other.end_value;
        let end_time = other.end_time;
        let start_value = self.start_value;
        let start_time = self.start_time;

        IntegralIV { volume, end_value, end_time, start_value, start_time, valid: true }
    }
}

impl From<(Value, Time)> for IntegralIV {
    fn from(v: (Value, Time)) -> IntegralIV {
        let f = match v.0 {
            Value::Signed(i) => (i as f64),
            Value::Unsigned(u) => (u as f64),
            Value::Float(f) => (f.into()),
            _ => unreachable!("Type error."),
        };
        IntegralIV { volume: 0f64, end_value: f, end_time: v.1, start_value: f, start_time: v.1, valid: true }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CountIV(u64);

impl WindowIV for CountIV {
    fn default(_time: Time) -> CountIV {
        CountIV(0)
    }
}

impl Into<Value> for CountIV {
    fn into(self) -> Value {
        Value::Unsigned(self.0)
    }
}

impl Add for CountIV {
    type Output = CountIV;
    fn add(self, other: CountIV) -> CountIV {
        CountIV(self.0 + other.0)
    }
}

impl From<(Value, Time)> for CountIV {
    fn from(_v: (Value, Time)) -> CountIV {
        CountIV(1)
    }
}

//////////////////// MIN/MAX ////////////////////

#[derive(Clone, Debug)]
pub(crate) struct MaxIV<G: WindowGeneric> {
    max: Value,
    _marker: PhantomData<G>,
}

impl<G: WindowGeneric> WindowIV for MaxIV<G> {
    fn default(_ts: Time) -> MaxIV<G> {
        Self::from((Value::None, Time::default()))
    }
}

impl<G: WindowGeneric> Into<Value> for MaxIV<G> {
    fn into(self) -> Value {
        self.max
    }
}

impl<G: WindowGeneric> Add for MaxIV<G> {
    type Output = MaxIV<G>;
    fn add(self, other: MaxIV<G>) -> MaxIV<G> {
        let max = match (self.max, other.max) {
            (Value::None, Value::None) => Value::None,
            (Value::None, rhs) => rhs,
            (lhs, Value::None) => lhs,
            (Value::Unsigned(lhs), Value::Unsigned(rhs)) => Value::Unsigned(lhs.max(rhs)),
            (Value::Signed(lhs), Value::Signed(rhs)) => Value::Signed(lhs.max(rhs)),
            (Value::Float(lhs), Value::Float(rhs)) => Value::Float(lhs.max(rhs)),
            _ => unreachable!("Mixed types in sliding window aggregation."),
        };
        MaxIV { max, _marker: PhantomData }
    }
}

impl<G: WindowGeneric> From<(Value, Time)> for MaxIV<G> {
    fn from(v: (Value, Time)) -> MaxIV<G> {
        MaxIV { max: v.0, _marker: PhantomData }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MinIV<G: WindowGeneric> {
    min: Value,
    _marker: PhantomData<G>,
}

impl<G: WindowGeneric> WindowIV for MinIV<G> {
    fn default(_ts: Time) -> MinIV<G> {
        Self::from((Value::None, Time::default()))
    }
}

impl<G: WindowGeneric> Into<Value> for MinIV<G> {
    fn into(self) -> Value {
        self.min
    }
}

impl<G: WindowGeneric> Add for MinIV<G> {
    type Output = MinIV<G>;
    fn add(self, other: MinIV<G>) -> MinIV<G> {
        let min = match (self.min, other.min) {
            (Value::None, Value::None) => Value::None,
            (Value::None, rhs) => rhs,
            (lhs, Value::None) => lhs,
            (Value::Unsigned(lhs), Value::Unsigned(rhs)) => Value::Unsigned(lhs.min(rhs)),
            (Value::Signed(lhs), Value::Signed(rhs)) => Value::Signed(lhs.min(rhs)),
            (Value::Float(lhs), Value::Float(rhs)) => Value::Float(lhs.min(rhs)),
            _ => unreachable!("Mixed types in sliding window aggregation."),
        };
        MinIV { min, _marker: PhantomData }
    }
}

impl<G: WindowGeneric> From<(Value, Time)> for MinIV<G> {
    fn from(v: (Value, Time)) -> MinIV<G> {
        MinIV { min: v.0, _marker: PhantomData }
    }
}
