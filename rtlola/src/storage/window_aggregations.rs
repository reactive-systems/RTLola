use crate::storage::{
    window::{WindowGeneric, WindowIV},
    Value,
};
use std::marker::PhantomData;
use std::ops::Add;
use std::time::SystemTime;

#[derive(Clone, Debug)]
pub(crate) struct SumIV<G: WindowGeneric> {
    v: Value,
    _marker: PhantomData<G>,
}

impl<G: WindowGeneric> WindowIV for SumIV<G> {
    fn default(time: SystemTime) -> SumIV<G> {
        let v = (G::from_value(Value::Unsigned(0)), time);
        Self::from(v)
    }
}

impl<G: WindowGeneric> Into<Value> for SumIV<G> {
    fn into(self) -> Value {
        self.v.clone()
    }
}

impl<G: WindowGeneric> Add for SumIV<G> {
    type Output = SumIV<G>;
    fn add(self, other: SumIV<G>) -> SumIV<G> {
        (self.v + other.v, SystemTime::now()).into() // Timestamp will be discarded, anyway.
    }
}

impl<G: WindowGeneric> From<(Value, SystemTime)> for SumIV<G> {
    fn from(v: (Value, SystemTime)) -> SumIV<G> {
        SumIV { v: G::from_value(v.0), _marker: PhantomData }
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
    fn default(time: SystemTime) -> AvgIV<G> {
        let v = (G::from_value(Value::Unsigned(0)), time);
        Self::from(v)
    }
}

impl<G: WindowGeneric> Default for AvgIV<G> {
    fn default() -> AvgIV<G> {
        let with_mock_ts = (G::from_value(Value::Unsigned(0)), SystemTime::now());
        Self::from(with_mock_ts)
    }
}

impl<G: WindowGeneric> Into<Value> for AvgIV<G> {
    fn into(self) -> Value {
        match self.sum {
            Value::Unsigned(u) => Value::Unsigned(u / self.num as u128),
            Value::Signed(u) => Value::Signed(u / self.num as i128),
            Value::Float(u) => Value::Float(u / self.num as f64),
            _ => panic!("Type error."),
        }
    }
}

impl<G: WindowGeneric> Add for AvgIV<G> {
    type Output = AvgIV<G>;
    fn add(self, other: AvgIV<G>) -> AvgIV<G> {
        let sum = self.sum + other.sum;
        let num = self.num + other.num;
        AvgIV { sum, num, _marker: PhantomData }
    }
}

impl<G: WindowGeneric> From<(Value, SystemTime)> for AvgIV<G> {
    fn from(v: (Value, SystemTime)) -> AvgIV<G> {
        AvgIV { sum: G::from_value(v.0), num: 1u64, _marker: PhantomData }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct IntegralIV {
    volume: f64,
    end_value: Value,
    end_time: SystemTime,
    start_value: Value,
    start_time: SystemTime,
    valid: bool,
}

impl WindowIV for IntegralIV {
    fn default(time: SystemTime) -> IntegralIV {
        IntegralIV {
            volume: 0f64,
            end_value: Value::new_float(0f64),
            end_time: time,
            start_value: Value::new_float(0f64),
            start_time: time,
            valid: false,
        }
    }
}

impl Into<Value> for IntegralIV {
    fn into(self) -> Value {
        Value::new_float(self.volume)
    }
}

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

        let ts2 = other.start_time;
        let ts1 = self.end_time;
        let time_diff = ts2.duration_since(ts1).expect("Time does not behave monotonically!");
        let time_diff_secs = (time_diff.as_secs() as f64) + (time_diff.subsec_nanos() as f64) / (100_000_000f64);
        let time_diff = Value::new_float(time_diff_secs);
        let value_sum = other.start_value.clone() + self.end_value.clone();

        let additional_volume_v = value_sum * time_diff / Value::new_float(2f64);
        let additional_volume: f64 = match additional_volume_v {
            Value::Float(f) => f.into(),
            _ => panic!("Bug."),
        };

        let volume = start_volume + additional_volume;
        let end_value = other.end_value.clone();
        let end_time = other.end_time;
        let start_value = self.start_value.clone();
        let start_time = self.start_time;

        IntegralIV { volume, end_value, end_time, start_value, start_time, valid: true }
    }
}

impl From<(Value, SystemTime)> for IntegralIV {
    fn from(v: (Value, SystemTime)) -> IntegralIV {
        IntegralIV {
            volume: 0f64,
            end_value: v.0.clone(),
            end_time: v.1,
            start_value: v.0,
            start_time: v.1,
            valid: true,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CountIV(u64);

impl WindowIV for CountIV {
    fn default(time: SystemTime) -> CountIV {
        CountIV(0)
    }
}

impl Into<Value> for CountIV {
    fn into(self) -> Value {
        Value::Unsigned(self.0 as u128)
    }
}

impl Add for CountIV {
    type Output = CountIV;
    fn add(self, other: CountIV) -> CountIV {
        CountIV(self.0 + other.0)
    }
}

impl From<(Value, SystemTime)> for CountIV {
    fn from(v: (Value, SystemTime)) -> CountIV {
        CountIV(1)
    }
}
