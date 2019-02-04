use super::Value;
use lola_parser::WindowOperation;
use std::collections::VecDeque;
use std::ops::Add;
use std::time::{Duration, Instant};

const SIZE: usize = 256;

pub(crate) enum SlidingWindow {
    Sum(WindowInstance<SumIV>),
}

impl SlidingWindow {
    pub(crate) fn new(dur: Duration, op: WindowOperation, ts: Instant) -> SlidingWindow {
        match op {
            WindowOperation::Sum => SlidingWindow::Sum(WindowInstance::new(dur, ts)),
            _ => unimplemented!(),
        }
    }

    pub(crate) fn get_value(&mut self, ts: Instant) -> Value {
        match self {
            SlidingWindow::Sum(wi) => wi.get_value(ts),
        }
    }

    pub(crate) fn accept_value(&mut self, v: Value, ts: Instant) {
        match self {
            SlidingWindow::Sum(wi) => wi.accept_value(v, ts),
        }
    }
}

pub(crate) trait WindowIV: Clone + From<Value> + Add<Output = Self> + Into<Value> + Sized + Default {}

pub(crate) struct WindowInstance<IV: WindowIV> {
    buckets: VecDeque<IV>,
    time_per_bucket: Duration,
    start_time: Instant,
    last_bucket_ix: usize,
}

impl<IV: WindowIV> WindowInstance<IV> {
    fn new(dur: Duration, ts: Instant) -> WindowInstance<IV> {
        let time_per_bucket = dur / (SIZE as u32);
        let buckets = VecDeque::from(vec![IV::default(); SIZE]);
        WindowInstance { buckets, time_per_bucket, start_time: ts, last_bucket_ix: 0 }
    }

    fn get_value(&mut self, ts: Instant) -> Value {
        self.invalidate_buckets(ts);
        self.buckets.iter().fold(IV::default(), |acc, e| acc + e.clone()).into()
    }

    fn accept_value(&mut self, v: Value, ts: Instant) {
        self.invalidate_buckets(ts);
        let b = self.buckets.get_mut(self.get_current_bucket(ts)).expect("Bug!");
        *b = b.clone() + v.into();
    }

    fn invalidate_buckets(&mut self, ts: Instant) {
        let curr = self.get_current_bucket(ts);
        let last = self.last_bucket_ix;
        let diff = if last <= curr {
            curr - last
        } else {
            self.buckets.len() - last + curr // This order prevents overflowing.
        };
        self.invalidate_n(diff);
    }

    fn invalidate_n(&mut self, n: usize) {
        for _ in 0..n {
            self.buckets.pop_front();
            self.buckets.push_back(IV::default());
        }
    }

    fn get_current_bucket(&self, ts: Instant) -> usize {
        //        let overall_ix = ts.duration_since(self.start_time).div_duration(self.time_per_bucket);
        let overall_ix = Self::quickfix_duration_div(ts.duration_since(self.start_time), self.time_per_bucket);
        let overall_ix = overall_ix.floor() as usize;
        overall_ix % self.buckets.len()
    }

    fn quickfix_duration_div(a: Duration, b: Duration) -> f64 {
        let a_secs = a.as_secs();
        let a_nanos = a.subsec_nanos();
        let b_secs = b.as_secs();
        let b_nanos = b.subsec_nanos();
        let a = (a_secs as f64) + f64::from(a_nanos) / f64::from(1_000_000_000);
        let b = (b_secs as f64) + f64::from(b_nanos) / f64::from(1_000_000_000);
        a / b
    }
}

#[derive(Clone)]
pub(crate) struct SumIV(Value);

impl WindowIV for SumIV {}

impl Add for SumIV {
    type Output = SumIV;
    fn add(self, other: SumIV) -> SumIV {
        (self.0 + other.0).into()
    }
}

impl Default for SumIV {
    fn default() -> SumIV {
        SumIV(Value::Signed(0i128))
    }
}

impl From<Value> for SumIV {
    fn from(v: Value) -> SumIV {
        match v {
            Value::Unsigned(u) => SumIV(Value::Signed(u as i128)),
            Value::Signed(_) => SumIV(v),
            _ => panic!("Type error."),
        }
    }
}

impl Into<Value> for SumIV {
    fn into(self) -> Value {
        self.0.clone()
    }
}
