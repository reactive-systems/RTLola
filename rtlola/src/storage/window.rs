use super::Value;
use lola_parser::WindowOperation;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Add;
use std::time::{Duration, Instant};

const SIZE: usize = 4;

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

pub(crate) trait WindowIV:
    Clone + From<Value> + Add<Output = Self> + Into<Value> + Sized + Default + Debug
{
}

pub(crate) struct WindowInstance<IV: WindowIV> {
    buckets: VecDeque<IV>,
    time_per_bucket: Duration,
    start_time: Instant,
    last_bucket_ix: BIx,
}

#[derive(Clone, Copy, Debug)]
struct BIx {
    period: usize,
    ix: usize,
}

impl BIx {
    fn new(period: usize, ix: usize) -> BIx {
        BIx { period, ix }
    }

    fn buckets_since(self, other: BIx, num_buckets: usize) -> usize {
        match self.period.cmp(&other.period) {
            Ordering::Less => panic!("`other` bucket is more recent than `self`."),
            Ordering::Greater => {
                let period_diff = self.period - other.period;
                match self.ix.cmp(&other.ix) {
                    Ordering::Equal => period_diff * num_buckets,
                    Ordering::Less => {
                        let actual_p_diff = period_diff - 1;
                        let ix_diff = num_buckets - other.ix + self.ix;
                        actual_p_diff * num_buckets + ix_diff
                    }
                    Ordering::Greater => period_diff * num_buckets + (self.ix - other.ix),
                }
            }
            Ordering::Equal => match self.ix.cmp(&other.ix) {
                Ordering::Equal => 0,
                Ordering::Less => panic!("`other` bucket is more recent than `self`."),
                Ordering::Greater => self.ix - other.ix,
            },
        }
    }
}

impl<IV: WindowIV> WindowInstance<IV> {
    fn new(dur: Duration, ts: Instant) -> WindowInstance<IV> {
        let time_per_bucket = dur / (SIZE as u32);
        let buckets = VecDeque::from(vec![IV::default(); SIZE]);
        // last bucket_ix is 1, so we consider all buckets, i.e. from 1 to end and from start to 0,
        // as in use. Whenever we progress by n buckets, we invalidate the pseudo-used ones.
        // This is safe since the value within is the neutral element of the operation.
        WindowInstance { buckets, time_per_bucket, start_time: ts, last_bucket_ix: BIx::new(0, 0) }
    }

    fn get_value(&mut self, ts: Instant) -> Value {
        self.update_buckets(ts);
        self.buckets.iter().fold(IV::default(), |acc, e| acc + e.clone()).into()
    }

    fn accept_value(&mut self, v: Value, ts: Instant) {
        self.update_buckets(ts);
        let b = self.buckets.get_mut(0).expect("Bug!");
        *b = b.clone() + v.into(); // TODO: Require add_assign rather than add.
    }

    fn update_buckets(&mut self, ts: Instant) {
        let curr = self.get_current_bucket(ts);
        let last = self.last_bucket_ix;

        if curr.period > 0 {
            // no invalidating or updating in the first cycle.
            let diff = curr.buckets_since(last, self.buckets.len());
            self.invalidate_n(diff);
            self.last_bucket_ix = curr;
        }
    }

    fn invalidate_n(&mut self, n: usize) {
        for _ in 0..n {
            self.buckets.pop_back();
            self.buckets.push_front(IV::default());
        }
    }

    fn get_current_bucket(&self, ts: Instant) -> BIx {
        //        let overall_ix = ts.duration_since(self.start_time).div_duration(self.time_per_bucket);
        let overall_ix = Self::quickfix_duration_div(ts.duration_since(self.start_time), self.time_per_bucket);
        let overall_ix = overall_ix.floor() as usize;
        let period = overall_ix / self.buckets.len();
        let ix = overall_ix % self.buckets.len();
        BIx { period, ix }
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

#[derive(Clone, Debug)]
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
