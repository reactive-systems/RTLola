use super::Value;
use lola_parser::{WindowOperation as WinOp, Type};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Add;
use std::time::{Duration, Instant};
use super::window_aggregations::*;

const SIZE: usize = 4;

pub(crate) enum SlidingWindow {
    SumUnsigned(WindowInstance<SumIV<WindowUnsigned>>),
    SumSigned(WindowInstance<SumIV<WindowSigned>>),
}

impl SlidingWindow {
    pub(crate) fn new(dur: Duration, op: WinOp, ts: Instant, ty: &Type) -> SlidingWindow {
        match (op, ty) {
            (WinOp::Sum, Type::Int(_)) => SlidingWindow::SumSigned(WindowInstance::new(dur, ts)),
            (WinOp::Sum, Type::UInt(_)) => SlidingWindow::SumUnsigned(WindowInstance::new(dur, ts)),
            (_, Type::Option(t)) => SlidingWindow::new(dur, op, ts, t),
            _ => unimplemented!(),
        }
    }

    pub(crate) fn get_value(&mut self, ts: Instant) -> Value {
        match self {
            SlidingWindow::SumSigned(wi) => wi.get_value(ts),
            SlidingWindow::SumUnsigned(wi) => wi.get_value(ts),
        }
    }

    pub(crate) fn accept_value(&mut self, v: Value, ts: Instant) {
        match self {
            SlidingWindow::SumSigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::SumUnsigned(wi) => wi.accept_value(v, ts),
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

pub(crate) trait WindowGeneric: Debug + Clone {
    fn from(v: Value) -> Value;
}

#[derive(Debug, Clone)]
pub(crate) struct WindowSigned {}
impl WindowGeneric for WindowSigned {
    fn from(v: Value) -> Value {
        match v {
            Value::Signed(_) => v,
            Value::Unsigned(u) => Value::Signed(u as i128),
            _ => panic!("Type error."),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WindowUnsigned {}
impl WindowGeneric for WindowUnsigned {
    fn from(v: Value) -> Value {
        match v {
            Value::Unsigned(_) => v,
            _ => panic!("Type error."),
        }
    }
}