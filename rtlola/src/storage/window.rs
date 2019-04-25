use super::window_aggregations::*;
use super::Value;
use ordered_float::NotNan;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Add;
use std::time::{Duration, SystemTime};
use streamlab_frontend::ir::{Type, WindowOperation as WinOp};

const SIZE: usize = 64;

pub(crate) enum SlidingWindow {
    SumUnsigned(WindowInstance<SumIV<WindowUnsigned>>),
    SumSigned(WindowInstance<SumIV<WindowSigned>>),
    SumFloat(WindowInstance<SumIV<WindowFloat>>),
    AvgUnsigned(WindowInstance<AvgIV<WindowUnsigned>>),
    AvgSigned(WindowInstance<AvgIV<WindowSigned>>),
    AvgFloat(WindowInstance<AvgIV<WindowFloat>>),
    Integral(WindowInstance<IntegralIV>),
    Count(WindowInstance<CountIV>),
}

impl SlidingWindow {
    pub(crate) fn new(dur: Duration, op: WinOp, ts: SystemTime, ty: &Type) -> SlidingWindow {
        match (op, ty) {
            (WinOp::Sum, Type::UInt(_)) => SlidingWindow::SumUnsigned(WindowInstance::new(dur, ts)),
            (WinOp::Sum, Type::Int(_)) => SlidingWindow::SumSigned(WindowInstance::new(dur, ts)),
            (WinOp::Sum, Type::Float(_)) => SlidingWindow::SumFloat(WindowInstance::new(dur, ts)),
            (WinOp::Average, Type::UInt(_)) => SlidingWindow::AvgUnsigned(WindowInstance::new(dur, ts)),
            (WinOp::Average, Type::Int(_)) => SlidingWindow::AvgSigned(WindowInstance::new(dur, ts)),
            (WinOp::Average, Type::Float(_)) => SlidingWindow::AvgFloat(WindowInstance::new(dur, ts)),
            (WinOp::Integral, _) => SlidingWindow::Integral(WindowInstance::new(dur, ts)),
            (WinOp::Count, _) => SlidingWindow::Count(WindowInstance::new(dur, ts)),
            (_, Type::Option(t)) => SlidingWindow::new(dur, op, ts, t),
            _ => unimplemented!(),
        }
    }

    pub(crate) fn get_value(&mut self, ts: SystemTime) -> Value {
        match self {
            SlidingWindow::SumUnsigned(wi) => wi.get_value(ts),
            SlidingWindow::SumSigned(wi) => wi.get_value(ts),
            SlidingWindow::SumFloat(wi) => wi.get_value(ts),
            SlidingWindow::Integral(wi) => wi.get_value(ts),
            SlidingWindow::Count(wi) => wi.get_value(ts),
            SlidingWindow::AvgUnsigned(wi) => wi.get_value(ts),
            SlidingWindow::AvgSigned(wi) => wi.get_value(ts),
            SlidingWindow::AvgFloat(wi) => wi.get_value(ts),
        }
    }

    pub(crate) fn accept_value(&mut self, v: Value, ts: SystemTime) {
        match self {
            SlidingWindow::SumUnsigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::SumSigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::SumFloat(wi) => wi.accept_value(v, ts),
            SlidingWindow::Integral(wi) => wi.accept_value(v, ts),
            SlidingWindow::Count(wi) => wi.accept_value(v, ts),
            SlidingWindow::AvgUnsigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::AvgSigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::AvgFloat(wi) => wi.accept_value(v, ts),
        }
    }
}

// TODO: Consider using None rather than Default.
pub(crate) trait WindowIV:
    Clone + Add<Output = Self> + From<(Value, SystemTime)> + Sized + Debug + Into<Value>
{
    fn default(ts: SystemTime) -> Self;
}

pub(crate) struct WindowInstance<IV: WindowIV> {
    buckets: VecDeque<IV>,
    time_per_bucket: Duration,
    start_time: SystemTime,
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
    fn new(dur: Duration, ts: SystemTime) -> WindowInstance<IV> {
        let time_per_bucket = dur / (SIZE as u32);
        let buckets = VecDeque::from(vec![IV::default(ts); SIZE]);
        // last bucket_ix is 1, so we consider all buckets, i.e. from 1 to end and from start to 0,
        // as in use. Whenever we progress by n buckets, we invalidate the pseudo-used ones.
        // This is safe since the value within is the neutral element of the operation.
        WindowInstance { buckets, time_per_bucket, start_time: ts, last_bucket_ix: BIx::new(0, 0) }
    }

    fn get_value(&mut self, ts: SystemTime) -> Value {
        self.update_buckets(ts);
        // Reversal is essential for non-commutative operations.
        self.buckets.iter().rev().fold(IV::default(ts), |acc, e| acc + e.clone()).into()
    }

    fn accept_value(&mut self, v: Value, ts: SystemTime) {
        self.update_buckets(ts);
        let b = self.buckets.get_mut(0).expect("Bug!");
        *b = b.clone() + (v, ts).into(); // TODO: Require add_assign rather than add.
    }

    fn update_buckets(&mut self, ts: SystemTime) {
        let curr = self.get_current_bucket(ts);
        let last = self.last_bucket_ix;

        if curr.period > 0 {
            // no invalidating or updating in the first cycle.
            let diff = curr.buckets_since(last, self.buckets.len());
            self.invalidate_n(diff, ts);
            self.last_bucket_ix = curr;
        }
    }

    fn invalidate_n(&mut self, n: usize, ts: SystemTime) {
        for _ in 0..n {
            self.buckets.pop_back();
            self.buckets.push_front(IV::default(ts));
        }
    }

    fn get_current_bucket(&self, ts: SystemTime) -> BIx {
        //        let overall_ix = ts.duration_since(self.start_time).div_duration(self.time_per_bucket);
        let overall_ix = Self::quickfix_duration_div(
            ts.duration_since(self.start_time).expect("Time does not behave monotonically!"),
            self.time_per_bucket,
        );
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
    fn from_value(v: Value) -> Value;
}

#[derive(Debug, Clone)]
pub(crate) struct WindowSigned {}
impl WindowGeneric for WindowSigned {
    fn from_value(v: Value) -> Value {
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
    fn from_value(v: Value) -> Value {
        match v {
            Value::Unsigned(_) => v,
            _ => panic!("Type error."),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WindowFloat {}
impl WindowGeneric for WindowFloat {
    fn from_value(v: Value) -> Value {
        let f = match v {
            Value::Signed(i) => (i as f64),
            Value::Unsigned(u) => (u as f64),
            Value::Float(f) => (f.into()),
            _ => panic!("Type error."),
        };
        Value::Float(NotNan::new(f).unwrap())
    }
}
