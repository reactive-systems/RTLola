use super::window_aggregations::*;
use super::Value;
use crate::basics::Time;
use ordered_float::NotNan;
use rtlola_frontend::ir::{Type, WindowOperation as WinOp};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Add;
use std::time::Duration;

const SIZE: usize = 64;

/// Representation of sliding window aggregations:
/// The enum differentiates the aggregation functions and between different value types, dependent on the aggregation function.
/// # Example:
/// * The aggregation function 'count' is independent of the value type.
/// * The aggregation function 'min' depends on the value type, e.g., the minimum value of unsigned values is 0, whereas the minimum value for signed values is negative.
pub(crate) enum SlidingWindow {
    Count(WindowInstance<CountIV>),
    MinUnsigned(WindowInstance<MinIV<WindowSigned>>),
    MinSigned(WindowInstance<MinIV<WindowUnsigned>>),
    MinFloat(WindowInstance<MinIV<WindowFloat>>),
    MaxUnsigned(WindowInstance<MaxIV<WindowSigned>>),
    MaxSigned(WindowInstance<MaxIV<WindowUnsigned>>),
    MaxFloat(WindowInstance<MaxIV<WindowFloat>>),
    SumUnsigned(WindowInstance<SumIV<WindowUnsigned>>),
    SumSigned(WindowInstance<SumIV<WindowSigned>>),
    SumFloat(WindowInstance<SumIV<WindowFloat>>),
    SumBool(WindowInstance<SumIV<WindowBool>>),
    AvgUnsigned(WindowInstance<AvgIV<WindowUnsigned>>),
    AvgSigned(WindowInstance<AvgIV<WindowSigned>>),
    AvgFloat(WindowInstance<AvgIV<WindowFloat>>),
    Integral(WindowInstance<IntegralIV>),
    Conjunction(WindowInstance<ConjIV>),
    Disjunction(WindowInstance<DisjIV>),
}

impl SlidingWindow {
    /// Returns a sliding window instance, from:
    /// # Arguments:
    /// * 'dur'- the duration of the window
    /// * 'wait' - the boolean flag to decide if the window returns its value after the complete duration has passed
    /// * 'op' - the type of the aggregation function
    /// * 'ts' - the starting time of the window
    /// * 'ty' - the value type of the aggregated stream
    pub(crate) fn new(dur: Duration, wait: bool, op: WinOp, ts: Time, ty: &Type) -> SlidingWindow {
        match (op, ty) {
            (WinOp::Count, _) => SlidingWindow::Count(WindowInstance::new(dur, wait, ts)),
            (WinOp::Min, Type::UInt(_)) => SlidingWindow::MinUnsigned(WindowInstance::new(dur, wait, ts)),
            (WinOp::Min, Type::Int(_)) => SlidingWindow::MinSigned(WindowInstance::new(dur, wait, ts)),
            (WinOp::Min, Type::Float(_)) => SlidingWindow::MinFloat(WindowInstance::new(dur, wait, ts)),
            (WinOp::Max, Type::UInt(_)) => SlidingWindow::MaxUnsigned(WindowInstance::new(dur, wait, ts)),
            (WinOp::Max, Type::Int(_)) => SlidingWindow::MaxSigned(WindowInstance::new(dur, wait, ts)),
            (WinOp::Max, Type::Float(_)) => SlidingWindow::MaxFloat(WindowInstance::new(dur, wait, ts)),
            (WinOp::Sum, Type::UInt(_)) => SlidingWindow::SumUnsigned(WindowInstance::new(dur, wait, ts)),
            (WinOp::Sum, Type::Int(_)) => SlidingWindow::SumSigned(WindowInstance::new(dur, wait, ts)),
            (WinOp::Sum, Type::Float(_)) => SlidingWindow::SumFloat(WindowInstance::new(dur, wait, ts)),
            (WinOp::Sum, Type::Bool) => SlidingWindow::SumBool(WindowInstance::new(dur, wait, ts)),
            (WinOp::Average, Type::UInt(_)) => SlidingWindow::AvgUnsigned(WindowInstance::new(dur, wait, ts)),
            (WinOp::Average, Type::Int(_)) => SlidingWindow::AvgSigned(WindowInstance::new(dur, wait, ts)),
            (WinOp::Average, Type::Float(_)) => SlidingWindow::AvgFloat(WindowInstance::new(dur, wait, ts)),
            (WinOp::Integral, _) => SlidingWindow::Integral(WindowInstance::new(dur, wait, ts)),
            (WinOp::Conjunction, Type::Bool) => SlidingWindow::Conjunction(WindowInstance::new(dur, wait, ts)),
            (WinOp::Disjunction, Type::Bool) => SlidingWindow::Disjunction(WindowInstance::new(dur, wait, ts)),
            (_, Type::Option(t)) => SlidingWindow::new(dur, wait, op, ts, t),
            _ => unimplemented!(),
        }
    }

    /// Updates the buckets of a sliding window instance with the given timestamp:
    /// # Arguments:
    /// * 'ts' - the current timestamp of the monitor
    pub(crate) fn update(&mut self, ts: Time) {
        match self {
            SlidingWindow::Count(wi) => wi.update_buckets(ts),
            SlidingWindow::MinUnsigned(wi) => wi.update_buckets(ts),
            SlidingWindow::MinSigned(wi) => wi.update_buckets(ts),
            SlidingWindow::MinFloat(wi) => wi.update_buckets(ts),
            SlidingWindow::MaxUnsigned(wi) => wi.update_buckets(ts),
            SlidingWindow::MaxSigned(wi) => wi.update_buckets(ts),
            SlidingWindow::MaxFloat(wi) => wi.update_buckets(ts),
            SlidingWindow::SumUnsigned(wi) => wi.update_buckets(ts),
            SlidingWindow::SumSigned(wi) => wi.update_buckets(ts),
            SlidingWindow::SumFloat(wi) => wi.update_buckets(ts),
            SlidingWindow::SumBool(wi) => wi.update_buckets(ts),
            SlidingWindow::Conjunction(wi) => wi.update_buckets(ts),
            SlidingWindow::Disjunction(wi) => wi.update_buckets(ts),
            SlidingWindow::AvgUnsigned(wi) => wi.update_buckets(ts),
            SlidingWindow::AvgSigned(wi) => wi.update_buckets(ts),
            SlidingWindow::AvgFloat(wi) => wi.update_buckets(ts),
            SlidingWindow::Integral(wi) => wi.update_buckets(ts),
        }
    }

    /// Computes the current value of a sliding window instance with the given timestamp:
    /// # Arguments:
    /// * 'ts' - the current timestamp of the monitor
    /// Note: You should always call `SlidingWindow::update` before calling `SlidingWindow::get_value()`!
    pub(crate) fn get_value(&self, ts: Time) -> Value {
        match self {
            SlidingWindow::Count(wi) => wi.get_value(ts),
            SlidingWindow::MinUnsigned(wi) => wi.get_value(ts),
            SlidingWindow::MinSigned(wi) => wi.get_value(ts),
            SlidingWindow::MinFloat(wi) => wi.get_value(ts),
            SlidingWindow::MaxUnsigned(wi) => wi.get_value(ts),
            SlidingWindow::MaxSigned(wi) => wi.get_value(ts),
            SlidingWindow::MaxFloat(wi) => wi.get_value(ts),
            SlidingWindow::SumUnsigned(wi) => wi.get_value(ts),
            SlidingWindow::SumSigned(wi) => wi.get_value(ts),
            SlidingWindow::SumFloat(wi) => wi.get_value(ts),
            SlidingWindow::SumBool(wi) => wi.get_value(ts),
            SlidingWindow::Conjunction(wi) => wi.get_value(ts),
            SlidingWindow::Disjunction(wi) => wi.get_value(ts),
            SlidingWindow::AvgUnsigned(wi) => wi.get_value(ts),
            SlidingWindow::AvgSigned(wi) => wi.get_value(ts),
            SlidingWindow::AvgFloat(wi) => wi.get_value(ts),
            SlidingWindow::Integral(wi) => wi.get_value(ts),
        }
    }

    /// Updates the value of the first bucket of a sliding window instance with the current value of the accessed stream:
    /// # Arguments:
    /// * 'v' - the current value of the accessed stream
    /// * 'ts' - the current timestamp of the monitor
    pub(crate) fn accept_value(&mut self, v: Value, ts: Time) {
        match self {
            SlidingWindow::Count(wi) => wi.accept_value(v, ts),
            SlidingWindow::MinUnsigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::MinSigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::MinFloat(wi) => wi.accept_value(v, ts),
            SlidingWindow::MaxUnsigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::MaxFloat(wi) => wi.accept_value(v, ts),
            SlidingWindow::MaxSigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::SumUnsigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::SumSigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::SumFloat(wi) => wi.accept_value(v, ts),
            SlidingWindow::SumBool(wi) => wi.accept_value(v, ts),
            SlidingWindow::Conjunction(wi) => wi.accept_value(v, ts),
            SlidingWindow::Disjunction(wi) => wi.accept_value(v, ts),
            SlidingWindow::AvgUnsigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::AvgSigned(wi) => wi.accept_value(v, ts),
            SlidingWindow::AvgFloat(wi) => wi.accept_value(v, ts),
            SlidingWindow::Integral(wi) => wi.accept_value(v, ts),
        }
    }
}

// TODO: Consider using None rather than Default.
/// Trait to summarize common logic for the different window aggregations, e.g., returning a default value for an empty bucket
pub(crate) trait WindowIV:
    Clone + Add<Output = Self> + From<(Value, Time)> + Sized + Debug + Into<Value>
{
    fn default(ts: Time) -> Self;
}

/// Struct to summarize common logic for the different window aggregations, e.g. iterating over the buckets to compute the result of an aggregation
pub(crate) struct WindowInstance<IV: WindowIV> {
    buckets: VecDeque<IV>,
    time_per_bucket: Duration,
    start_time: Time,
    last_bucket_ix: BIx,
    wait: bool,
    wait_duration: Duration,
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
    fn new(dur: Duration, wait: bool, ts: Time) -> WindowInstance<IV> {
        let time_per_bucket = dur / (SIZE as u32);
        let buckets = VecDeque::from(vec![IV::default(ts); SIZE]);
        // last bucket_ix is 1, so we consider all buckets, i.e. from 1 to end and from start to 0,
        // as in use. Whenever we progress by n buckets, we invalidate the pseudo-used ones.
        // This is safe since the value within is the neutral element of the operation.
        WindowInstance {
            buckets,
            time_per_bucket,
            start_time: ts,
            last_bucket_ix: BIx::new(0, 0),
            wait,
            wait_duration: dur,
        }
    }

    /// You should always call `WindowInstance::update_buckets` before calling `WindowInstance::get_value()`!
    fn get_value(&self, ts: Time) -> Value {
        // Reversal is essential for non-commutative operations.
        if self.wait && ts < self.wait_duration {
            return Value::None;
        }
        self.buckets.iter().rev().fold(IV::default(ts), |acc, e| acc + e.clone()).into()
    }

    fn accept_value(&mut self, v: Value, ts: Time) {
        self.update_buckets(ts);
        let b = self.buckets.get_mut(0).expect("Bug!");
        *b = b.clone() + (v, ts).into(); // TODO: Require add_assign rather than add.
    }

    fn update_buckets(&mut self, ts: Time) {
        let curr = self.get_current_bucket(ts);
        let last = self.last_bucket_ix;

        let diff = curr.buckets_since(last, self.buckets.len());
        self.invalidate_n(diff, ts);
        self.last_bucket_ix = curr;
    }

    fn invalidate_n(&mut self, n: usize, ts: Time) {
        for _ in 0..n {
            self.buckets.pop_back();
            self.buckets.push_front(IV::default(ts));
        }
    }

    fn get_current_bucket(&self, ts: Time) -> BIx {
        // let overall_ix = ts.duration_since(self.start_time).div_duration(self.time_per_bucket);
        assert!(ts >= self.start_time, "Time does not behave monotonically!");
        let overall_ix = Self::quickfix_duration_div(ts - self.start_time, self.time_per_bucket);
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
            Value::Unsigned(u) => Value::Signed(u as i64),
            _ => unreachable!("Type error."),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WindowBool {}
impl WindowGeneric for WindowBool {
    fn from_value(v: Value) -> Value {
        match v {
            Value::Bool(b) if b => Value::Unsigned(1),
            Value::Bool(_) => Value::Unsigned(0),
            _ => unreachable!("Type error."),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WindowUnsigned {}
impl WindowGeneric for WindowUnsigned {
    fn from_value(v: Value) -> Value {
        match v {
            Value::Unsigned(_) => v,
            _ => unreachable!("Type error."),
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
            _ => unreachable!("Type error."),
        };
        Value::Float(NotNan::new(f).unwrap())
    }
}
