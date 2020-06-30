use crate::ir::{OutputReference, RTLolaIR, Stream};
use std::time::Duration;

use num::rational::Rational64 as Rational;
use num::{One, ToPrimitive};
use uom::si::rational64::Time as UOM_Time;
use uom::si::time::{nanosecond, second};

/**
A deadline used inside the hyper-period of a `Schedule`.
*/
#[derive(Debug, Clone)]
pub struct Deadline {
    /**
    The pause between this evaluation and the next deadline.
    */
    pub pause: Duration,
    /**
    All output streams that need to be computed at this deadline.
    */
    pub due: Vec<OutputReference>,
}

/**
A schedule for the periodic streams.

The schedule is represented as a hyper-period, which describes the period of the combined period streams, and a list of deadlines within a hyper-period.
*/
#[derive(Debug, Clone)]
pub struct Schedule {
    #[rustfmt::skip]
    /**
    The `hyper_period` is the shortest duration, after which all periodic streams start a new period at the same time.

    It is therefore the least common multiple of all periods.  
    Example:  
    streams: (c @ 2000ms), (b @ 1000ms), (a @ 500ms)  
    Hyper period: 2000ms
    */
    pub hyper_period: Duration,
    #[rustfmt::skip]
    /**
    The deadlines inside the hyper-period.

    Deadlines represent points in time, at which at least one periodic stream gets updated.  
    Each deadline knows which streams need to be computed and how much time there is until the next deadline (or the end of the hyper-period).  
    The first deadline is at the start of the hyper-period.  
    There are only entries for deadlines before the completion of a hyper-period because at this point we start a new hyper-period with time 0ms.

    Example:  
    streams: (b @ 1500ms), (a @ 1000ms)  
    Hyper period: 3000ms  
    Deadlines ([a,b] --1000ms-->) ([a] --500ms-->) ([b] --500ms-->) ([a] --1000ms-->)
    */
    pub deadlines: Vec<Deadline>,
}

impl Schedule {
    pub(crate) fn from(ir: &RTLolaIR) -> Result<Schedule, String> {
        let periods: Vec<UOM_Time> = ir.time_driven.iter().map(|s| s.period).collect();
        let gcd = Self::find_extend_period(&periods);
        let hyper_period = Self::find_hyper_period(&periods);

        let extend_steps = Self::build_extend_steps(ir, gcd, hyper_period)?;
        let extend_steps = Self::apply_periodicity(&extend_steps);
        let mut deadlines = Self::condense_deadlines(gcd, extend_steps);
        Self::sort_deadlines(ir, &mut deadlines);

        let hyper_period = Duration::from_nanos(hyper_period.get::<nanosecond>().to_integer().to_u64().unwrap());
        Ok(Schedule { deadlines, hyper_period })
    }

    /// Determines the max amount of time the process can wait between successive checks for
    /// due deadlines without missing one.
    fn find_extend_period(rates: &[UOM_Time]) -> UOM_Time {
        assert!(!rates.is_empty());
        let rates: Vec<Rational> = rates.iter().map(|r| r.get::<nanosecond>()).collect();
        let gcd = math::rational_gcd_all(&rates);
        UOM_Time::new::<nanosecond>(gcd)
    }

    /// Determines the hyper period of the given `rates`.
    fn find_hyper_period(rates: &[UOM_Time]) -> UOM_Time {
        assert!(!rates.is_empty());
        let rates: Vec<Rational> = rates.iter().map(|r| r.get::<nanosecond>()).collect();
        let lcm = math::rational_lcm_all(&rates);
        let lcm = math::rational_lcm(lcm, Rational::one()); // needs to be multiple of 1 ns
        UOM_Time::new::<nanosecond>(lcm)
    }

    /// Takes a vec of gcd-sized intervals. In each interval, there are streams that need
    /// to be scheduled periodically at this point in time.
    /// Example:
    /// Hyper period: 2 seconds, gcd: 500ms, streams: (c @ .5Hz), (b @ 1Hz), (a @ 2Hz)
    /// Input:  `[[a] [b]   []  [c]]`
    /// Output: `[[a] [a,b] [a] [a,b,c]]`
    fn apply_periodicity(steps: &[Vec<OutputReference>]) -> Vec<Vec<OutputReference>> {
        // Whenever there are streams in a cell at index `i`,
        // add them to every cell with index k*i within bounds, where k > 1.
        // k = 0 would always schedule them initially, so this must be skipped.
        // TODO: Skip last half of the array.
        let mut res = vec![Vec::new(); steps.len()];
        for (ix, streams) in steps.iter().enumerate() {
            if !streams.is_empty() {
                let mut k = 1;
                while let Some(target) = res.get_mut(k * (ix + 1) - 1) {
                    target.extend(streams);
                    k += 1;
                }
            }
        }
        res
    }

    /// Build extend steps for each gcd-sized time interval up to the hyper period.
    /// Example:
    /// Hyper period: 2 seconds, gcd: 500ms, streams: (c @ .5Hz), (b @ 1Hz), (a @ 2Hz)
    /// Result: `[[a] [b] [] [c]]`
    /// Meaning: `a` starts being scheduled after one gcd, `b` after two gcds, `c` after 4 gcds.
    fn build_extend_steps(
        ir: &RTLolaIR,
        gcd: UOM_Time,
        hyper_period: UOM_Time,
    ) -> Result<Vec<Vec<OutputReference>>, String> {
        let num_steps = hyper_period.get::<second>() / gcd.get::<second>();
        assert!(num_steps.is_integer());
        let num_steps = num_steps.to_integer() as usize;
        if num_steps >= 10_000_000 {
            return Err("stream frequencies are too incompatible to generate schedule".to_string());
        }
        let mut extend_steps = vec![Vec::new(); num_steps];
        for s in ir.time_driven.iter() {
            let ix = s.period.get::<second>() / gcd.get::<second>();
            assert!(ix.is_integer());
            let ix = ix.to_integer() as usize;
            let ix = ix - 1;
            extend_steps[ix].push(s.reference.out_ix());
        }
        Ok(extend_steps)
    }

    fn condense_deadlines(gcd: UOM_Time, extend_steps: Vec<Vec<OutputReference>>) -> Vec<Deadline> {
        let mut empty_counter = 0;
        let mut deadlines: Vec<Deadline> = vec![];
        for step in extend_steps.iter() {
            if step.is_empty() {
                empty_counter += 1;
                continue;
            }
            let pause = gcd.get::<nanosecond>() * (empty_counter + 1);
            let pause = Duration::from_nanos(pause.to_integer() as u64);
            empty_counter = 0;
            let deadline = Deadline { pause, due: step.clone() };
            deadlines.push(deadline);
        }
        // There cannot be some gcd periods left at the end of the hyper period.
        assert!(empty_counter == 0);
        deadlines
    }
    fn sort_deadlines(ir: &RTLolaIR, deadlines: &mut Vec<Deadline>) {
        for deadline in deadlines {
            deadline.due.sort_by_key(|s| ir.outputs[*s].eval_layer());
        }
    }
}
mod math {
    use num::integer::gcd as num_gcd;
    use num::integer::lcm as num_lcm;
    use num::rational::Rational64 as Rational;

    pub(crate) fn rational_gcd(a: Rational, b: Rational) -> Rational {
        let numer = num_gcd(*a.numer(), *b.numer());
        let denom = num_lcm(*a.denom(), *b.denom());
        Rational::new(numer, denom)
    }

    pub(crate) fn rational_lcm(a: Rational, b: Rational) -> Rational {
        let numer = num_lcm(*a.numer(), *b.numer());
        let denom = num_gcd(*a.denom(), *b.denom());
        Rational::new(numer, denom)
    }

    pub(crate) fn rational_gcd_all(v: &[Rational]) -> Rational {
        assert!(!v.is_empty());
        v.iter().fold(v[0], |a, b| rational_gcd(a, *b))
    }

    pub(crate) fn rational_lcm_all(v: &[Rational]) -> Rational {
        assert!(!v.is_empty());
        v.iter().fold(v[0], |a, b| rational_lcm(a, *b))
    }
}

#[cfg(test)]
mod tests {
    use super::math::*;
    use super::*;
    use crate::ir::RTLolaIR;
    use crate::FrontendConfig;
    #[allow(unused_imports)]
    use num::FromPrimitive;
    use num::ToPrimitive;

    macro_rules! rat {
        ($i:expr) => {
            Rational::from_i64($i).unwrap()
        };
        ($n:expr, $d:expr) => {
            Rational::new($n, $d)
        };
    }
    #[test]
    fn test_gcd() {
        assert_eq!(rational_gcd(rat!(3), rat!(18)), rat!(3));
        assert_eq!(rational_gcd(rat!(18), rat!(3)), rat!(3));
        assert_eq!(rational_gcd(rat!(1), rat!(25)), rat!(1));
        assert_eq!(rational_gcd(rat!(5), rat!(13)), rat!(1));
        assert_eq!(rational_gcd(rat!(25), rat!(40)), rat!(5));
        assert_eq!(rational_gcd(rat!(7), rat!(7)), rat!(7));
        assert_eq!(rational_gcd(rat!(7), rat!(7)), rat!(7));

        assert_eq!(rational_gcd(rat!(1, 4), rat!(1, 2)), rat!(1, 4));
        assert_eq!(rational_lcm(rat!(1, 4), rat!(1, 2)), rat!(1, 2));
        assert_eq!(rational_gcd(rat!(2, 3), rat!(1, 8)), rat!(1, 24));
        assert_eq!(rational_lcm(rat!(2, 3), rat!(1, 8)), rat!(2));
    }

    fn to_ir(spec: &str) -> RTLolaIR {
        crate::parse("stdin", spec, FrontendConfig::default()).expect("spec was invalid")
    }

    /// Divides two durations. If `rhs` is not a divider of `lhs`, a warning is emitted and the
    /// rounding strategy `round_up` is applied.
    fn divide_durations(lhs: Duration, rhs: Duration, round_up: bool) -> usize {
        // The division of durations is currently unstable (feature duration_float) because
        // it falls back to using floats which cannot necessarily represent the durations
        // accurately. We, however, fall back to nanoseconds as u128. Regardless, some inaccuracies
        // might occur, rendering this code TODO *not stable for real-time devices!*
        let lhs = lhs.as_nanos();
        let rhs = rhs.as_nanos();
        let representable = lhs % rhs == 0;
        let mut div = lhs / rhs;
        if !representable {
            println!("Warning: Spec unstable: Cannot accurately represent extend periods.");
            // TODO: Introduce better mechanism for emitting such warnings.
            if round_up {
                div += 1;
            }
        }
        div as usize
    }

    #[test]
    fn test_extension_rate_extraction() {
        let input = "input a: UInt64\n";
        let hz50 = "output b: UInt64 @50Hz := 1\n";
        let hz40 = "output c: UInt64 @40Hz := 2\n";
        let ms20 = "output d: UInt64 @20ms := 3\n"; // 50Hz
        let ms1 = "output e: UInt64 @1ms := 4\n"; // 100Hz

        let case1 = (format!("{}{}", input, hz50), 20_000_000);
        let case2 = (format!("{}{}", input, hz40), 25_000_000);
        let case3 = (format!("{}{}{}", input, hz50, hz40), 5_000_000);
        let case4 = (format!("{}{}{}", input, hz50, ms1), 1_000_000);
        let case5 = (format!("{}{}{}{}", input, hz50, ms20, ms1), 1_000_000);

        let cases = [case1, case2, case3, case4, case5];
        for (spec, expected) in cases.iter() {
            let periods: Vec<_> = to_ir(spec).time_driven.iter().map(|s| s.period).collect();
            let was = Schedule::find_extend_period(&periods);
            let was = was.get::<nanosecond>().to_integer().to_u64().expect("");
            assert_eq!(*expected, was);
        }
    }

    #[test]
    fn test_divide_durations_round_down() {
        type TestDurations = ((u64, u32), (u64, u32), usize);
        let case1: TestDurations = ((1, 0), (1, 0), 1);
        let case2: TestDurations = ((1, 0), (0, 100_000_000), 10);
        let case3: TestDurations = ((1, 0), (0, 100_000), 10_000);
        let case4: TestDurations = ((1, 0), (0, 20_000), 50_000);
        let case5: TestDurations = ((0, 40_000), (0, 30_000), 1);
        let case6: TestDurations = ((3, 1_000), (3, 5_000), 0);

        let cases = [case1, case2, case3, case4, case5, case6];
        for (a, b, expected) in &cases {
            let to_dur = |(s, n)| Duration::new(s, n);
            let was = divide_durations(to_dur(*a), to_dur(*b), false);
            assert_eq!(was, *expected, "Expected {}, but was {}.", expected, was);
        }
    }

    #[test]
    fn test_divide_durations_round_up() {
        type TestDurations = ((u64, u32), (u64, u32), usize);
        let case1: TestDurations = ((1, 0), (1, 0), 1);
        let case2: TestDurations = ((1, 0), (0, 100_000_000), 10);
        let case3: TestDurations = ((1, 0), (0, 100_000), 10_000);
        let case4: TestDurations = ((1, 0), (0, 20_000), 50_000);
        let case5: TestDurations = ((0, 40_000), (0, 30_000), 2);
        let case6: TestDurations = ((3, 1_000), (3, 5_000), 1);

        let cases = [case1, case2, case3, case4, case5, case6];
        for (a, b, expected) in &cases {
            let to_dur = |(s, n)| Duration::new(s, n);
            let was = divide_durations(to_dur(*a), to_dur(*b), true);
            assert_eq!(was, *expected, "Expected {}, but was {}.", expected, was);
        }
    }
}
