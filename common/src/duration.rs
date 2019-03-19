use std::time::Duration;

const NANOS_PER_SEC: u128 = 1_000_000_000;

pub fn dur_as_nanos(dur: Duration) -> u128 {
    u128::from(dur.as_secs()) * NANOS_PER_SEC + u128::from(dur.subsec_nanos())
}

pub fn dur_from_nanos(dur: u128) -> Duration {
    // TODO: Introduce sanity checks for `dur` s.t. cast is safe.
    let secs = (dur / NANOS_PER_SEC) as u64; // safe cast for realistic values of `dur`.
    let nanos = (dur % NANOS_PER_SEC) as u32; // safe case
    Duration::new(secs, nanos)
}

/// Divides two durations. If `rhs` is not a divider of `lhs`, a warning is emitted and the
/// rounding strategy `round_up` is applied.
pub fn divide_durations(lhs: Duration, rhs: Duration, round_up: bool) -> usize {
    // The division of durations is currently unstable (feature duration_float) because
    // it falls back to using floats which cannot necessarily represent the durations
    // accurately. We, however, fall back to nanoseconds as u128. Regardless, some inaccuracies
    // might occur, rendering this code TODO *not stable for real-time devices!*
    let lhs = dur_as_nanos(lhs);
    let rhs = dur_as_nanos(rhs);
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
