use std::time::Duration;

/// Divides two durations. If `rhs` is not a divider of `lhs`, a warning is emitted and the
/// rounding strategy `round_up` is applied.
pub fn divide_durations(lhs: Duration, rhs: Duration, round_up: bool) -> usize {
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
