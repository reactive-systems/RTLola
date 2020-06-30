//! End-to-end tests of the RTLola evaluator

use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

fn run(spec: &str, data: &str) -> Result<Arc<OutputHandler>, Box<dyn std::error::Error>> {
    let ir = rtlola_frontend::parse("stdin", spec, CONFIG).unwrap_or_else(|e| panic!("spec is invalid: {}", e));
    let mut file = NamedTempFile::new().expect("failed to create temporary file");
    write!(file, "{}", data).expect("writing tempfile failed");
    let cfg = EvalConfig::new(
        EventSourceConfig::CSV { src: CSVInputSource::file(file.path().to_str().unwrap().to_string(), None, None) },
        Statistics::Debug,
        Verbosity::Silent,
        OutputChannel::StdErr,
        EvaluatorChoice::ClosureBased,
        ExecutionMode::Offline,
        TimeRepresentation::Hide,
    );
    let config = Config { cfg, ir };
    config.run()
}

#[test]
fn zero_wait_time_regression() {
    let spec = r#"
input a: Int64

output b @ 10Hz := a.hold().defaults(to:10)
output c @ 5Hz := a.hold().defaults(to:10)
    "#;

    let data = r#"a,time
1,0.0
2,0.01
1,0.11
2,0.21
1,0.31
2,0.41
1,0.51
2,0.61
1,0.71
"#;

    let _ = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
}

#[test]
fn test_parse_event() {
    let spec = r#"
input bool: Bool
input unsigned: UInt64
input signed: Int64
input float: Float64
input str: String

trigger bool = true
trigger unsigned = 3
trigger signed = -5
trigger float = -123.456
trigger str = "foobar"
        "#;

    let data = r#"float,bool,time,signed,str,unsigned
-123.456,true,1547627523.600536,-5,"foobar",3"#;

    let output_handler = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    macro_rules! assert_eq_num_trigger {
        ($ix:expr, $num:expr) => {
            assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger($ix), $num);
        };
    }
    assert_eq_num_trigger!(0, 1);
    assert_eq_num_trigger!(1, 1);
    assert_eq_num_trigger!(2, 1);
    assert_eq_num_trigger!(3, 1);
    assert_eq_num_trigger!(4, 1);
}

#[test]
fn add_two_i32_streams() {
    let spec = r#"
input a: Int64
input b: Int64

output c := a + b

trigger c > 2 "c is too large"
        "#;

    let data = r#"a,b,time
#,#,1547627523.000536
3,#,1547627523.100536
#,3,1547627523.200536
1,1,1547627523.300536
#,3,1547627523.400536
3,#,1547627523.500536
2,2,1547627523.600536"#;

    let output_handler = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 1);
}

#[test]
fn regex_simple() {
    let spec = r#"
import regex

input a: String

output x := matches(a, regex: "sub")
output y := a.matches(regex: "^sub")

trigger x "sub"
trigger y "^sub"
        "#;

    let data = r#"a,time
xub,24.8
sajhasdsub,24.9
subsub,25.0"#;

    let output_handler = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 2);
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(1), 1);
}

#[test]
fn regex_bytes() {
    let spec = r#"
import regex

input a: Bytes

trigger a.matches(regex: "^sub") "^sub"
        "#;

    let data = r#"a,time
xub,24.8
sajhasdsub,24.9
subsub,25.0"#;

    let handler = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    assert_eq!(handler.statistics.as_ref().unwrap().get_num_trigger(0), 1);
}

#[test]
fn timed_dependencies() {
    let spec = r#"
        output a @ 1Hz := b
        output b @ 1Hz := true
        output c @ 1Hz := d[-1].defaults(to: false)
        output d @ 1Hz := !d[-1].defaults(to: false)
        trigger a "a"
        trigger b "b"
        trigger c "c"
        trigger d "d"
    "#;

    let data = r#"time
0.0
1.0"#;

    let output_handler = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    // The test case is 1secs.
    // The execution should look as follows:
    //
    // time     | 0 | 1 |
    // a        | 1 | 1 |
    // b        | 1 | 1 |
    // c        | 0 | 1 |
    // d        | 1 | 0 |
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 2);
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(1), 2);
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(2), 1);
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(3), 1);
}

#[test]
fn event_based_parallel_past_lookup() {
    let spec = r#"
input time : Float64
output a @ time := b[-1].defaults(to: false)
output b @ time := a[-1].defaults(to: true)
trigger a "a"
trigger b "b"
trigger a∧b "a∧b"
trigger a∨b "a∨b"
    "#;

    let data = r#"time
0.0
1.0
2.0
3.0
4.0
5.0
6.0"#;

    let output_handler = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    // The test case is 6secs, there should be 3 triggers.
    // The execution should look as follows:
    //
    // time     | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
    // a        | 0 | 1 | 0 | 1 | 0 | 1 | 0 |
    // b        | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
    // trig a   | 0 | 1 | 0 | 1 | 0 | 1 | 0 |
    // trig b   | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
    // trig a∧b | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    // trig a∨b | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 3);
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(1), 4);
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(2), 0);
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(3), 7);
}

#[test]
fn timed_parallel_past_lookup() {
    let spec = r#"
output a @ 1Hz := b[-1].defaults(to: false)
output b @ 1Hz := a[-1].defaults(to: true)
trigger a "a"
trigger b "b"
trigger a∧b "a∧b"
trigger a∨b "a∨b"
    "#;

    let data = r#"time
0.0
1.0
2.0
3.0
4.0
5.0
6.0"#;

    let output_handler = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    // The test case is 6secs, there should be 3 triggers.
    // The execution should look as follows:
    //
    // time     | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
    // a        | 0 | 1 | 0 | 1 | 0 | 1 | 0 |
    // b        | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
    // trig a   | 0 | 1 | 0 | 1 | 0 | 1 | 0 |
    // trig b   | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
    // trig a∧b | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    // trig a∨b | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 3);
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(1), 4);
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(2), 0);
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(3), 7);
}

#[test]
fn event_based_counter() {
    let spec = r#"
input time : Float64
output b @ time := b[-1].defaults(to: 0) + 1
trigger b > 3
    "#;

    let data = r#"time
0.0
1.0
2.0
3.0
4.0
5.0
6.0"#;

    let output_handler = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    // the test case is 6secs, the counter starts with 1 at 0.0 and increases every second, thus, there should be 4 trigger (4 times counter > 3)
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 4);
}

#[test]
fn timed_counter() {
    let spec = r#"
output b @10Hz := b[-1].defaults(to: 0) + 1
trigger b > 3
    "#;

    let data = r#"a,time
1,0.0
2,0.01
1,0.11
2,0.21
1,0.31
2,0.41
1,0.51
2,0.61
1,0.71
"#;

    let output_handler = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    // the test case is 710ms, the counter starts with 1 at 0.0 and increases every 100ms, thus, there should be 5 trigger
    //
    // The execution should look as follows:
    //
    // time  | 0.0 | 0.01 | 0.1 | 0.11 | 0.2 | 0.21 | 0.3 | 0.31 | 0.4 | 0.41 | 0.5 | 0.51 | 0.6 | 0.61 | 0.7 | 0.71
    // in a  |   1 |    2 |   - |    1 |   - |    2 |   - |    1 |   - |    2 |   - |    1 |   - |    2 |   - |    1
    // out b |   1 |    - |   2 |    - |   3 |    - |   4 |    - |   5 |    - |   6 |    - |   7 |    - |   8 |    -
    // trig  |   0 |    - |   0 |    - |   0 |    - |   1 |    - |   1 |    - |   1 |    - |   1 |    - |   1 |    -
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 5);
}

#[test]
fn min_max_window() {
    let spec = r#"
input a: Int64

output min @ 1Hz := a.aggregate(over: 1s, using: min).defaults(to: 0)
output max @ 1Hz := a.aggregate(over: 1s, using: max).defaults(to: 0)

trigger min == 1
trigger max == 2
    "#;

    let data = r#"a,time
0,0
1,0.1
2,0.5
3,1.1
"#;

    let output_handler = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 1);
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(1), 1);
}

#[test]
fn bytes_at() {
    let spec = r#"
input a: Bytes

output x := a.at(index: 0).defaults(to: 0)

trigger x == 49 // utf-8 character value of "1"
trigger x == 50 // utf-8 character value of "2"
        "#;

    let data = r#"a,time
0,0
1,0.1
2,0.5
3,1.1"#;

    let handler = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    assert_eq!(handler.statistics.as_ref().unwrap().get_num_trigger(0), 1);
    assert_eq!(handler.statistics.as_ref().unwrap().get_num_trigger(1), 1);
}

#[test]
fn rtlola_stream_but_eventbased() {
    let spec = r#"
    input acc_x: Float64
    input timing: Float64
    output vel_x@10Hz:= vel_x.offset(by:-1).defaults(to:0.0) + acc_x.aggregate(over:1s, using:integral)
    output test := vel_x.hold().defaults(to:0.0) + timing > -1.0
    trigger test
    "#;
    let data = r#"acc_x,timing,time
1.0,0.0,1
"#;
    let output_handler = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 1);
    //    assert_eq!(output_handler.statistics.as_ref().unwrap().get_num_trigger(1), 1);
}
