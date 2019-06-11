//! End-to-end tests of the StreamLAB evaluator

use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

fn run(spec: &str, data: &str) -> Result<Controller, Box<dyn std::error::Error>> {
    let ir = streamlab_frontend::parse(spec);
    let mut file = NamedTempFile::new().expect("failed to create temporary file");
    write!(file, "{}", data).expect("writing tempfile failed");
    let cfg = EvalConfig::new(
        InputSource::file(file.path().to_str().unwrap().to_string(), None, None),
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
#[ignore] // needs to be fixed after merge (see https://gitlab.com/reactive-systems/lolaparser/issues/47)
fn zero_wait_time_regression() {
    let spec = r#"
input a: Int32

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
input unsigned: UInt8
input signed: Int8
input float: Float32
input str: String

trigger bool = true
trigger unsigned = 3
trigger signed = -5
trigger float = -123.456
trigger str = "foobar"
        "#;

    let data = r#"float,bool,time,signed,str,unsigned
-123.456,true,1547627523.600536,-5,"foobar",3"#;

    let ctrl = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    macro_rules! assert_eq_num_trigger {
        ($ix:expr, $num:expr) => {
            assert_eq!(ctrl.output_handler.statistics.as_ref().unwrap().get_num_trigger($ix), $num);
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
input a: Int32
input b: Int32

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

    let ctrl = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    assert_eq!(ctrl.output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 1);
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

    let ctrl = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    assert_eq!(ctrl.output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 2);
    assert_eq!(ctrl.output_handler.statistics.as_ref().unwrap().get_num_trigger(1), 1);
}

#[test]
fn event_based_counter() {
    let spec = r#"
input time : Float32
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

    let ctrl = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    // the test case is 6secs, the counter starts with 1 at 0.0 and increases every second, thus, there should be 4 trigger (4 times counter > 3)
    assert_eq!(ctrl.output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 4);
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

    let ctrl = run(spec, data).unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    // the test case is 710ms, the counter starts with 1 at 0.0 and increases every 100ms, thus, there should be 5 trigger
    //
    // The execution should look as follows:
    //
    // time  | 0.0 | 0.01 | 0.1 | 0.11 | 0.2 | 0.21 | 0.3 | 0.31 | 0.4 | 0.41 | 0.5 | 0.51 | 0.6 | 0.61 | 0.7 | 0.71
    // in a  |   1 |    2 |   - |    1 |   - |    2 |   - |    1 |   - |    2 |   - |    1 |   - |    2 |   - |    1
    // out b |   1 |    - |   2 |    - |   3 |    - |   4 |    - |   5 |    - |   6 |    - |   7 |    - |   8 |    -
    // trig  |   0 |    - |   0 |    - |   0 |    - |   1 |    - |   1 |    - |   1 |    - |   1 |    - |   1 |    -
    assert_eq!(ctrl.output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 5);
}
