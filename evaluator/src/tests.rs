//! End-to-end tests of the StreamLAB evaluator

use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

const ALTERNATING_SINGLE_INT32: &str = r#"a,time
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

#[test]
#[ignore] // needs to be fixed after merge (see https://gitlab.com/reactive-systems/lolaparser/issues/47)
fn zero_wait_time_regression() {
    let spec = r#"
        input a: Int32

        output b @ 10Hz := a.hold().defaults(to:10)
        output c @ 5Hz := a.hold().defaults(to:10)
    "#;
    let ir = streamlab_frontend::parse(spec);
    let mut file = NamedTempFile::new().expect("failed to create temporary file");
    write!(file, "{}", ALTERNATING_SINGLE_INT32).expect("writing tempfile failed");

    let cfg = EvalConfig::new(
        InputSource::file(file.path().to_str().unwrap().to_string(), None, None),
        Verbosity::Progress,
        OutputChannel::StdErr,
        EvaluatorChoice::ClosureBased,
        ExecutionMode::Offline,
    );
    let config = Config { cfg, ir };
    config.run().unwrap_or_else(|e| panic!("E2E test failed: {}", e));
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
    let ir = streamlab_frontend::parse(spec);
    let mut file = NamedTempFile::new().expect("failed to create temporary file");
    write!(
        file,
        r#"float,bool,time,signed,str,unsigned
-123.456,true,1547627523.600536,-5,"foobar",3"#
    )
    .expect("writing tempfile failed");

    let cfg = EvalConfig::new(
        InputSource::file(file.path().to_str().unwrap().to_string(), None, None),
        Verbosity::Progress,
        OutputChannel::StdErr,
        EvaluatorChoice::ClosureBased,
        ExecutionMode::Offline,
    );
    let config = Config { cfg, ir };
    let ctrl = config.run().unwrap_or_else(|e| panic!("E2E test failed: {}", e));
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
    let ir = streamlab_frontend::parse(spec);
    let mut file = NamedTempFile::new().expect("failed to create temporary file");
    write!(
        file,
        "a,b,time
#,#,1547627523.000536
3,#,1547627523.100536
#,3,1547627523.200536
1,1,1547627523.300536
#,3,1547627523.400536
3,#,1547627523.500536
2,2,1547627523.600536"
    )
    .expect("writing tempfile failed");

    let cfg = EvalConfig::new(
        InputSource::file(file.path().to_str().unwrap().to_string(), None, None),
        Verbosity::Progress,
        OutputChannel::StdErr,
        EvaluatorChoice::ClosureBased,
        ExecutionMode::Offline,
    );
    let config = Config { cfg, ir };
    let ctrl = config.run().unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    assert_eq!(ctrl.output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 1);
}

#[test]
fn regex_simple() {
    let spec = r#"
            import regex

            input a: String

            output x := matches(a, regex: "sub")
            output y := matches(a, regex: "^sub")

            trigger x "sub"
            trigger y "^sub"
        "#;
    let ir = streamlab_frontend::parse(spec);
    let mut file = NamedTempFile::new().expect("failed to create temporary file");
    write!(
        file,
        "a,time
xub,24.8
sajhasdsub,24.9
subsub,25.0"
    )
    .expect("writing tempfile failed");

    let cfg = EvalConfig::new(
        InputSource::file(file.path().to_str().unwrap().to_string(), None, None),
        Verbosity::Progress,
        OutputChannel::StdErr,
        EvaluatorChoice::ClosureBased,
        ExecutionMode::Offline,
    );
    let config = Config { cfg, ir };
    let ctrl = config.run().unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    assert_eq!(ctrl.output_handler.statistics.as_ref().unwrap().get_num_trigger(0), 2);
    assert_eq!(ctrl.output_handler.statistics.as_ref().unwrap().get_num_trigger(1), 1);
}
