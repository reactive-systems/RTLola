#![feature(test)]

extern crate test;

use rtlola_interpreter::Config;
use test::Bencher;

#[bench]
fn endtoend_semi_complex_spec(b: &mut Bencher) {
    let config = Config::new(&[
        "rtlola-interpreter".to_string(),
        "monitor".to_string(),
        "../traces/spec_offline.lola".to_string(),
        "--csv-in=../traces/timed/trace_0.csv".to_string(),
        "--verbosity=silent".to_string(),
        "--offline".to_string(),
    ]);
    b.iter(|| {
        config.clone().run().unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    });
}

#[bench]
fn endtoend_semi_complex_spec_triggers(b: &mut Bencher) {
    let config = Config::new(&[
        "rtlola-interpreter".to_string(),
        "monitor".to_string(),
        "../traces/spec_offline.lola".to_string(),
        "--csv-in=../traces/timed/trace_0_small.csv".to_string(),
        "--verbosity=triggers".to_string(),
        "--time-info-rep=relative".to_string(),
        "--offline".to_string(),
    ]);
    b.iter(|| {
        config.clone().run().unwrap_or_else(|e| panic!("E2E test failed: {}", e));
    });
}
