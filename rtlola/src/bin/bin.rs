const SPEC: &str = "\
                    input in: UInt32 \
                    output a: UInt32 { extend @ 2Hz } := (in[0] ! 0) + 1000 \
                    output b: UInt32 { extend @ 1Hz } := (in[0] ! 0) + 10000 \
                    output c: UInt32 { extend @ 0.5Hz } := (in[0] ! 0) + 100000 \
                    ";

const PATH: &str = "rtlola/src/bin/successive.csv";

use rtlola::EvalConfig;
use rtlola::InputSource;
use std::time::Duration;

fn main() {
    println!("Starting RTLola.");
    println!("Using static spec: \n{}", SPEC);
    let ir = lola_parser::parse(SPEC);
    let src = InputSource::with_delay(String::from(PATH), Duration::new(0, 250_000_000));
    let cfg = EvalConfig::new(src, rtlola::Verbosity::Debug, rtlola::OutputChannel::StdOut);
    rtlola::start_evaluation(ir, cfg, None);
}
