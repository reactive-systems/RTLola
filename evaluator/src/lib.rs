#![deny(unsafe_code)] // disallow unsafe code by default
#![forbid(unused_must_use)] // disallow discarding errors

pub mod basics;
mod closuregen;
mod coordination;
mod evaluator;
mod storage;

use crate::coordination::Controller;
use basics::{EvalConfig, InputSource, OutputChannel, Verbosity};
use clap::{value_t, App, Arg, ArgGroup};
use std::fs::File;
use std::io::Read;
use std::time::Duration;
use streamlab_frontend;
use streamlab_frontend::ir::LolaIR;

pub struct Config {
    cfg: EvalConfig,
    offline: bool,
    ir: LolaIR,
}

impl Config {
    pub fn new(args: &[String]) -> Self {
        let parse_matches = App::new("StreamLAB")
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about("StreamLAB is a tool to analyze and monitor Lola specifications") // TODO description
        .arg(
            Arg::with_name("SPEC")
                .help("Sets the specification file to use")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name("STDIN")
                .help("Read CSV input from stdin [Default]")
                .long("stdin")
        )
        .arg(
            Arg::with_name("CSV_INPUT_FILE")
                .help("Read CSV input from a file")
                .long("csv-in")
                .takes_value(true)
                .conflicts_with("STDIN")
        )
        .arg(
            Arg::with_name("STDOUT")
                .help("Output to stdout")
                .long("stdout")
        )
        .arg(
            Arg::with_name("STDERR")
                .help("Output to stderr")
                .long("stderr")
                .conflicts_with_all(&["STDOUT", "OUTPUT_FILE"])
        )
        .arg(
            Arg::with_name("DELAY")
                .short("d")
                .long("delay")
                .help("Delay [ms] between reading in two lines from the input. Only used for file input.")
                .requires("CSV_INPUT_FILE")
                .conflicts_with("OFFLINE")
                .takes_value(true)
        ).
        arg(
            Arg::with_name("VERBOSITY")
                .short("l")
                .long("verbosity")
                .possible_values(&["debug", "outputs", "triggers", "warnings", "progress", "silent", "quiet"])
                .default_value("triggers")
        )
        .arg(
            Arg::with_name("ONLINE")
                .long("online")
                .help("Use the current system time for timestamps")
        )
        .arg(
            Arg::with_name("OFFLINE")
                .long("offline")
                .help("Use the timestamps from the input.\nThe column name must be one of [time,timestamp,ts].\nThe column must produce a monotonically increasing sequence of values.")
        )
        .arg(
            Arg::with_name("INTERPRETED").long("interpreted").help("Interpret expressions instead of compilation")
        )
        .group(
            ArgGroup::with_name("MODE")
                .required(true)
                .args(&["ONLINE", "OFFLINE"])
        )
        .get_matches_from(args);

        // Now we have a reference to clone's matches
        let filename = parse_matches.value_of("SPEC").map(|s| s.to_string()).unwrap();

        let mut file = File::open(&filename).unwrap_or_else(|e| panic!("Could not open file {}: {}", filename, e));
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap_or_else(|e| panic!("Could not read file {}: {}", filename, e));

        let ir = streamlab_frontend::parse(contents.as_str());

        let delay = if parse_matches.is_present("DELAY") {
            value_t!(parse_matches, "DELAY", u32).unwrap_or_else(|_| {
                eprintln!(
                    "DELAY value `{}` is not a number.\nUsing no delay.",
                    parse_matches.value_of("DELAY").expect("We set a default value.")
                );
                0
            })
        } else {
            0
        };
        let delay = Duration::new(0, 1_000_000 * delay);

        let src = if let Some(file) = parse_matches.value_of("CSV_INPUT_FILE") {
            InputSource::with_delay(String::from(file), delay)
        } else {
            InputSource::stdin()
        };

        let out = if parse_matches.is_present("STDOUT") {
            OutputChannel::StdOut
        } else if let Some(file) = parse_matches.value_of("OUTPUT_FILE") {
            OutputChannel::File(String::from(file))
        } else {
            OutputChannel::StdErr
        };

        let verbosity = match parse_matches.value_of("VERBOSITY").unwrap() {
            "debug" => Verbosity::Debug,
            "outputs" => Verbosity::Outputs,
            "triggers" => Verbosity::Triggers,
            "warnings" => Verbosity::WarningsOnly,
            "progress" => Verbosity::Progress,
            "silent" | "quiet" => Verbosity::Silent,
            _ => unreachable!(),
        };

        let closure_based_evaluator = !parse_matches.is_present("INTERPRETED");

        let cfg = EvalConfig::new(src, verbosity, out, closure_based_evaluator);

        Config { cfg, offline: parse_matches.is_present("OFFLINE"), ir }
    }

    pub fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        if self.offline {
            Controller::evaluate_offline(self.ir, self.cfg)
        } else {
            Controller::evaluate_online(self.ir, self.cfg);
        }
    }
}
