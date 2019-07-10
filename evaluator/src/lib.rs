#![deny(unsafe_code)] // disallow unsafe code by default
#![forbid(unused_must_use)] // disallow discarding errors

pub mod basics;
mod closuregen;
mod coordination;
mod evaluator;
mod storage;
#[cfg(test)]
mod tests;

use crate::coordination::Controller;
use basics::{
    EvalConfig, EvaluatorChoice, ExecutionMode, InputSource, OutputChannel, Statistics, TimeFormat, TimeRepresentation,
    Verbosity,
};
use clap::{App, AppSettings, Arg, ArgGroup};
use std::fs;
use streamlab_frontend;
use streamlab_frontend::ir::LolaIR;

#[derive(Debug, Clone)]
pub struct Config {
    cfg: EvalConfig,
    ir: LolaIR,
}

impl Config {
    pub fn new(args: &[String]) -> Self {
        let parse_matches = App::new("StreamLAB")
        .version(env!("CARGO_PKG_VERSION"))
        .author(clap::crate_authors!("\n"))
        .about("StreamLAB is a tool to analyze and monitor Lola specifications.") // TODO description
        .setting(AppSettings::ArgRequiredElseHelp)
        .arg(
            Arg::with_name("SPEC")
                .help("Sets the specification file to use")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name("STDIN")
                .help("Read CSV input from stdin [default]")
                .long("stdin")
        )
        .arg(
            Arg::with_name("CSV_INPUT_FILE")
                .help("Read CSV input from a file")
                .long("csv-in")
                .takes_value(true)
                .number_of_values(1)
                .conflicts_with("STDIN")
        )
        .arg(
            Arg::with_name("CSV_TIME_COLUMN")
                .help("The column in the CSV that contains time info")
                .long("csv-time-column")
                .requires("CSV_INPUT_FILE")
                .takes_value(true)
                .number_of_values(1)
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
                .help("Delay [ms] between reading in two lines from the input\nOnly used for file input.")
                .long("delay")
                .requires("CSV_INPUT_FILE")
                .conflicts_with("ONLINE")
                .takes_value(true)
                .number_of_values(1)
        )
        .arg(
            Arg::with_name("VERBOSITY")
                .help("Sets the verbosity\n")
                .long("verbosity")
                .possible_values(&["debug", "outputs", "triggers", "warnings", "progress", "silent", "quiet"])
                .default_value("triggers")
        )
        .arg(
            Arg::with_name("TIMEREPRESENTATION")
                .help("Sets the trigger time info representation\n")
                .long("time-info-rep")
                .possible_values(&[
                    "hide",
                    "relative",
                    "relative_nanos", "relative_uint_nanos",
                    "relative_secs", "relative_float_secs",
                    "relative_human", "relative_human_time",
                    "absolute",
                    "absolute_nanos", "absolute_uint_nanos",
                    "absolute_secs", "absolute_float_secs",
                    "absolute_human", "absolute_human_time",
                ])
                .default_value("hide")
        )
        .arg(
            Arg::with_name("ONLINE")
                .long("online")
                .help("Use the current system time for timestamps")
        )
        .arg(
            Arg::with_name("OFFLINE")
                .long("offline")
                .help("Use the timestamps from the input\nThe column name must be one of [time,timestamp,ts](case insensitive).\nThe column must produce a monotonically increasing sequence of values.")
        )
        .group(
            ArgGroup::with_name("MODE")
                .required(true)
                .args(&["ONLINE", "OFFLINE"])
        )
        .arg(
            Arg::with_name("INTERPRETED")
                .long("interpreted")
                .help("Interpret expressions instead of compilation")
        )
        .get_matches_from(args);

        // Now we have a reference to clone's matches
        let filename = parse_matches.value_of("SPEC").map(|s| s.to_string()).unwrap();

        let contents =
            fs::read_to_string(&filename).unwrap_or_else(|e| panic!("Could not read file {}: {}", filename, e));

        let ir = match streamlab_frontend::parse(contents.as_str()) {
            Ok(ir) => ir,
            Err(err) => {
                eprintln!("{}", err);
                std::process::exit(1);
            }
        };

        let delay = match parse_matches.value_of("DELAY") {
            None => None,
            Some(delay_str) => {
                let d = delay_str
                    .parse::<humantime::Duration>()
                    .unwrap_or_else(|e| panic!("Could not parse DELAY value `{}`: {}.", delay_str, e));
                Some(d.into())
            }
        };

        let csv_time_column = parse_matches.value_of("CSV_TIME_COLUMN").map(|col| {
            let col = col.parse::<usize>().expect("time column needs to be a positive integer");
            if col == 0 {
                panic!("time column needs to be a positive integer (first column = 1)");
            }
            col
        });

        let src = if let Some(file) = parse_matches.value_of("CSV_INPUT_FILE") {
            InputSource::file(String::from(file), delay, csv_time_column)
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

        use Verbosity::*;
        let verbosity = match parse_matches.value_of("VERBOSITY").unwrap() {
            "debug" => Debug,
            "outputs" => Outputs,
            "triggers" => Triggers,
            "warnings" => WarningsOnly,
            "progress" => Progress,
            "silent" | "quiet" => Silent,
            _ => unreachable!(),
        };

        use EvaluatorChoice::*;
        let mut evaluator = ClosureBased;
        if parse_matches.is_present("INTERPRETED") {
            evaluator = Interpreted;
        }

        use ExecutionMode::*;
        let mut mode = Offline;
        if parse_matches.is_present("ONLINE") {
            mode = Online;
        }

        use TimeFormat::*;
        use TimeRepresentation::*;
        let time_representation = match parse_matches.value_of("TIMEREPRESENTATION").unwrap() {
            "hide" => Hide,
            "relative_nanos" | "relative_uint_nanos" => Relative(UIntNanos),
            "relative" | "relative_secs" | "relative_float_secs" => Relative(FloatSecs),
            "relative_human" | "relative_human_time" => Relative(HumanTime),
            "absolute_nanos" | "absolute_uint_nanos" => Absolute(UIntNanos),
            "absolute" | "absolute_secs" | "absolute_float_secs" => Absolute(FloatSecs),
            "absolute_human" | "absolute_human_time" => Absolute(HumanTime),
            _ => unreachable!(),
        };

        let cfg = EvalConfig::new(src, Statistics::None, verbosity, out, evaluator, mode, time_representation);

        Config { cfg, ir }
    }

    pub fn run(self) -> Result<Controller, Box<dyn std::error::Error>> {
        let controller = Controller::new(self.ir, self.cfg);
        controller.start()?;
        Ok(controller)
    }
}
