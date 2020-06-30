#![forbid(unused_must_use)] // disallow discarding errors
#![warn(
    // missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]

mod basics;
mod closuregen;
mod coordination;
mod evaluator;
mod storage;
#[cfg(test)]
mod tests;

use crate::basics::OutputHandler;
use crate::coordination::Controller;
use basics::{
    CSVInputSource, EvalConfig, EvaluatorChoice, EventSourceConfig, ExecutionMode, OutputChannel, PCAPInputSource,
    Statistics, TimeFormat, TimeRepresentation, Verbosity,
};
use clap::{App, AppSettings, Arg, ArgGroup, SubCommand};
use rtlola_frontend;
use rtlola_frontend::ir::RTLolaIR;
use rtlola_frontend::{FrontendConfig, TypeConfig};
use std::fs;
use std::sync::Arc;

pub use crate::coordination::Monitor;
pub use crate::storage::Value;

// TODO add example to doc

/**
`Config` combines an RTLola specification in `LolaIR` form with an `EvalConfig`.

The evaluation configuration describes how the specification should be executed.
The `Config` can then be turned into a monitor for use via the API or simply executed.
*/
#[derive(Debug, Clone)]
pub struct Config {
    cfg: EvalConfig,
    ir: RTLolaIR,
}

const CONFIG: FrontendConfig =
    FrontendConfig { ty: TypeConfig { use_64bit_only: true, type_aliases: false }, allow_parameters: false };

impl Config {
    // TODO find appropriate name
    /**
    Creates a new `Config` which can then be turned into a `Monitor` by `into_monitor`.
    */
    pub fn new_api(cfg: EvalConfig, ir: RTLolaIR) -> Config {
        Config { cfg, ir }
    }

    /**
    Parses command line arguments and return a `Config` if successful.

    If the arguments are not valid, this function will print an error message and exit the process with value 1.
    */
    #[allow(unsafe_code)]
    pub fn new(args: &[String]) -> Self {
        let parse_matches = App::new("RTLola")
        .version(env!("CARGO_PKG_VERSION"))
        .author(clap::crate_authors!("\n"))
        .about("RTLola is a tool to analyze and monitor Lola specifications.") // TODO description
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .subcommand(
            SubCommand::with_name("monitor")
            .about("Start monitoring for the given specification")
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
                    .hidden(cfg!(feature = "public"))
            )
        )
        .subcommand(
            SubCommand::with_name("analyze")
            .about("Parses the input file and runs semantic analysis")
            .arg(
                Arg::with_name("SPEC")
                    .help("Sets the specification file to use")
                    .required(true)
                    .index(1),
            )
        )
        .subcommand(
            SubCommand::with_name("ids")
            .about("Use the rtlola monitor as a network intrusion detection system")
            .arg(
                Arg::with_name("SPEC")
                    .help("Sets the specification file to use")
                    .required(true)
                    .index(1)
            )
            .arg(
                Arg::with_name("LOCAL_NETWORK")
                    .help("The local ip range given in CIDR notation")
                    .required(true)
                    .index(2)
            )
            .arg(
                Arg::with_name("NETWORK_INTERFACE")
                    .help("Read the packets from a network interface")
                    .long("net-iface")
                    .takes_value(true)
                    .number_of_values(1)
            )
            .arg(
                Arg::with_name("PCAP_INPUT_FILE")
                    .help("Read pcap input from file.")
                    .long("pcap-in")
                    .takes_value(true)
                    .number_of_values(1)
            )
            .group(
                ArgGroup::with_name("INPUT_MODE")
                .args(&["NETWORK_INTERFACE", "PCAP_INPUT_FILE"])
                .required(true)
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
                    .conflicts_with("STDOUT")
            )
            .arg(
                Arg::with_name("DELAY")
                    .help("Delay [ms] between reading in two lines from the input\nOnly used for file input.")
                    .long("delay")
                    .requires("PCAP_INPUT_FILE")
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
                Arg::with_name("INTERPRETED")
                    .long("interpreted")
                    .help("Interpret expressions instead of compilation")
                    .hidden(cfg!(feature = "public"))
            )
        )
        .get_matches_from(args);

        if let Some(parse_matches) = parse_matches.subcommand_matches("analyze") {
            let filename = parse_matches.value_of("SPEC").map(|s| s.to_string()).unwrap();
            rtlola_frontend::analyze(filename.as_str(), CONFIG);
            std::process::exit(0);
        }
        let mut ids_mode = false;
        let parse_matches = if let Some(matches) = parse_matches.subcommand_matches("monitor") {
            matches
        } else if let Some(matches) = parse_matches.subcommand_matches("ids") {
            ids_mode = true;
            matches
        } else {
            eprintln!("Unknown subcommand. See help for more information.");
            std::process::exit(1)
        };

        let filename = parse_matches.value_of("SPEC").map(|s| s.to_string()).unwrap();
        let contents = fs::read_to_string(&filename).unwrap_or_else(|e| {
            eprintln!("Could not read file `{}`: {}", filename, e);
            std::process::exit(1)
        });

        let ir = match rtlola_frontend::parse(&filename, contents.as_str(), CONFIG) {
            Ok(ir) => ir,
            Err(err) => {
                eprintln!("{}", err);
                std::process::exit(1);
            }
        };

        let delay = match parse_matches.value_of("DELAY") {
            None => None,
            Some(delay_str) => {
                let d = delay_str.parse::<humantime::Duration>().unwrap_or_else(|e| {
                    eprintln!("Could not parse DELAY value `{}`: {}.", delay_str, e);
                    std::process::exit(1);
                });
                Some(d.into())
            }
        };

        let csv_time_column = if ids_mode {
            None
        } else {
            parse_matches.value_of("CSV_TIME_COLUMN").map(|col| {
                let col = col.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("time column needs to be a positive integer");
                    std::process::exit(1)
                });
                if col == 0 {
                    eprintln!("time column needs to be a positive integer (first column = 1)");
                    std::process::exit(1);
                }
                col
            })
        };

        let src = if ids_mode {
            let pcap_load = unsafe { pcap_on_demand::load_pcap_library() };
            if let Err(err) = pcap_load {
                eprintln!("Could not load PCAP library: {}", err.to_string());
                std::process::exit(1);
            }
            let local_network = String::from(parse_matches.value_of("LOCAL_NETWORK").unwrap());
            if let Some(file) = parse_matches.value_of("PCAP_INPUT_FILE") {
                EventSourceConfig::PCAP {
                    src: PCAPInputSource::File { path: String::from(file), delay, local_network },
                }
            } else if let Some(iface) = parse_matches.value_of("NETWORK_INTERFACE") {
                EventSourceConfig::PCAP { src: PCAPInputSource::Device { name: String::from(iface), local_network } }
            } else {
                unreachable!(); //Excluded by CLAP
            }
        } else if let Some(file) = parse_matches.value_of("CSV_INPUT_FILE") {
            EventSourceConfig::CSV { src: CSVInputSource::file(String::from(file), delay, csv_time_column) }
        } else {
            EventSourceConfig::CSV { src: CSVInputSource::stdin() }
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
        let evaluator = if parse_matches.is_present("INTERPRETED") { Interpreted } else { ClosureBased };

        use ExecutionMode::*;
        let mut mode = Offline;
        if !ids_mode {
            if parse_matches.is_present("ONLINE") {
                mode = Online;
            }
        } else if parse_matches.is_present("NETWORK_INTERFACE") {
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

    /**
    Turns a `Config` that was created through a call to `new_api` into a `Monitor`.
    */
    pub fn into_monitor(self) -> Result<Monitor, Box<dyn std::error::Error>> {
        assert_eq!(self.cfg.mode, ExecutionMode::API);
        Controller::new(self.ir, self.cfg).start().map(|res| res.left().unwrap())
    }

    /**
    Runs a `Config` that was created through a call to `new`.
    */
    pub fn run(self) -> Result<Arc<OutputHandler>, Box<dyn std::error::Error>> {
        // TODO: Rather than returning OutputHandler publicly --- let alone an Arc ---, transform into more suitable format or make OutputHandler more accessible.
        Controller::new(self.ir, self.cfg)
            .start()
            .map(|r| r.right().expect("Running the config should never return a Monitor."))
    }
}
