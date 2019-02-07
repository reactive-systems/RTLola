use clap::{value_t, App, Arg, ArgGroup, SubCommand};
use lola_parser;
use rtlola::EvalConfig;
use rtlola::InputSource;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::process;
use std::time::Duration;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    let matches = App::new("rtlola")
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS")) //TODO authors
        .about("rtlola is a tool to analyze and monitor Lola specifications") // TODO description
        .subcommand(
            SubCommand::with_name("analyze")
                .about("Analyze the specification")
                .arg(
                    Arg::with_name("SPEC")
                        .help("Sets the specification file to use")
                        .required(true)
                        .index(1),
                ),
        )
        .subcommand(
            SubCommand::with_name("monitor")
                .about("Monitors the specification")
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
                        .possible_values(&["debug", "outputs", "triggers", "warnings", "silent"])
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
                .group(
                    ArgGroup::with_name("MODE")
                        .required(true)
                        .args(&["ONLINE", "OFFLINE"])
                )
        )
        .get_matches();

    match matches.subcommand() {
        ("analyze", Some(parse_matches)) => {
            // Now we have a reference to clone's matches
            let filename = parse_matches.value_of("SPEC").map(|s| s.to_string()).unwrap();
            let mut file = File::open(&filename)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            let _ = lola_parser::parse(contents.as_str());
            Ok(())
        }
        ("monitor", Some(parse_matches)) => {
            // Now we have a reference to clone's matches
            let filename = parse_matches.value_of("SPEC").map(|s| s.to_string()).unwrap();

            let mut file = File::open(&filename)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;

            let ir = lola_parser::parse(contents.as_str());

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
                rtlola::OutputChannel::StdOut
            } else if let Some(file) = parse_matches.value_of("OUTPUT_FILE") {
                rtlola::OutputChannel::File(String::from(file))
            } else {
                rtlola::OutputChannel::StdErr
            };

            let verbosity = match parse_matches.value_of("VERBOSITY").unwrap() {
                "debug" => rtlola::Verbosity::Debug,
                "outputs" => rtlola::Verbosity::Outputs,
                "triggers" => rtlola::Verbosity::Triggers,
                "warnings" => rtlola::Verbosity::WarningsOnly,
                "silent" => rtlola::Verbosity::Silent,
                _ => unreachable!(),
            };

            let cfg = EvalConfig::new(src, verbosity, out);

            if parse_matches.is_present("OFFLINE") {
                rtlola::start_evaluation_offline(ir, cfg);
            } else {
                rtlola::start_evaluation_online(ir, cfg);
            }

            //            Ok(())
        }
        ("", None) => {
            println!("No subcommand was used");
            println!("{}", matches.usage());

            process::exit(1)
        }
        _ => unreachable!(),
    }
}
