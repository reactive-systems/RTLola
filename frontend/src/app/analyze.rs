//! This module contains the logic for the `streamlab-analyze` binary.

use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;

use clap::{App, Arg, SubCommand};
use pest::Parser;
use simplelog::*;

use crate::analysis;
use crate::ir::lowering::Lowering;
use crate::parse::{LolaParser, Rule, SourceMapper};
use crate::reporting::Handler;
use crate::ty::TypeConfig;
use crate::FrontendConfig;

const CONFIG: FrontendConfig =
    FrontendConfig { ty: TypeConfig { use_64bit_only: true, type_aliases: true }, allow_parameters: true };

enum Analysis {
    Parse,
    AST,
    Prettyprint,
    Analyze,
    IR,
}

pub struct Config {
    which: Analysis,
    filename: String,
}

impl Config {
    pub fn new(args: &[String]) -> Self {
        let matches = App::new("streamlab-analyze")
            .version(env!("CARGO_PKG_VERSION"))
            .author(env!("CARGO_PKG_AUTHORS"))
            .about("streamlab-analyze is a tool to analyze Lola specifications")
            .arg(Arg::with_name("v").short("v").multiple(true).required(false).help("Sets the level of verbosity"))
            .arg(Arg::with_name("INPUT").help("Sets the input file to use").required(true).index(1))
            .subcommand(SubCommand::with_name("parse").about("Parses the input file and outputs parse tree"))
            .subcommand(
                SubCommand::with_name("ast")
                    .about("Parses the input file and outputs internal representation of abstract syntax tree"),
            )
            .subcommand(
                SubCommand::with_name("pretty-print")
                    .about("Parses the input file and outputs pretty printed representation"),
            )
            .subcommand(SubCommand::with_name("analyze").about("Parses the input file and runs semantic analysis"))
            .subcommand(
                SubCommand::with_name("ir").about("Parses the input file and returns the intermediate representation"),
            )
            .get_matches_from(args);

        let verbosity = match matches.occurrences_of("v") {
            0 => LevelFilter::Warn,
            1 => LevelFilter::Info,
            2 => LevelFilter::Debug,
            3 | _ => LevelFilter::Trace,
        };

        let filename = matches.value_of("INPUT").map(std::string::ToString::to_string).unwrap();
        eprintln!("Input file `{}`", filename);

        let mut logger: Vec<Box<dyn SharedLogger>> = Vec::new();
        if let Some(term_logger) =
            TermLogger::new(verbosity, simplelog::Config::default(), simplelog::TerminalMode::default())
        {
            logger.push(term_logger);
        } else {
            logger.push(SimpleLogger::new(verbosity, simplelog::Config::default()))
        }

        CombinedLogger::init(logger).expect("failed to initialize logging framework");

        match matches.subcommand() {
            ("parse", Some(_)) => Config { which: Analysis::Parse, filename },
            ("ast", Some(_)) => Config { which: Analysis::AST, filename },
            ("pretty-print", Some(_)) => Config { which: Analysis::Prettyprint, filename },
            ("analyze", Some(_)) => Config { which: Analysis::Analyze, filename },
            ("ir", Some(_)) | ("intermediate-representation", Some(_)) => Config { which: Analysis::IR, filename },
            ("", None) => {
                // default to `analyze`
                Config { which: Analysis::Analyze, filename }
            }
            _ => unreachable!(),
        }
    }

    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        let mut file = File::open(&self.filename)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let mapper = SourceMapper::new(PathBuf::from(&self.filename), &contents);
        let handler = Handler::new(mapper);
        match &self.which {
            Analysis::Parse => {
                let result = LolaParser::parse(Rule::Spec, &contents).unwrap_or_else(|e| {
                    eprintln!("parse error:\n{}", e);
                    std::process::exit(1)
                });
                println!("{:#?}", result);
                Ok(())
            }
            Analysis::AST => {
                let spec = crate::parse::parse(&contents, &handler, self::CONFIG).unwrap_or_else(|e| {
                    eprintln!("parse error:\n{}", e);
                    std::process::exit(1)
                });
                println!("{:#?}", spec);
                Ok(())
            }
            Analysis::Prettyprint => {
                let spec = crate::parse::parse(&contents, &handler, self::CONFIG).unwrap_or_else(|e| {
                    eprintln!("parse error:\n{}", e);
                    std::process::exit(1)
                });
                println!("{}", spec);
                Ok(())
            }
            Analysis::Analyze => {
                let spec = crate::parse::parse(&contents, &handler, self::CONFIG).unwrap_or_else(|e| {
                    eprintln!("parse error:\n{}", e);
                    std::process::exit(1)
                });
                let report = analysis::analyze(&spec, &handler, self::CONFIG);
                //println!("{:?}", report);
                use crate::analysis::graph_based_analysis::MemoryBound;
                if let Some(r) = report.graph_analysis_result {
                    match r.memory_requirements {
                        MemoryBound::Unbounded => println!("The specification has no bound on the memory consumption."),
                        MemoryBound::Bounded(bytes) => println!("The specification uses at most {} bytes.", bytes),
                        MemoryBound::Unknown => {
                            println!("Incomplete specification: we cannot determine the memory consumption.")
                        }
                    }
                };
                Ok(())
            }
            Analysis::IR => {
                let spec = crate::parse::parse(&contents, &handler, self::CONFIG).unwrap_or_else(|e| {
                    eprintln!("parse error:\n{}", e);
                    std::process::exit(1)
                });

                let analysis_result = crate::analysis::analyze(&spec, &handler, self::CONFIG);
                if !analysis_result.is_success() {
                    return Ok(()); // TODO throw a good `Error`
                }
                assert!(analysis_result.is_success());
                let ir = Lowering::new(&spec, &analysis_result).lower();
                println!("{:#?}", ir);
                Ok(())
            }
        }
    }
}
