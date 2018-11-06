//! This module contains the logic for the `lola-analyze` binary.

use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::process;

use clap::{App, Arg, SubCommand};
use pest::Parser;

use super::super::analysis;
use super::super::parse::{LolaParser, Rule};

enum Analysis {
    Parse,
    AST,
    Prettyprint,
}

pub struct Config {
    which: Analysis,
    filename: String,
}

impl Config {
    pub fn new(args: &[String]) -> Self {
        let matches = App::new("lola-analyze")
            .version(env!("CARGO_PKG_VERSION"))
            .author(env!("CARGO_PKG_AUTHORS"))
            .about("lola-anlyze is a tool to parse, type check, and analyze Lola specifications")
            .subcommand(
                SubCommand::with_name("parse")
                    .about("Parses the input file and outputs parse tree")
                    .arg(
                        Arg::with_name("INPUT")
                            .help("Sets the input file to use")
                            .required(true)
                            .index(1),
                    ),
            ).subcommand(
                SubCommand::with_name("ast")
                    .about("Parses the input file and outputs internal representation of abstract syntax tree")
                    .arg(
                        Arg::with_name("INPUT")
                            .help("Sets the input file to use")
                            .required(true)
                            .index(1),
                    ),
            ).subcommand(
                SubCommand::with_name("pretty-print")
                    .about("Parses the input file and outputs pretty printed representation")
                    .arg(
                        Arg::with_name("INPUT")
                            .help("Sets the input file to use")
                            .required(true)
                            .index(1),
                    ),
            ).get_matches_from(args);

        match matches.subcommand() {
            ("parse", Some(parse_matches)) => {
                // Now we have a reference to clone's matches
                let filename = parse_matches
                    .value_of("INPUT")
                    .map(|s| s.to_string())
                    .unwrap();
                eprintln!("Input file `{}`", filename);

                Config {
                    which: Analysis::Parse,
                    filename,
                }
            }
            ("ast", Some(parse_matches)) => {
                // Now we have a reference to clone's matches
                let filename = parse_matches
                    .value_of("INPUT")
                    .map(|s| s.to_string())
                    .unwrap();
                eprintln!("Input file `{}`", filename);

                Config {
                    which: Analysis::AST,
                    filename,
                }
            }
            ("pretty-print", Some(parse_matches)) => {
                // Now we have a reference to clone's matches
                let filename = parse_matches
                    .value_of("INPUT")
                    .map(|s| s.to_string())
                    .unwrap();
                eprintln!("Input file `{}`", filename);

                Config {
                    which: Analysis::Prettyprint,
                    filename,
                }
            }
            ("", None) => {
                println!("No subcommand was used");
                println!("{}", matches.usage());

                process::exit(1);
            }
            _ => unreachable!(),
        }
    }

    pub fn run(&self) -> Result<(), Box<Error>> {
        let mut file = File::open(&self.filename)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        match &self.which {
            Analysis::Parse => {
                let result =
                    LolaParser::parse(Rule::Spec, &contents).unwrap_or_else(|e| panic!("{}", e));
                println!("{:#?}", result);
                Ok(())
            }
            Analysis::AST => {
                let spec = crate::parse::parse(&contents).unwrap_or_else(|e| panic!("{}", e));
                println!("{:#?}", spec);
                Ok(())
            }
            Analysis::Prettyprint => {
                let spec = crate::parse::parse(&contents).unwrap_or_else(|e| panic!("{}", e));
                println!("{}", spec);
                Ok(())
            }
        }
    }
}
