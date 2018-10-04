extern crate lola_parser;

use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    let config = lola_parser::app::analyze::Config::new(&args);
    config.run().unwrap_or_else(|err| {
        eprintln!("Problem while executing lola-analyze: {}", err);
        process::exit(1);
    });
}
