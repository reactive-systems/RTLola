use color_backtrace;
use std::env;
use std::process;
use streamlab_frontend::app;

fn main() {
    // Improved backtraces
    color_backtrace::install();

    let args: Vec<String> = env::args().collect();

    let config = app::analyze::Config::new(&args);
    config.run().unwrap_or_else(|err| {
        eprintln!("Problem while executing lola-analyze: {}", err);
        process::exit(1);
    });
}
