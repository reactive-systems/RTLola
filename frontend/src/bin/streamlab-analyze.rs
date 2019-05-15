use color_backtrace;
use std::env;
use std::error::Error;
use streamlab_frontend::app;

fn main() -> Result<(), Box<dyn Error>> {
    // Improved backtraces
    color_backtrace::install();

    let args: Vec<String> = env::args().collect();

    let config = app::analyze::Config::new(&args);
    config.run()
}
