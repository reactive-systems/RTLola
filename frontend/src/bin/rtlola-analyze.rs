use rtlola_frontend::app;
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Improved backtraces
    color_backtrace::install();

    let args: Vec<String> = env::args().collect();

    app::analyze::runAnalysisCLI(&args)
}
