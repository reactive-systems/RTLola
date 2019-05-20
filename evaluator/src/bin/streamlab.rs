use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    let config = streamlab_evaluator::Config::new(&args);
    config.run()?;
    Ok(())
}
