use std::env;
use std::error::Error;

#[cfg(feature = "public")]
use human_panic::setup_panic;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "public")]
    {
        setup_panic!(Metadata {
            name: env!("CARGO_PKG_NAME").into(),
            version: env!("CARGO_PKG_VERSION").into(),
            authors: "StreamLAB Team <stream-lab@react.uni-saarland.de>".into(),
            homepage: "www.stream-lab.org".into(),
        });
    }

    let args: Vec<String> = env::args().collect();

    let config = streamlab_evaluator::Config::new(&args);
    config.run()?;
    Ok(())
}
