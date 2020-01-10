mod config;
mod csv_input;
mod io_handler;
mod pcap_input;

pub type Time = Duration;

pub use self::config::{
    EvalConfig, EvaluatorChoice, ExecutionMode, Statistics, TimeFormat, TimeRepresentation, Verbosity,
};
pub(crate) use self::io_handler::{create_event_source, EventSource, EventSourceConfig, OutputHandler};
pub use self::io_handler::OutputChannel;

pub use self::csv_input::{CSVEventSource, CSVInputSource};

pub use self::pcap_input::{PCAPEventSource, PCAPInputSource};
use std::time::Duration;
