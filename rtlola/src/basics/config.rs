use super::{InputSource, OutputChannel};
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct EvalConfig {
    pub source: InputSource,
    pub verbosity: Verbosity,
    pub output_channel: OutputChannel,
    pub closure_based_evaluator: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verbosity {
    /// Prints fine-grained debug information. Not suitable for production.
    Debug,
    /// Prints information about all or a subset of output streams whenever they produce a new
    /// value.
    Outputs,
    /// Prints only triggers and runtime warnings.
    Triggers,
    /// Prints nothing but runtime warnings about potentially critical states, e.g. dropped
    /// evaluation cycles.
    WarningsOnly,
    /// Suppresses any kind of logging.
    Silent,
}

impl Verbosity {
    fn as_num(self) -> u8 {
        match self {
            Verbosity::Debug => 4,
            Verbosity::Outputs => 3,
            Verbosity::Triggers => 2,
            Verbosity::WarningsOnly => 1,
            Verbosity::Silent => 0,
        }
    }
}

impl PartialOrd for Verbosity {
    fn partial_cmp(&self, other: &Verbosity) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Verbosity {
    fn cmp(&self, other: &Verbosity) -> Ordering {
        self.as_num().cmp(&other.as_num())
    }
}

impl EvalConfig {
    pub fn new(
        source: InputSource,
        verbosity: Verbosity,
        output: OutputChannel,
        closure_based_evaluator: bool,
    ) -> Self {
        EvalConfig { source, verbosity, output_channel: output, closure_based_evaluator }
    }

    pub fn debug() -> Self {
        let mut cfg = EvalConfig::default();
        cfg.verbosity = Verbosity::Debug;
        cfg
    }

    pub fn release(path: String, output: OutputChannel, closure_based_evaluator: bool) -> Self {
        EvalConfig::new(InputSource::for_file(path), Verbosity::Triggers, output, closure_based_evaluator)
    }
}

impl Default for EvalConfig {
    fn default() -> EvalConfig {
        EvalConfig {
            source: InputSource::StdIn,
            verbosity: Verbosity::Triggers,
            output_channel: OutputChannel::StdOut,
            closure_based_evaluator: true,
        }
    }
}
