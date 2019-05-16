use super::{InputSource, OutputChannel};

#[derive(Clone, Debug)]
pub struct EvalConfig {
    pub source: InputSource,
    pub verbosity: Verbosity,
    pub output_channel: OutputChannel,
    pub closure_based_evaluator: bool,
    pub offline: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Verbosity {
    /// Suppresses any kind of logging.
    Silent,
    /// Prints statistical information like number of events, triggers, etc.
    Progress,
    /// Prints nothing but runtime warnings about potentially critical states, e.g. dropped
    /// evaluation cycles.
    WarningsOnly,
    /// Prints only triggers and runtime warnings.
    Triggers,
    /// Prints information about all or a subset of output streams whenever they produce a new
    /// value.
    Outputs,
    /// Prints fine-grained debug information. Not suitable for production.
    Debug,
}

impl EvalConfig {
    pub fn new(
        source: InputSource,
        verbosity: Verbosity,
        output: OutputChannel,
        closure_based_evaluator: bool,
        offline: bool,
    ) -> Self {
        EvalConfig { source, verbosity, output_channel: output, closure_based_evaluator, offline }
    }

    pub fn debug() -> Self {
        let mut cfg = EvalConfig::default();
        cfg.verbosity = Verbosity::Debug;
        cfg
    }

    pub fn release(path: String, output: OutputChannel, closure_based_evaluator: bool, offline: bool) -> Self {
        EvalConfig::new(InputSource::for_file(path), Verbosity::Triggers, output, closure_based_evaluator, offline)
    }
}

impl Default for EvalConfig {
    fn default() -> EvalConfig {
        EvalConfig {
            source: InputSource::StdIn,
            verbosity: Verbosity::Triggers,
            output_channel: OutputChannel::StdOut,
            closure_based_evaluator: true,
            offline: true,
        }
    }
}
