
pub struct EvalConfig {
    source: InputSource,
    verbosity: Verbosity,
    output_channel: OutputChannel,
}

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

pub enum OutputChannel {
    StdOut,
    StdErr,
    File(String),
}

pub enum InputSource {
    StdIn,
    File(String),
}

impl Default for EvalConfig {
    fn default() -> EvalConfig {
        EvalConfig {
            source: InputSource::StdIn,
            verbosity: Verbosity::TriggersOnly,
            output_channel: OutputChannel::StdOut,
        }
    }
}

impl EvalConfig {
    fn print_outputs(mut self) -> Self {
        self.verbosity = Verbosity::Outputs;
        self
    }

    fn debug_mode(mut self) -> Self {
        self.verbosity = Verbosity::Debug;
        self
    }

    fn print_triggers_only(mut self) -> Self {
        self.verbosity = Verbosity::TriggersOnly;
        self
    }

    fn with_input_file(mut self, path: &str) -> Self {
        self.source = InputSource::File(String::from(path));
        self
    }

    fn with_std_input(mut self) -> Self {
        self.source = InputSource::StdIn;
        self
    }

    fn with_std_out(mut self) -> Self {
        self.output_channel = OutputChannel::StdOut;
        self
    }

    fn with_std_err(mut self) -> Self {
        self.output_channel = OutputChannel::StdErr;
        self
    }

    fn with_output_file(mut self, path: &str) -> Self {
        self.output_channel = OutputChannel::File(String::from(path));
        self
    }
}