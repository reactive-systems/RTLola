use super::{CSVInputSource, EventSourceConfig, OutputChannel};

#[derive(Clone, Debug)]
pub struct EvalConfig {
    pub source: EventSourceConfig,
    pub statistics: Statistics,
    pub verbosity: Verbosity,
    pub output_channel: OutputChannel,
    pub evaluator: EvaluatorChoice,
    pub mode: ExecutionMode,
    pub time_presentation: TimeRepresentation,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Statistics {
    None,
    Debug,
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExecutionMode {
    Offline,
    Online,
    API,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EvaluatorChoice {
    ClosureBased,
    Interpreted,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeRepresentation {
    Hide,
    Relative(TimeFormat),
    Absolute(TimeFormat),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeFormat {
    UIntNanos,
    FloatSecs,
    HumanTime,
}

impl EvalConfig {
    pub fn new(
        source: EventSourceConfig,
        statistics: Statistics,
        verbosity: Verbosity,
        output: OutputChannel,
        evaluator: EvaluatorChoice,
        mode: ExecutionMode,
        time_presentation: TimeRepresentation,
    ) -> Self {
        EvalConfig { source, statistics, verbosity, output_channel: output, evaluator, mode, time_presentation }
    }

    pub fn debug() -> Self {
        let mut cfg = EvalConfig::default();
        cfg.statistics = Statistics::Debug;
        cfg.verbosity = Verbosity::Debug;
        cfg
    }

    pub fn release(
        path: String,
        output: OutputChannel,
        evaluator: EvaluatorChoice,
        mode: ExecutionMode,
        time_presentation: TimeRepresentation,
    ) -> Self {
        EvalConfig::new(
            EventSourceConfig::CSV { src: CSVInputSource::file(path, None, None) },
            Statistics::None,
            Verbosity::Triggers,
            output,
            evaluator,
            mode,
            time_presentation,
        )
    }

    pub fn api(time_representation: TimeRepresentation) -> Self {
        EvalConfig::new(
            EventSourceConfig::API,
            Statistics::None,
            Verbosity::Triggers,
            OutputChannel::None,
            EvaluatorChoice::ClosureBased,
            ExecutionMode::API,
            time_representation,
        )
    }
}

impl Default for EvalConfig {
    fn default() -> EvalConfig {
        EvalConfig {
            source: EventSourceConfig::CSV { src: CSVInputSource::StdIn },
            statistics: Statistics::None,
            verbosity: Verbosity::Triggers,
            output_channel: OutputChannel::StdOut,
            evaluator: EvaluatorChoice::ClosureBased,
            mode: ExecutionMode::Offline,
            time_presentation: TimeRepresentation::Hide,
        }
    }
}
