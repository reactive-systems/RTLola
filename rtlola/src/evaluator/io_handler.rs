use crate::evaluator::config::{EvalConfig, Verbosity};

pub(crate) struct OutputHandler {
    verbosity: Verbosity,
}

impl OutputHandler {
    pub(crate) fn new(config: &EvalConfig) -> OutputHandler {
        unimplemented!()
    }

    /// Accepts a message and forwards it to the appropriate output channel.
    /// If the configuration prohibits printing the message, `msg` is never called.
    pub(crate) fn emit<T>(&self, kind: OutputKind, msg: T)
    where
        T: FnOnce() -> String,
    {
        match self.verbosity {
            Verbosity::Debug => self.print(msg()),
            Verbosity::Outputs => match kind {
                OutputKind::Trigger | OutputKind::Output | OutputKind::RuntimeWarning => {
                    self.print(msg())
                }
                OutputKind::Debug => {}
            },
            Verbosity::Triggers => match kind {
                OutputKind::Trigger | OutputKind::RuntimeWarning => self.print(msg()),
                OutputKind::Debug | OutputKind::Output => {}
            },
            Verbosity::WarningsOnly => match kind {
                OutputKind::RuntimeWarning => self.print(msg()),
                OutputKind::Output | OutputKind::Debug | OutputKind::Trigger => {}
            },
            Verbosity::Silent => {}
        }
    }

    fn print(&self, msg: String) {
        unimplemented!()
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub(crate) enum OutputKind {
    /// Information about trigger events.
    Trigger,
    /// Information about the values of output streams.
    Output,
    /// Fine-grained debug information.
    Debug,
    /// Warning for problems occurring during runtime such as too high pressure,
    /// invalid inputs, skipped evaluation steps, or high memory consumption.
    RuntimeWarning,
}

pub(crate) struct InputHandler {}

impl InputHandler {
    pub(crate) fn new(config: &EvalConfig) -> InputHandler {
        unimplemented!()
    }
}
