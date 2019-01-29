use super::{Verbosity, EvalConfig};
use std::io::{stdout, stderr, Write};
use std::fs::File;

#[derive(Debug, Clone)]
pub enum OutputChannel {
    StdOut,
    StdErr,
    File(String),
}

#[derive(Debug, Clone)]
pub enum InputSource {
    StdIn,
    File(String),
}

pub(crate) struct OutputHandler {
    pub(crate) verbosity: Verbosity,
    channel: OutputChannel,
    file: Option<File>
}

impl OutputHandler {
    pub(crate) fn new(config: &EvalConfig) -> OutputHandler {
        OutputHandler {
            verbosity: config.verbosity,
            channel: config.output_channel.clone(),
            file: None,
        }
    }

    pub(crate) fn runtime_warning<F, T: Into<String>>(&self, msg: F) where F: FnOnce() -> T {
        self.emit(Verbosity::WarningsOnly, msg);
    }

    pub(crate) fn trigger<F, T: Into<String>>(&self, msg: F) where F: FnOnce() -> T {
        self.emit(Verbosity::Triggers, msg);
    }

    pub(crate) fn debug<F, T: Into<String>>(&self, msg: F) where F: FnOnce() -> T {
        self.emit(Verbosity::Debug, msg);
    }

    pub(crate) fn output<F, T: Into<String>>(&self, msg: F) where F: FnOnce() -> T {
        self.emit(Verbosity::Outputs, msg);
    }

    /// Accepts a message and forwards it to the appropriate output channel.
    /// If the configuration prohibits printing the message, `msg` is never called.
    fn emit<F, T: Into<String>>(&self, kind: Verbosity, msg: F) where F: FnOnce() -> T, {
        if kind >= self.verbosity {
            self.print(msg().into());
        }
    }

    fn print(&self, msg: String) {
        use crate::basics::OutputChannel;
        let _ = match self.channel {
            OutputChannel::StdOut => stdout().write(msg.as_bytes()),
            OutputChannel::StdErr => stderr().write(msg.as_bytes()),
            OutputChannel::File(_) => self.file.as_ref().unwrap().write(msg.as_bytes()),
        }; // TODO: Decide how to handle the result.
    }
}

impl Default for OutputHandler {
    fn default() -> OutputHandler {
        OutputHandler::new(&EvalConfig::default())
    }
}