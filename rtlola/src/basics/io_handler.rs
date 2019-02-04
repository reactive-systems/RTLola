use super::{EvalConfig, Verbosity};
use csv::{Reader as CSVReader, Result as ReaderResult, StringRecord};
use std::fs::File;
use std::io::{stderr, stdin, stdout, Read, Write};

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

struct ColumnMapping(Vec<usize>);

impl ColumnMapping {
    fn from_header(names: &[&str], header: &StringRecord) -> ColumnMapping {
        assert_eq!(names.len(), header.len()); // TODO: Handle better.
        let mapping: Vec<usize> = header
            .iter()
            .map(|name| {
                names
                    .iter()
                    .position(|i| i == &name)
                    .expect(&format!("CVS header does not contain an entry for stream {}.", name))
            })
            .collect();
        ColumnMapping(mapping)
    }

    fn stream_ix_for_col_id(&self, col_id: usize) -> usize {
        self.0[col_id]
    }

    fn num_streams(&self) -> usize {
        self.0.len()
    }
}

enum ReaderWrapper {
    Std(CSVReader<std::io::Stdin>),
    File(CSVReader<File>),
}

impl ReaderWrapper {
    fn read_record(&mut self, rec: &mut StringRecord) -> ReaderResult<bool> {
        match self {
            ReaderWrapper::Std(r) => r.read_record(rec),
            ReaderWrapper::File(r) => r.read_record(rec),
        }
    }

    fn get_header(&mut self) -> ReaderResult<&StringRecord> {
        match self {
            ReaderWrapper::Std(r) => r.headers(),
            ReaderWrapper::File(r) => r.headers(),
        }
    }
}

pub(crate) struct InputReader {
    reader: ReaderWrapper,
    mapping: ColumnMapping,
    record: StringRecord,
}

impl InputReader {
    pub(crate) fn from(src: InputSource, names: &[&str]) -> ReaderResult<InputReader> {
        let mut wrapper = match src {
            InputSource::StdIn => ReaderWrapper::Std(CSVReader::from_reader(stdin())),
            InputSource::File(path) => ReaderWrapper::File(CSVReader::from_path(path)?),
        };

        let mapping = ColumnMapping::from_header(names, wrapper.get_header()?);

        Ok(InputReader { reader: wrapper, mapping, record: StringRecord::new() })
    }

    pub(crate) fn read_blocking(&mut self, buffer: &mut [String]) -> ReaderResult<()> {
        assert_eq!(buffer.len(), self.mapping.num_streams());

        assert!(self.reader.read_record(&mut self.record)?);
        if cfg!(debug_assertion) {
            // Reset all buffered strings.
            buffer.iter_mut().for_each(|v| *v = String::new());
        }

        assert_eq!(self.record.len(), self.mapping.num_streams());

        self.record.iter().enumerate().for_each(|(ix, str_val)| {
            let stream_ix = self.mapping.stream_ix_for_col_id(ix);
            buffer[stream_ix] = String::from(str_val);
        });

        if cfg!(debug_assertion) {
            // Reset all buffered strings.
            assert!(buffer.iter().all(|v| !v.is_empty())) // TODO Runtime Error.
        }

        Ok(())
    }
}

pub(crate) struct OutputHandler {
    pub(crate) verbosity: Verbosity,
    channel: OutputChannel,
    file: Option<File>,
}

impl OutputHandler {
    pub(crate) fn new(config: &EvalConfig) -> OutputHandler {
        OutputHandler { verbosity: config.verbosity, channel: config.output_channel.clone(), file: None }
    }

    pub(crate) fn runtime_warning<F, T: Into<String>>(&self, msg: F)
    where
        F: FnOnce() -> T,
    {
        self.emit(Verbosity::WarningsOnly, msg);
    }

    pub(crate) fn trigger<F, T: Into<String>>(&self, msg: F)
    where
        F: FnOnce() -> T,
    {
        self.emit(Verbosity::Triggers, msg);
    }

    pub(crate) fn debug<F, T: Into<String>>(&self, msg: F)
    where
        F: FnOnce() -> T,
    {
        self.emit(Verbosity::Debug, msg);
    }

    pub(crate) fn output<F, T: Into<String>>(&self, msg: F)
    where
        F: FnOnce() -> T,
    {
        self.emit(Verbosity::Outputs, msg);
    }

    /// Accepts a message and forwards it to the appropriate output channel.
    /// If the configuration prohibits printing the message, `msg` is never called.
    fn emit<F, T: Into<String>>(&self, kind: Verbosity, msg: F)
    where
        F: FnOnce() -> T,
    {
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
