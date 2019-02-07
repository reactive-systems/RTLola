use super::{EvalConfig, Verbosity};
use csv::{Reader as CSVReader, Result as ReaderResult, StringRecord};
use itertools::Itertools;
use std::fs::File;
use std::io::{stderr, stdin, stdout, Write};
use std::time::Duration;

#[derive(Debug, Clone)]
pub enum OutputChannel {
    StdOut,
    StdErr,
    File(String),
}

#[derive(Debug, Clone)]
pub enum InputSource {
    StdIn,
    File { path: String, reading_delay: Option<Duration> },
}

impl InputSource {
    pub fn for_file(path: String) -> InputSource {
        InputSource::File { path, reading_delay: None }
    }

    pub fn with_delay(path: String, delay: Duration) -> InputSource {
        InputSource::File { path, reading_delay: Some(delay) }
    }

    pub fn stdin() -> InputSource {
        InputSource::StdIn
    }
}

struct ColumnMapping {
    mapping: Vec<usize>,
    time: Option<usize>,
}

impl ColumnMapping {
    fn from_header(names: &[&str], header: &StringRecord) -> ColumnMapping {
        assert_eq!(names.len(), header.len()); // TODO: Handle better.
        dbg!(names);
        dbg!(header);
        let mapping: Vec<usize> = header
            .iter()
            .map(|name| {
                names
                    .iter()
                    .position(|i| i == &name)
                    .unwrap_or_else(|| panic!("CVS header does not contain an entry for stream {}.", name))
            })
            .collect();
        let time_pos = header.iter().find_position(|name| name == &"time" || name == &"ts" || name == &"timestamp");
        let time = time_pos.map(|(ix, _)| ix);
        ColumnMapping { mapping, time }
    }

    fn stream_ix_for_col_id(&self, col_id: usize) -> usize {
        self.mapping[col_id]
    }

    fn num_streams(&self) -> usize {
        self.mapping.len()
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
    reading_delay: Option<Duration>,
}

impl InputReader {
    pub(crate) fn from(src: InputSource, names: &[&str]) -> ReaderResult<InputReader> {
        let mut delay = None;
        let mut wrapper = match src {
            InputSource::StdIn => ReaderWrapper::Std(CSVReader::from_reader(stdin())),
            InputSource::File { path, reading_delay } => {
                delay = reading_delay;
                ReaderWrapper::File(CSVReader::from_path(path)?)
            }
        };

        let mapping = ColumnMapping::from_header(names, wrapper.get_header()?);

        Ok(InputReader { reader: wrapper, mapping, record: StringRecord::new(), reading_delay: delay })
    }

    pub(crate) fn read_blocking(&mut self, buffer: &mut [String]) -> ReaderResult<bool> {
        assert_eq!(buffer.len(), self.mapping.num_streams());

        if let Some(delay) = self.reading_delay {
            std::thread::sleep(delay);
        }

        if !self.reader.read_record(&mut self.record)? {
            return Ok(false);
        }
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

        Ok(true)
    }

    pub(crate) fn time_index(&self) -> Option<usize> {
        self.mapping.time
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

    #[allow(dead_code)]
    pub(crate) fn trigger<F, T: Into<String>>(&self, msg: F)
    where
        F: FnOnce() -> T,
    {
        self.emit(Verbosity::Triggers, msg);
    }

    #[allow(dead_code)]
    pub(crate) fn debug<F, T: Into<String>>(&self, msg: F)
    where
        F: FnOnce() -> T,
    {
        self.emit(Verbosity::Debug, msg);
    }

    #[allow(dead_code)]
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
        if kind <= self.verbosity {
            self.print(msg().into());
        }
    }

    fn print(&self, msg: String) {
        use crate::basics::OutputChannel;
        let _ = match self.channel {
            OutputChannel::StdOut => stdout().write((msg + "\n").as_bytes()),
            OutputChannel::StdErr => stderr().write((msg + "\n").as_bytes()),
            OutputChannel::File(_) => self.file.as_ref().unwrap().write(msg.as_bytes()),
        }; // TODO: Decide how to handle the result.
    }
}

impl Default for OutputHandler {
    fn default() -> OutputHandler {
        OutputHandler::new(&EvalConfig::default())
    }
}
