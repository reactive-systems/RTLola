use super::{EvalConfig, Verbosity};
use csv::{Reader as CSVReader, Result as ReaderResult, StringRecord};
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
    /// Mapping from stream index/reference to input column index
    str2col: Vec<usize>,
    /// Mapping from column index to input stream index/reference
    col2str: Vec<Option<usize>>,

    /// Column index of time (if existent)
    time_ix: Option<usize>,
}

impl ColumnMapping {
    fn from_header(names: &[&str], header: &StringRecord) -> ColumnMapping {
        let str2col: Vec<usize> = names
            .iter()
            .map(|name| {
                header
                    .iter()
                    .position(|entry| &entry == name)
                    .unwrap_or_else(|| panic!("CVS header does not contain an entry for stream {}.", name))
            })
            .collect();

        let mut col2str: Vec<Option<usize>> = vec![None; header.len()];
        for (str_ix, header_ix) in str2col.iter().enumerate() {
            col2str[*header_ix] = Some(str_ix);
        }

        let time_ix = header.iter().position(|name| name == "time" || name == "ts" || name == "timestamp");
        ColumnMapping { str2col, col2str, time_ix }
    }

    fn stream_ix_for_col_ix(&self, col_ix: usize) -> Option<usize> {
        self.col2str[col_ix]
    }

    #[allow(dead_code)]
    fn time_is_stream(&self) -> bool {
        match self.time_ix {
            None => false,
            Some(col_ix) => self.col2str[col_ix].is_some(),
        }
    }

    fn num_columns(&self) -> usize {
        self.col2str.len()
    }

    #[allow(dead_code)]
    fn num_streams(&self) -> usize {
        self.str2col.len()
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

    pub(crate) fn read_blocking(&mut self) -> ReaderResult<bool> {
        if let Some(delay) = self.reading_delay {
            std::thread::sleep(delay);
        }

        if cfg!(debug_assertion) {
            // Reset record.
            self.record.clear();
        }

        if !self.reader.read_record(&mut self.record)? {
            return Ok(false);
        }
        assert_eq!(self.record.len(), self.mapping.num_columns());

        //TODO(marvin): this assertion seems wrong, empty strings could be valid values
        if cfg!(debug_assertion) {
            assert!(self
                .record
                .iter()
                .enumerate()
                .filter(|(ix, _)| self.mapping.stream_ix_for_col_ix(*ix).is_some())
                .all(|(_, str)| !str.is_empty()));
        }

        Ok(true)
    }

    pub(crate) fn str_ref_for_stream_ix(&self, stream_ix: usize) -> &str {
        &self.record[self.mapping.str2col[stream_ix]]
    }

    pub(crate) fn str_ref_for_time(&self) -> &str {
        assert!(self.time_index().is_some());
        &self.record[self.time_index().unwrap()]
    }

    pub(crate) fn time_index(&self) -> Option<usize> {
        self.mapping.time_ix
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
