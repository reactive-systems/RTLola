#![allow(clippy::mutex_atomic)]

use super::{EvalConfig, Verbosity};
use crossterm::{cursor, terminal, ClearType};
use csv::{Reader as CSVReader, Result as ReaderResult, StringRecord};
use std::fs::File;
use std::io::{stderr, stdin, stdout, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone)]
pub enum OutputChannel {
    StdOut,
    StdErr,
    File(String),
}

#[derive(Debug, Clone)]
pub enum InputSource {
    StdIn,
    File { path: String, reading_delay: Option<Duration>, time_col: Option<usize> },
}

impl InputSource {
    pub fn for_file(path: String, time_col: Option<usize>) -> InputSource {
        InputSource::File { path, reading_delay: None, time_col }
    }

    pub fn with_delay(path: String, delay: Duration, time_col: Option<usize>) -> InputSource {
        InputSource::File { path, reading_delay: Some(delay), time_col }
    }

    pub fn stdin() -> InputSource {
        InputSource::StdIn
    }
}

pub(crate) struct ColumnMapping {
    /// Mapping from stream index/reference to input column index
    str2col: Vec<usize>,
    /// Mapping from column index to input stream index/reference
    pub(crate) col2str: Vec<Option<usize>>,

    /// Column index of time (if existent)
    time_ix: Option<usize>,
}

impl ColumnMapping {
    fn from_header(names: &[&str], header: &StringRecord, time_col: Option<usize>) -> ColumnMapping {
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

        let time_ix =
            time_col.or_else(|| header.iter().position(|name| name == "time" || name == "ts" || name == "timestamp"));
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
    pub(crate) mapping: ColumnMapping,
    pub(crate) record: StringRecord,
    reading_delay: Option<Duration>,
}

impl InputReader {
    pub(crate) fn from(src: InputSource, names: &[&str]) -> ReaderResult<InputReader> {
        let mut delay = None;
        let (mut wrapper, time_col) = match src {
            InputSource::StdIn => (ReaderWrapper::Std(CSVReader::from_reader(stdin())), None),
            InputSource::File { path, reading_delay, time_col } => {
                delay = reading_delay;
                (ReaderWrapper::File(CSVReader::from_path(path)?), time_col)
            }
        };

        let mapping = ColumnMapping::from_header(names, wrapper.get_header()?, time_col);

        Ok(InputReader { reader: wrapper, mapping, record: StringRecord::new(), reading_delay: delay })
    }

    pub(crate) fn read_blocking(&mut self) -> ReaderResult<bool> {
        if let Some(delay) = self.reading_delay {
            thread::sleep(delay);
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

    pub(crate) fn str_for_time(&self) -> &str {
        assert!(self.time_index().is_some());
        &self.record[self.time_index().unwrap()]
    }

    fn time_index(&self) -> Option<usize> {
        self.mapping.time_ix
    }
}

pub(crate) struct OutputHandler {
    pub(crate) verbosity: Verbosity,
    channel: OutputChannel,
    file: Option<File>,
    pub(crate) statistics: Option<Statistics>,
}

impl OutputHandler {
    pub(crate) fn new(config: &EvalConfig, num_trigger: usize) -> OutputHandler {
        OutputHandler {
            verbosity: config.verbosity,
            channel: config.output_channel.clone(),
            file: None,
            statistics: if config.verbosity == Verbosity::Progress { Some(Statistics::new(num_trigger)) } else { None },
        }
    }

    pub(crate) fn runtime_warning<F, T: Into<String>>(&self, msg: F)
    where
        F: FnOnce() -> T,
    {
        self.emit(Verbosity::WarningsOnly, msg);
    }

    #[allow(dead_code)]
    pub(crate) fn trigger<F, T: Into<String>>(&self, msg: F, trigger_idx: usize)
    where
        F: FnOnce() -> T,
    {
        self.emit(Verbosity::Triggers, msg);
        if let Some(statistics) = &self.statistics {
            statistics.trigger(trigger_idx);
        }
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

    pub(crate) fn new_event(&self) {
        if let Some(statistics) = &self.statistics {
            statistics.new_event();
        }
    }

    pub(crate) fn terminate(&self) {
        if let Some(statistics) = &self.statistics {
            statistics.terminate();
        }
    }
}

struct StatisticsData {
    start: SystemTime,
    num_events: AtomicU64,
    num_triggers: Vec<AtomicU64>,
    done: Mutex<bool>,
}

impl StatisticsData {
    fn new(num_trigger: usize) -> Self {
        Self {
            start: SystemTime::now(),
            num_events: AtomicU64::new(0),
            num_triggers: (0..num_trigger).map(|_| AtomicU64::new(0)).collect(),
            done: Mutex::new(false),
        }
    }
}

pub(crate) struct Statistics {
    data: Arc<StatisticsData>,
}

impl Statistics {
    fn new(num_trigger: usize) -> Self {
        let data = Arc::new(StatisticsData::new(num_trigger));
        // print intitial info
        Self::print_progress_info(&data, ' ');
        let copy = data.clone();
        thread::spawn(move || {
            // this thread is responsible for displaying progress information
            let mut spinner = "⠁⠁⠉⠙⠚⠒⠂⠂⠒⠲⠴⠤⠄⠄⠤⠠⠠⠤⠦⠖⠒⠐⠐⠒⠓⠋⠉⠈⠈ "
                .chars()
                .cycle();
            loop {
                thread::sleep(Duration::from_millis(100));
                #[allow(clippy::mutex_atomic)]
                let done = copy.done.lock().unwrap();
                if *done {
                    return;
                }
                Self::clear_progress_info();
                Self::print_progress_info(&copy, spinner.next().unwrap());
            }
        });

        Statistics { data }
    }

    fn new_event(&self) {
        self.data.num_events.fetch_add(1, Ordering::Relaxed);
    }

    fn trigger(&self, trigger_idx: usize) {
        self.data.num_triggers[trigger_idx].fetch_add(1, Ordering::Relaxed);
    }

    #[allow(clippy::mutex_atomic)]
    pub(crate) fn terminate(&self) {
        let mut done = self.data.done.lock().unwrap();
        Self::clear_progress_info();
        Self::print_progress_info(&self.data, ' ');
        *done = true;
    }

    fn print_progress_info(data: &Arc<StatisticsData>, spin_char: char) {
        let mut out = std::io::stderr();

        // write event statistics
        let now = SystemTime::now();
        let elapsed_total = now.duration_since(data.start).unwrap().as_nanos();
        let num_events: u128 = data.num_events.load(Ordering::Relaxed).into();
        if num_events > 0 {
            let events_per_second = (num_events * Duration::from_secs(1).as_nanos()) / elapsed_total;
            let nanos_per_event = elapsed_total / num_events;
            writeln!(
                out,
                "{} {} events, {} events per second, {} nsec per event",
                spin_char, num_events, events_per_second, nanos_per_event
            )
            .unwrap_or_else(|_| {});
        } else {
            writeln!(out, "{} {} events", spin_char, num_events).unwrap_or_else(|_| {});
        }

        // write trigger statistics
        let num_triggers =
            data.num_triggers.iter().fold(0, |val, num_trigger| val + num_trigger.load(Ordering::Relaxed));
        writeln!(out, "  {} triggers", num_triggers).unwrap_or_else(|_| {});
    }

    fn clear_progress_info() {
        let terminal = terminal();
        // clear screen as much as written in `print_progress_info`
        cursor().move_up(1);
        terminal.clear(ClearType::CurrentLine).unwrap_or_else(|_| {});
        cursor().move_up(1);
        terminal.clear(ClearType::CurrentLine).unwrap_or_else(|_| {});
    }

    #[cfg(test)]
    pub(crate) fn get_num_trigger(&self, trigger_idx: usize) -> u64 {
        self.data.num_triggers[trigger_idx].fetch_add(1, Ordering::Relaxed)
    }
}
