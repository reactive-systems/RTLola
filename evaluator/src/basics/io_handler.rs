#![allow(clippy::mutex_atomic)]

use super::{EvalConfig, TimeFormat, TimeRepresentation, Verbosity};
use crate::storage::Value;
use crossterm::{cursor, terminal, ClearType};
use csv::{Reader as CSVReader, Result as ReaderResult, StringRecord};
use std::fs::File;
use std::io::{stderr, stdin, stdout, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use streamlab_frontend::ir::{LolaIR, Type};

pub type Time = Duration;

#[derive(Debug, Clone)]
pub enum OutputChannel {
    StdOut,
    StdErr,
    File(String),
}

#[derive(Debug, Clone)]
pub enum InputSource {
    StdIn,
    File { path: String, delay: Option<Duration>, time_col: Option<usize> },
}

impl InputSource {
    pub fn file(path: String, delay: Option<Duration>, time_col: Option<usize>) -> InputSource {
        InputSource::File { path, delay, time_col }
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

enum TimeHandling {
    RealTime { start: Instant },
    FromFile { start: Option<SystemTime> },
    Delayed { delay: Duration, time: Time },
}

// TODO(marvin): this can be a trait
pub(crate) struct EventSource {
    reader: ReaderWrapper,
    record: StringRecord,
    mapping: ColumnMapping,
    in_types: Vec<Type>,
    timer: TimeHandling,
}

impl EventSource {
    pub(crate) fn from(src: &InputSource, ir: &LolaIR, start_time: Instant) -> ReaderResult<EventSource> {
        use InputSource::*;
        let (mut wrapper, time_col) = match src {
            StdIn => (ReaderWrapper::Std(CSVReader::from_reader(stdin())), None),
            File { path, delay: _, time_col } => (ReaderWrapper::File(CSVReader::from_path(path)?), *time_col),
        };

        let stream_names: Vec<&str> = ir.inputs.iter().map(|i| i.name.as_str()).collect();
        let mapping = ColumnMapping::from_header(stream_names.as_slice(), wrapper.get_header()?, time_col);
        let in_types: Vec<Type> = ir.inputs.iter().map(|i| i.ty.clone()).collect();

        use TimeHandling::*;
        let timer = match src {
            StdIn => RealTime { start: start_time },
            File { path: _, delay, time_col: _ } => match delay {
                Some(d) => Delayed { delay: *d, time: Duration::default() },
                None => FromFile { start: None },
            },
        };

        Ok(EventSource { reader: wrapper, record: StringRecord::new(), mapping, in_types, timer })
    }

    pub(crate) fn read_blocking(&mut self) -> ReaderResult<bool> {
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

    pub(crate) fn has_event(&mut self) -> bool {
        self.read_blocking().unwrap_or_else(|e| panic!("Error reading data. {}", e))
    }

    pub(crate) fn get_event(&mut self) -> (Vec<Value>, Time) {
        let event = self.read_event();
        let time = self.get_time();
        (event, time)
    }

    fn get_time(&mut self) -> Time {
        use TimeHandling::*;
        match self.timer {
            RealTime { start } => Instant::now() - start,
            FromFile { start } => {
                let now = self.read_time().unwrap();
                match start {
                    None => {
                        self.timer = FromFile { start: Some(now) };
                        Time::default()
                    }
                    Some(start) => now.duration_since(start).expect("Time did not behave monotonically!"),
                }
            }
            Delayed { delay, ref mut time } => {
                *time += delay;
                *time
            }
        }
    }

    fn read_event(&self) -> Vec<Value> {
        let mut buffer = vec![Value::None; self.in_types.len()];
        for (col_ix, s) in self.record.iter().enumerate() {
            if let Some(str_ix) = self.mapping.col2str[col_ix] {
                if s != "#" {
                    let t = &self.in_types[str_ix];
                    buffer[str_ix] = Value::try_from(s, t)
                        .unwrap_or_else(|| panic!("Failed to parse {} as value of type {:?}.", s, t))
                }
            }
        }
        buffer
    }

    pub fn read_time(&self) -> Option<SystemTime> {
        let time_str = self.str_for_time();
        if time_str.is_none() {
            return None;
        }
        let time_str = time_str.unwrap();
        let mut time_str_split = time_str.split('.');
        let secs_str: &str = match time_str_split.next() {
            Some(s) => s,
            None => panic!("Failed to parse time string {}.", time_str),
        };
        let secs = match secs_str.parse::<u64>() {
            Ok(u) => u,
            Err(e) => panic!("Failed to parse time string {}: {}", time_str, e),
        };
        let d: Duration = if let Some(nanos_str) = time_str_split.next() {
            let mut chars = nanos_str.chars();
            let mut nanos: u32 = 0;
            for _ in 1..=9 {
                nanos *= 10;
                if let Some(c) = chars.next() {
                    if let Some(d) = c.to_digit(10) {
                        nanos += d;
                    }
                }
            }
            assert!(time_str_split.next().is_none());
            Duration::new(secs, nanos)
        } else {
            Duration::from_nanos(secs)
        };
        Some(UNIX_EPOCH + d)
    }

    pub(crate) fn str_for_time(&self) -> Option<&str> {
        self.time_index().map(|ix| &self.record[ix])
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
    pub(crate) start_time: Mutex<SystemTime>,
    time_representation: TimeRepresentation,
}

impl OutputHandler {
    pub(crate) fn new(config: &EvalConfig, num_trigger: usize) -> OutputHandler {
        let statistics = if config.verbosity == Verbosity::Progress {
            let stats = Statistics::new(num_trigger);
            stats.start_print_progress();
            Some(stats)
        } else if config.statistics == crate::basics::Statistics::Debug {
            Some(Statistics::new(num_trigger))
        } else {
            None
        };
        OutputHandler {
            verbosity: config.verbosity,
            channel: config.output_channel.clone(),
            file: None,
            statistics,
            start_time: Mutex::new(SystemTime::now()),
            time_representation: config.time_presentation,
        }
    }

    pub(crate) fn runtime_warning<F, T: Into<String>>(&self, msg: F)
    where
        F: FnOnce() -> T,
    {
        self.emit(Verbosity::WarningsOnly, msg);
    }

    fn time_info(&self, time: Time) -> Option<String> {
        use TimeFormat::*;
        use TimeRepresentation::*;
        match self.time_representation {
            Hide => None,
            Relative(format) => {
                let d = time;
                match format {
                    UIntNanos => Some(format!("{}", d.as_nanos())),
                    FloatSecs => Some(format!("{}.{:09}", d.as_secs(), d.subsec_nanos())),
                    HumanTime => Some(format!("{}", humantime::format_duration(d))),
                }
            }
            Absolute(format) => {
                let mut d = time;
                d += self
                    .start_time
                    .lock()
                    .unwrap()
                    .duration_since(UNIX_EPOCH)
                    .expect("Computation of duration failed!");
                match format {
                    UIntNanos => Some(format!("{}", d.as_nanos())),
                    FloatSecs => Some(format!("{}.{:09}", d.as_secs(), d.subsec_nanos())),
                    HumanTime => {
                        let ts = UNIX_EPOCH + d;
                        Some(format!("{}", humantime::format_rfc3339(ts)))
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn trigger<F, T: Into<String>>(&self, msg: F, trigger_idx: usize, time: Time)
    where
        F: FnOnce() -> T,
    {
        let msg = || {
            if let Some(ti) = self.time_info(time) {
                format!("{}: {}", ti, msg().into())
            } else {
                format!("{}", msg().into())
            }
        };
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
            if self.verbosity == Verbosity::Progress {
                statistics.terminate();
            }
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
        Statistics { data }
    }

    fn start_print_progress(&self) {
        // print intitial info
        Self::print_progress_info(&self.data, ' ');
        let copy = self.data.clone();
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
