#![allow(clippy::mutex_atomic)]

use super::{EvalConfig, TimeFormat, TimeRepresentation, Verbosity};
use crate::basics::{CSVEventSource, CSVInputSource, PCAPEventSource, PCAPInputSource, Time};
use crate::storage::Value;
use crossterm::{cursor, terminal, ClearType};
use rtlola_frontend::ir::RTLolaIR;
use std::error::Error;
use std::fs::File;
use std::io::{stderr, stdout, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

//Input Handling

#[derive(Debug, Clone)]
pub enum EventSourceConfig {
    CSV { src: CSVInputSource },
    PCAP { src: PCAPInputSource },
    API,
}

/// A trait that represents the functioniality needed for an event source.
/// The order in which the functions are called is:
/// has_event -> get_event -> read_time
pub(crate) trait EventSource {
    /// Returns true if another event can be obtained from the Event Source
    fn has_event(&mut self) -> bool;

    /// Returns an Event consisting of a vector of Values for the input streams and the time passed since the start of the evaluation
    fn get_event(&mut self) -> (Vec<Value>, Time);

    /// Returns the Unix timestamp of the current event
    fn read_time(&self) -> Option<SystemTime>;
}

pub(crate) fn create_event_source(
    config: EventSourceConfig,
    ir: &RTLolaIR,
    start_time: Instant,
) -> Result<Box<dyn EventSource>, Box<dyn Error>> {
    use EventSourceConfig::*;
    match config {
        CSV { src } => CSVEventSource::setup(&src, ir, start_time),
        PCAP { src } => PCAPEventSource::setup(&src, ir, start_time),
        API => unimplemented!("Currently, there is no need to create an event source for the API."),
    }
}

// Output Handling

#[derive(Debug, Clone)]
pub enum OutputChannel {
    StdOut,
    StdErr,
    File(String),
    None,
}

#[derive(Debug)]
pub struct OutputHandler {
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
                msg().into()
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
        let _ = match self.channel {
            OutputChannel::StdOut => stdout().write((msg + "\n").as_bytes()),
            OutputChannel::StdErr => stderr().write((msg + "\n").as_bytes()),
            OutputChannel::File(_) => self.file.as_ref().unwrap().write(msg.as_bytes()),
            OutputChannel::None => Ok(0),
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

#[derive(Debug)]
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

#[derive(Debug, Clone)]
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
            let mut spinner = "⠁⠁⠉⠙⠚⠒⠂⠂⠒⠲⠴⠤⠄⠄⠤⠠⠠⠤⠦⠖⠒⠐⠐⠒⠓⠋⠉⠈⠈ ".chars().cycle();
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
        let mut out = stderr();

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
