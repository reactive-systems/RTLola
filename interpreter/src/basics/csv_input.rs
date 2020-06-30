#![allow(clippy::mutex_atomic)]

use crate::basics::io_handler::EventSource;
use crate::basics::Time;
use crate::storage::Value;
use csv::{ByteRecord, Reader as CSVReader, Result as ReaderResult, StringRecord};
use rtlola_frontend::ir::{RTLolaIR, Type};
use std::error::Error;
use std::fs::File;
use std::io::stdin;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
enum TimeHandling {
    RealTime { start: Instant },
    FromFile { start: Option<SystemTime> },
    Delayed { delay: Duration, time: Time },
}

#[derive(Debug, Clone)]
pub enum CSVInputSource {
    StdIn,
    File { path: String, delay: Option<Duration>, time_col: Option<usize> },
}

impl CSVInputSource {
    pub fn file(path: String, delay: Option<Duration>, time_col: Option<usize>) -> CSVInputSource {
        CSVInputSource::File { path, delay, time_col }
    }

    pub fn stdin() -> CSVInputSource {
        CSVInputSource::StdIn
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CSVColumnMapping {
    /// Mapping from column index to input stream index/reference
    pub(crate) col2str: Vec<Option<usize>>,

    /// Column index of time (if existent)
    time_ix: Option<usize>,
}

impl CSVColumnMapping {
    fn from_header(names: &[&str], header: &StringRecord, time_col: Option<usize>) -> CSVColumnMapping {
        let str2col: Vec<usize> = names
            .iter()
            .map(|name| {
                header.iter().position(|entry| &entry == name).unwrap_or_else(|| {
                    eprintln!("error: CSV header does not contain an entry for stream `{}`.", name);
                    std::process::exit(1)
                })
            })
            .collect();

        let mut col2str: Vec<Option<usize>> = vec![None; header.len()];
        for (str_ix, header_ix) in str2col.iter().enumerate() {
            col2str[*header_ix] = Some(str_ix);
        }

        let time_ix = time_col.map(|col| col - 1).or_else(|| {
            header.iter().position(|name| {
                let name = name.to_lowercase();
                name == "time" || name == "ts" || name == "timestamp"
            })
        });
        CSVColumnMapping { col2str, time_ix }
    }

    fn input_to_stream(&self, input_ix: usize) -> Option<usize> {
        self.col2str[input_ix]
    }

    fn num_inputs(&self) -> usize {
        self.col2str.len()
    }
}

#[derive(Debug)]
enum ReaderWrapper {
    Std(CSVReader<std::io::Stdin>),
    File(CSVReader<File>),
}

impl ReaderWrapper {
    fn read_record(&mut self, rec: &mut ByteRecord) -> ReaderResult<bool> {
        match self {
            ReaderWrapper::Std(r) => r.read_byte_record(rec),
            ReaderWrapper::File(r) => r.read_byte_record(rec),
        }
    }

    fn get_header(&mut self) -> ReaderResult<&StringRecord> {
        match self {
            ReaderWrapper::Std(r) => r.headers(),
            ReaderWrapper::File(r) => r.headers(),
        }
    }
}

#[derive(Debug)]
pub struct CSVEventSource {
    reader: ReaderWrapper,
    record: ByteRecord,
    mapping: CSVColumnMapping,
    in_types: Vec<Type>,
    timer: TimeHandling,
}

impl CSVEventSource {
    pub(crate) fn setup(
        src: &CSVInputSource,
        ir: &RTLolaIR,
        start_time: Instant,
    ) -> Result<Box<dyn EventSource>, Box<dyn Error>> {
        use CSVInputSource::*;
        let (mut wrapper, time_col) = match src {
            StdIn => (ReaderWrapper::Std(CSVReader::from_reader(stdin())), None),
            File { path, time_col, .. } => (ReaderWrapper::File(CSVReader::from_path(path)?), *time_col),
        };

        let stream_names: Vec<&str> = ir.inputs.iter().map(|i| i.name.as_str()).collect();
        let mapping = CSVColumnMapping::from_header(stream_names.as_slice(), wrapper.get_header()?, time_col);
        let in_types: Vec<Type> = ir.inputs.iter().map(|i| i.ty.clone()).collect();

        use TimeHandling::*;
        let timer = match src {
            StdIn => RealTime { start: start_time },
            File { delay, .. } => match delay {
                Some(d) => Delayed { delay: *d, time: Duration::default() },
                None => FromFile { start: None },
            },
        };

        Ok(Box::new(CSVEventSource { reader: wrapper, record: ByteRecord::new(), mapping, in_types, timer }))
    }

    fn read_blocking(&mut self) -> Result<bool, Box<dyn Error>> {
        if cfg!(debug_assertion) {
            // Reset record.
            self.record.clear();
        }
        let read_res = match self.reader.read_record(&mut self.record) {
            Ok(v) => v,
            Err(e) => {
                return Err(e.into());
            }
        };
        if !read_res {
            return Ok(false);
        }
        assert_eq!(self.record.len(), self.mapping.num_inputs());

        //TODO(marvin): this assertion seems wrong, empty strings could be valid values
        if cfg!(debug_assertion) {
            assert!(self
                .record
                .iter()
                .enumerate()
                .filter(|(ix, _)| self.mapping.input_to_stream(*ix).is_some())
                .all(|(_, str)| !str.is_empty()));
        }

        Ok(true)
    }

    pub(crate) fn str_for_time(&self) -> Option<&str> {
        self.time_index().map(|ix| &self.record[ix]).and_then(|bytes| std::str::from_utf8(bytes).ok())
    }

    fn time_index(&self) -> Option<usize> {
        self.mapping.time_ix
    }

    fn get_time(&mut self) -> Time {
        use self::TimeHandling::*;
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
                // utf8-encoding (as [u8]) of string "#"
                if s != [35] {
                    let t = &self.in_types[str_ix];
                    buffer[str_ix] = Value::try_from(s, t).unwrap_or_else(|| {
                        if let Ok(s) = std::str::from_utf8(s) {
                            eprintln!(
                                "error: problem with data source; failed to parse {} as value of type {:?}.",
                                s, t
                            );
                        } else {
                            eprintln!(
                                "error: problem with data source; failed to parse non-utf8 {:?} as value of type {:?}.",
                                s, t
                            );
                        }
                        std::process::exit(1)
                    })
                }
            }
        }
        buffer
    }
}

impl EventSource for CSVEventSource {
    fn has_event(&mut self) -> bool {
        self.read_blocking().unwrap_or_else(|e| {
            eprintln!("error: failed to read data. {}", e);
            std::process::exit(1)
        })
    }

    fn get_event(&mut self) -> (Vec<Value>, Time) {
        let event = self.read_event();
        let time = self.get_time();
        (event, time)
    }

    fn read_time(&self) -> Option<SystemTime> {
        let time_str = self.str_for_time()?;
        let mut time_str_split = time_str.split('.');
        let secs_str: &str = match time_str_split.next() {
            Some(s) => s,
            None => {
                eprintln!("error: problem with data source; failed to parse time string {}.", time_str);
                std::process::exit(1)
            }
        };
        let secs = match secs_str.parse::<u64>() {
            Ok(u) => u,
            Err(e) => {
                eprintln!("error: problem with data source; failed to parse time string {}: {}", time_str, e);
                std::process::exit(1)
            }
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
}
