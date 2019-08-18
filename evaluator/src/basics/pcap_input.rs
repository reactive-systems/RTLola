#![allow(clippy::mutex_atomic)]

use crate::basics::io_handler::{EventSource, Time};
use crate::storage::Value;
use etherparse::{Ethernet2Header, InternetSlice, Ipv4Header, Ipv6Header, LinkSlice, SlicedPacket};
use pcap::{Activated, Capture, Device, Packet as PCAPPacket, Precision};
use std::error::Error;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use streamlab_frontend::ir::LolaIR;

// ################################
// Event Source Handling
// ################################
#[derive(Debug, Clone)]
pub enum PCAPInputSource {
    Device { name: String },
    File { path: String, delay: Option<Duration> },
}

enum TimeHandling {
    RealTime { start: Instant },
    FromFile { start: Option<SystemTime> },
    Delayed { delay: Duration, time: Time },
}

pub struct PCAPEventSource {
    capture_handle: Capture<Activated>,
    timer: TimeHandling,
    mapping: Vec<fn(&SlicedPacket) -> Value>,

    event: Option<(Vec<Value>, Time)>,
    last_timestamp: Option<SystemTime>,
}

impl PCAPEventSource {
    pub(crate) fn setup(
        src: &PCAPInputSource,
        ir: &LolaIR,
        start_time: Instant,
    ) -> Result<Box<dyn EventSource>, Box<dyn Error>> {
        let capture_handle = match src {
            PCAPInputSource::Device { name } => {
                let all_devices = Device::list()?;
                let dev: Device = all_devices.into_iter().filter(|d| d.name == *name).nth(0).unwrap_or_else(|| {
                    eprintln!("Could not find network interface with name: {}", *name);
                    std::process::exit(1);
                });

                let capture_handle =
                    Capture::from_device(dev)?.promisc(true).snaplen(65535).precision(Precision::Nano).open()?;
                capture_handle.into()
            }
            PCAPInputSource::File { path, .. } => {
                let capture_handle = Capture::from_file(path)?;
                capture_handle.into()
            }
        };

        use TimeHandling::*;
        let timer = match src {
            PCAPInputSource::Device { .. } => RealTime { start: start_time },
            PCAPInputSource::File { delay, .. } => match delay {
                Some(d) => Delayed { delay: *d, time: Duration::default() },
                None => FromFile { start: None },
            },
        };
        let input_names: Vec<String> = ir.inputs.iter().map(|i| i.name.clone()).collect();

        // Generate Mapping that given a parsed packet returns the value for the corresponding input stream
        let mut mapping: Vec<fn(&SlicedPacket) -> Value> = Vec::with_capacity(input_names.len());
        // TODO: provide mapping

        Ok(Box::new(PCAPEventSource { capture_handle, timer, mapping, event: None, last_timestamp: None }))
    }

    fn process_packet(&mut self) -> Result<bool, Box<dyn Error>> {
        unimplemented!();
    }
}

impl EventSource for PCAPEventSource {
    fn has_event(&mut self) -> bool {
        self.process_packet().unwrap_or_else(|e| {
            eprintln!("error: failed to process packet. {}", e);
            std::process::exit(1)
        })
    }

    fn get_event(&mut self) -> (Vec<Value>, Time) {
        if let Some((event, t)) = &self.event {
            (event.clone(), *t)
        } else {
            eprintln!("No event available!");
            std::process::exit(1);
        }
    }

    fn read_time(&self) -> Option<SystemTime> {
        self.last_timestamp
    }
}
