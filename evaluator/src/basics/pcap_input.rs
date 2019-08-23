#![allow(clippy::mutex_atomic)]

use crate::basics::io_handler::{EventSource, Time};
use crate::storage::Value;
use etherparse::{
    Ethernet2Header, InternetSlice, Ipv4Header, Ipv6Header, LinkSlice, SlicedPacket, TcpHeader, TransportSlice,
    UdpHeader,
};
use pcap::{Activated, Capture, Device, Packet as PCAPPacket, Precision};
use std::error::Error;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use streamlab_frontend::ir::LolaIR;

// ################################
// Packet parsing functions
// ################################

//ethernet functions
fn get_ethernet_header(packet: &SlicedPacket) -> Option<Ethernet2Header> {
    match &packet.link {
        Some(s) => match s {
            LinkSlice::Ethernet2(hdr) => Some(hdr.to_header()),
        },
        None => None,
    }
}

fn ethernet_source(packet: &SlicedPacket) -> Value {
    if let Some(ethernet_header) = get_ethernet_header(packet) {
        let values: Vec<Value> = ethernet_header.source.iter().map(|v: &u8| Value::Unsigned((*v).into())).collect();
        Value::Tuple(values.into_boxed_slice())
    } else {
        Value::None
    }
}
fn ethernet_destination(packet: &SlicedPacket) -> Value {
    if let Some(ethernet_header) = get_ethernet_header(packet) {
        let values: Vec<Value> =
            ethernet_header.destination.iter().map(|v: &u8| Value::Unsigned((*v).into())).collect();
        Value::Tuple(values.into_boxed_slice())
    } else {
        Value::None
    }
}
fn ethernet_type(packet: &SlicedPacket) -> Value {
    if let Some(ethernet_header) = get_ethernet_header(packet) {
        Value::Unsigned(ethernet_header.ether_type.into())
    } else {
        Value::None
    }
}

//Ipv4 functions
fn get_ipv4_header(packet: &SlicedPacket) -> Option<Ipv4Header> {
    match &packet.ip {
        Some(int_slice) => match int_slice {
            InternetSlice::Ipv4(h) => Some(h.to_header()),
            InternetSlice::Ipv6(_, _) => None,
        },
        None => None,
    }
}

fn ipv4_source(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        let values: Vec<Value> = header.source.iter().map(|v: &u8| Value::Unsigned((*v).into())).collect();
        Value::Tuple(values.into_boxed_slice())
    } else {
        Value::None
    }
}
fn ipv4_destination(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        let values: Vec<Value> = header.destination.iter().map(|v: &u8| Value::Unsigned((*v).into())).collect();
        Value::Tuple(values.into_boxed_slice())
    } else {
        Value::None
    }
}
fn ipv4_ihl(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        Value::Unsigned(header.ihl().into())
    } else {
        Value::None
    }
}
fn ipv4_dscp(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        Value::Unsigned(header.differentiated_services_code_point.into())
    } else {
        Value::None
    }
}
fn ipv4_ecn(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        Value::Unsigned(header.explicit_congestion_notification.into())
    } else {
        Value::None
    }
}
fn ipv4_length(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        Value::Unsigned(header.total_len().into())
    } else {
        Value::None
    }
}
fn ipv4_id(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        Value::Unsigned(header.identification.into())
    } else {
        Value::None
    }
}
fn ipv4_fragment_offset(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        Value::Unsigned(header.fragments_offset.into())
    } else {
        Value::None
    }
}
fn ipv4_ttl(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        Value::Unsigned(header.time_to_live.into())
    } else {
        Value::None
    }
}
fn ipv4_protocol(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        Value::Unsigned(header.protocol.into())
    } else {
        Value::None
    }
}
fn ipv4_checksum(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        Value::Unsigned(header.header_checksum.into())
    } else {
        Value::None
    }
}
fn ipv4_flags_df(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        Value::Bool(header.dont_fragment)
    } else {
        Value::None
    }
}
fn ipv4_flags_mf(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv4_header(packet) {
        Value::Bool(header.more_fragments)
    } else {
        Value::None
    }
}

//IPv6 functions
fn get_ipv6_header(packet: &SlicedPacket) -> Option<Ipv6Header> {
    match &packet.ip {
        Some(int_slice) => match int_slice {
            InternetSlice::Ipv4(_) => None,
            InternetSlice::Ipv6(h, _) => Some(h.to_header()),
        },
        None => None,
    }
}

fn ipv6_source(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv6_header(packet) {
        let values: Vec<Value> = header.source.iter().map(|v: &u8| Value::Unsigned((*v).into())).collect();
        Value::Tuple(values.into_boxed_slice())
    } else {
        Value::None
    }
}
fn ipv6_destination(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv6_header(packet) {
        let values: Vec<Value> = header.destination.iter().map(|v: &u8| Value::Unsigned((*v).into())).collect();
        Value::Tuple(values.into_boxed_slice())
    } else {
        Value::None
    }
}
fn ipv6_traffic_class(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv6_header(packet) {
        Value::Unsigned(header.traffic_class.into())
    } else {
        Value::None
    }
}
fn ipv6_flow_label(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv6_header(packet) {
        Value::Unsigned(header.flow_label.into())
    } else {
        Value::None
    }
}
fn ipv6_length(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv6_header(packet) {
        Value::Unsigned(header.payload_length.into())
    } else {
        Value::None
    }
}
fn ipv6_hop_limit(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_ipv6_header(packet) {
        Value::Unsigned(header.hop_limit.into())
    } else {
        Value::None
    }
}

//TCP functions
fn get_tcp_header(packet: &SlicedPacket) -> Option<TcpHeader> {
    use TransportSlice::*;
    match &packet.transport {
        Some(t) => match t {
            Tcp(h) => Some(h.to_header()),
            Udp(_) => None,
        },
        None => None,
    }
}

fn tcp_source_port(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Unsigned(header.source_port.into())
    } else {
        Value::None
    }
}
fn tcp_destination_port(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Unsigned(header.destination_port.into())
    } else {
        Value::None
    }
}
fn tcp_seq_number(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Unsigned(header.sequence_number.into())
    } else {
        Value::None
    }
}
fn tcp_ack_number(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Unsigned(header.acknowledgment_number.into())
    } else {
        Value::None
    }
}
fn tcp_data_offset(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Unsigned(header.data_offset().into())
    } else {
        Value::None
    }
}
fn tcp_flags_ns(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Bool(header.ns)
    } else {
        Value::None
    }
}
fn tcp_flags_fin(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Bool(header.fin)
    } else {
        Value::None
    }
}
fn tcp_flags_syn(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Bool(header.syn)
    } else {
        Value::None
    }
}
fn tcp_flags_rst(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Bool(header.rst)
    } else {
        Value::None
    }
}
fn tcp_flags_psh(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Bool(header.psh)
    } else {
        Value::None
    }
}
fn tcp_flags_ack(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Bool(header.ack)
    } else {
        Value::None
    }
}
fn tcp_flags_urg(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Bool(header.urg)
    } else {
        Value::None
    }
}
fn tcp_flags_ece(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Bool(header.ece)
    } else {
        Value::None
    }
}
fn tcp_flags_cwr(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Bool(header.cwr)
    } else {
        Value::None
    }
}
fn tcp_window_size(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Unsigned(header.window_size.into())
    } else {
        Value::None
    }
}
fn tcp_checksum(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Unsigned(header.checksum.into())
    } else {
        Value::None
    }
}
fn tcp_urgent_pointer(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_tcp_header(packet) {
        Value::Unsigned(header.urgent_pointer.into())
    } else {
        Value::None
    }
}

//UDP functions
fn get_udp_header(packet: &SlicedPacket) -> Option<UdpHeader> {
    use TransportSlice::*;
    match &packet.transport {
        Some(t) => match t {
            Tcp(_) => None,
            Udp(h) => Some(h.to_header()),
        },
        None => None,
    }
}
fn udp_source_port(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_udp_header(packet) {
        Value::Unsigned(header.source_port.into())
    } else {
        Value::None
    }
}
fn udp_destination_port(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_udp_header(packet) {
        Value::Unsigned(header.destination_port.into())
    } else {
        Value::None
    }
}
fn udp_length(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_udp_header(packet) {
        Value::Unsigned(header.length.into())
    } else {
        Value::None
    }
}
fn udp_checksum(packet: &SlicedPacket) -> Value {
    if let Some(header) = get_udp_header(packet) {
        Value::Unsigned(header.checksum.into())
    } else {
        Value::None
    }
}

//Misc Packet Functions
fn get_packet_payload(packet: &SlicedPacket) -> Value {
    Value::Str(String::from_utf8_lossy(packet.payload).as_ref().into())
}

fn get_packet_protocol(packet: &SlicedPacket) -> Value {
    match &packet.transport {
        Some(transport) => match transport {
            TransportSlice::Tcp(_) => Value::Str("TCP".into()),
            TransportSlice::Udp(_) => Value::Str("UDP".into()),
        },
        None => match &packet.ip {
            Some(ip) => match ip {
                InternetSlice::Ipv4(_) => Value::Str("IPv4".into()),
                InternetSlice::Ipv6(_, _) => Value::Str("IPv6".into()),
            },
            None => match &packet.link {
                Some(link) => match link {
                    LinkSlice::Ethernet2(_) => Value::Str("Ethernet2".into()),
                },
                None => Value::Str("Unknown".into()),
            },
        },
    }
}

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
