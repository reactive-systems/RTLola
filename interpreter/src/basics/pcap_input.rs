#![allow(clippy::mutex_atomic)]

use crate::basics::io_handler::EventSource;
use crate::basics::Time;
use crate::storage::Value;
use etherparse::{
    Ethernet2Header, InternetSlice, Ipv4Header, Ipv6Header, LinkSlice, SlicedPacket, TcpHeader, TransportSlice,
    UdpHeader,
};
use ip_network::IpNetwork;
use pcap_on_demand::{Activated, Capture, Device, Error as PCAPError};
use rtlola_frontend::ir::RTLolaIR;
use std::error::Error;
use std::net::IpAddr;
use std::str::FromStr;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
    Device { name: String, local_network: String },
    File { path: String, delay: Option<Duration>, local_network: String },
}

enum TimeHandling {
    RealTime { start: Instant },
    FromFile { start: Option<SystemTime> },
    Delayed { delay: Duration, time: Time },
}

#[allow(missing_debug_implementations)] // Capture -> PcapOnDemand does not implement Debug.
pub struct PCAPEventSource {
    capture_handle: Capture<dyn Activated>,
    timer: TimeHandling,
    mapping: Vec<Box<dyn Fn(&SlicedPacket) -> Value>>,

    event: Option<(Vec<Value>, Time)>,
    last_timestamp: Option<SystemTime>,
}

impl PCAPEventSource {
    pub(crate) fn setup(
        src: &PCAPInputSource,
        ir: &RTLolaIR,
        start_time: Instant,
    ) -> Result<Box<dyn EventSource>, Box<dyn Error>> {
        let capture_handle = match src {
            PCAPInputSource::Device { name, .. } => {
                let all_devices = Device::list()?;
                let dev: Device = all_devices.into_iter().filter(|d| d.name == *name).nth(0).unwrap_or_else(|| {
                    eprintln!("Could not find network interface with name: {}", *name);
                    std::process::exit(1);
                });

                let capture_handle = Capture::from_device(dev)?.promisc(true).snaplen(65535).open()?;
                capture_handle.into()
            }
            PCAPInputSource::File { path, .. } => {
                let capture_handle = Capture::from_file(path)?;
                capture_handle.into()
            }
        };

        let local_network_range = match src {
            PCAPInputSource::Device { local_network, .. } => local_network,
            PCAPInputSource::File { local_network, .. } => local_network,
        };
        let local_network = IpNetwork::from_str(local_network_range.as_ref()).unwrap_or_else(|e| {
            eprintln!("Could not parse local network range: {}. Error: {}", *local_network_range, e);
            std::process::exit(1);
        });

        let get_packet_direction = move |packet: &SlicedPacket| -> Value {
            let addr: IpAddr = match &packet.ip {
                Some(ip) => match ip {
                    InternetSlice::Ipv4(header) => IpAddr::V4(header.destination_addr()),
                    InternetSlice::Ipv6(header, _) => IpAddr::V6(header.destination_addr()),
                },
                None => return Value::None,
            };
            if local_network.contains(addr) {
                Value::Str("Incoming".into())
            } else {
                Value::Str("Outgoing".into())
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
        let mut mapping: Vec<Box<dyn Fn(&SlicedPacket) -> Value>> = Vec::with_capacity(input_names.len());
        for name in input_names.iter() {
            let layers: Vec<&str> = name.split("::").collect();
            if layers.len() > 3 || layers.is_empty() {
                eprintln!("Malformed input name: {}", name);
                std::process::exit(1);
            }

            let val: Box<dyn Fn(&SlicedPacket) -> Value> = match layers[0] {
                "Ethernet" => {
                    if layers.len() != 2 {
                        eprintln!("Malformed input name: {}", name);
                        std::process::exit(1);
                    };
                    match layers[1] {
                        "source" => Box::new(ethernet_source),
                        "destination" => Box::new(ethernet_destination),
                        "type" => Box::new(ethernet_type),
                        _ => {
                            eprintln!("Unknown input name: {}", name);
                            std::process::exit(1);
                        }
                    }
                }
                "IPv4" => {
                    if layers.len() < 2 {
                        eprintln!("Malformed input name: {}", name);
                        std::process::exit(1);
                    };
                    match layers[1] {
                        "source" => Box::new(ipv4_source),
                        "destination" => Box::new(ipv4_destination),
                        "ihl" => Box::new(ipv4_ihl),
                        "dscp" => Box::new(ipv4_dscp),
                        "ecn" => Box::new(ipv4_ecn),
                        "length" => Box::new(ipv4_length),
                        "identification" => Box::new(ipv4_id),
                        "fragment_offset" => Box::new(ipv4_fragment_offset),
                        "ttl" => Box::new(ipv4_ttl),
                        "protocol" => Box::new(ipv4_protocol),
                        "checksum" => Box::new(ipv4_checksum),
                        "flags" => {
                            if layers.len() < 3 {
                                eprintln!("Malformed input name: {}", name);
                                std::process::exit(1);
                            }
                            match layers[2] {
                                "df" => Box::new(ipv4_flags_df),
                                "mf" => Box::new(ipv4_flags_mf),
                                _ => {
                                    eprintln!("Unknown input name: {}", name);
                                    std::process::exit(1);
                                }
                            }
                        }
                        _ => {
                            eprintln!("Unknown input name: {}", name);
                            std::process::exit(1);
                        }
                    }
                }
                "IPv6" => {
                    if layers.len() < 2 {
                        eprintln!("Malformed input name: {}", name);
                        std::process::exit(1);
                    };
                    match layers[1] {
                        "source" => Box::new(ipv6_source),
                        "destination" => Box::new(ipv6_destination),
                        "traffic_class" => Box::new(ipv6_traffic_class),
                        "flow_label" => Box::new(ipv6_flow_label),
                        "length" => Box::new(ipv6_length),
                        "hop_limit" => Box::new(ipv6_hop_limit),
                        _ => {
                            eprintln!("Unknown input name: {}", name);
                            std::process::exit(1);
                        }
                    }
                }
                //"ICMP" => {},
                "TCP" => {
                    if layers.len() < 2 {
                        eprintln!("Malformed input name: {}", name);
                        std::process::exit(1);
                    };
                    match layers[1] {
                        "source" => Box::new(tcp_source_port),
                        "destination" => Box::new(tcp_destination_port),
                        "seq_number" => Box::new(tcp_seq_number),
                        "ack_number" => Box::new(tcp_ack_number),
                        "data_offset" => Box::new(tcp_data_offset),
                        "window_size" => Box::new(tcp_window_size),
                        "checksum" => Box::new(tcp_checksum),
                        "urgent_pointer" => Box::new(tcp_urgent_pointer),
                        "flags" => {
                            if layers.len() < 3 {
                                eprintln!("Malformed input name: {}", name);
                                std::process::exit(1);
                            };
                            match layers[2] {
                                "ns" => Box::new(tcp_flags_ns),
                                "fin" => Box::new(tcp_flags_fin),
                                "syn" => Box::new(tcp_flags_syn),
                                "rst" => Box::new(tcp_flags_rst),
                                "psh" => Box::new(tcp_flags_psh),
                                "ack" => Box::new(tcp_flags_ack),
                                "urg" => Box::new(tcp_flags_urg),
                                "ece" => Box::new(tcp_flags_ece),
                                "cwr" => Box::new(tcp_flags_cwr),
                                _ => {
                                    eprintln!("Unknown input name: {}", name);
                                    std::process::exit(1);
                                }
                            }
                        }
                        _ => {
                            eprintln!("Unknown input name: {}", name);
                            std::process::exit(1);
                        }
                    }
                }
                "UDP" => {
                    if layers.len() < 2 {
                        eprintln!("Malformed input name: {}", name);
                        std::process::exit(1);
                    };
                    match layers[1] {
                        "source" => Box::new(udp_source_port),
                        "destination" => Box::new(udp_destination_port),
                        "length" => Box::new(udp_length),
                        "checksum" => Box::new(udp_checksum),
                        _ => {
                            eprintln!("Unknown input name: {}", name);
                            std::process::exit(1);
                        }
                    }
                }
                "payload" => Box::new(get_packet_payload),
                "direction" => Box::new(get_packet_direction),
                "protocol" => Box::new(get_packet_protocol),
                _ => {
                    eprintln!("Unknown input name: {}", name);
                    std::process::exit(1);
                }
            };
            mapping.push(val);
        }

        Ok(Box::new(PCAPEventSource { capture_handle, timer, mapping, event: None, last_timestamp: None }))
    }

    fn process_packet(&mut self) -> Result<bool, Box<dyn Error>> {
        let raw_packet = match self.capture_handle.next() {
            Ok(pkt) => pkt,
            Err(e) => match e {
                PCAPError::NoMorePackets => return Ok(false),
                _ => return Err(e.into()),
            },
        };

        use std::convert::TryInto;
        let d = Duration::new(
            raw_packet.header.ts.tv_sec.try_into().unwrap(),
            (raw_packet.header.ts.tv_usec * 1000).try_into().unwrap(),
        );
        self.last_timestamp = Some(UNIX_EPOCH + d);

        let p = SlicedPacket::from_ethernet(raw_packet.data);
        //Todo (Florian): Track underlying error
        if p.is_err() {
            return Ok(false);
        }
        let packet = p.unwrap();

        let mut event: Vec<Value> = Vec::with_capacity(self.mapping.len());
        for parse_function in self.mapping.iter() {
            event.push(parse_function(&packet));
        }

        //compute duration since start
        use TimeHandling::*;
        let dur: Time = match self.timer {
            RealTime { start } => Instant::now() - start,
            FromFile { start } => {
                let now = self.last_timestamp.unwrap();
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
        };

        self.event = Some((event, dur));

        Ok(true)
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
