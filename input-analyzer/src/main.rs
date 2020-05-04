use std::error::Error;
use std::fs::File;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use pcap::Capture;

use crate::output::{FileType, StreamDataType, StreamInfo};
use orion::hazardous::hash::sha512::Sha512;
use std::io::Read;

mod output;

fn available_types() -> Vec<StreamDataType> {
    vec![
        StreamDataType::BOOL,
        StreamDataType::STRING,
        StreamDataType::INT64,
        StreamDataType::UINT64,
        // StreamDataType::INT32,
        // StreamDataType::UINT32,
        // StreamDataType::INT16,
        // StreamDataType::UINT16,
        // StreamDataType::INT8,
        // StreamDataType::UINT8,
        // StreamDataType::FLOAT32,
        StreamDataType::FLOAT64,
    ]
}

fn try_to_analyze_as_pcap() -> Result<bool, Box<dyn Error>> {
    let cap = Capture::from_file("input");
    if cap.is_err() {
        return Ok(false);
    }
    std::mem::drop(cap.unwrap());
    let mut buffer = vec![];
    let mut file = File::open("input").unwrap();
    file.read_to_end(&mut buffer)?;
    std::mem::drop(file);
    let hash = Sha512::digest(buffer.as_slice())?;
    let hash_hex = hex::encode(hash);
    let file_result = output::Output {
        hash: hash_hex,
        file_type: FileType::PCAP,
        possible_clocks: vec![],
        available_streams: vec![
            StreamInfo {
                name: "Ethernet::destination".to_string(),
                possible_type: vec![StreamDataType::TUPLE(vec![
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                ])],
                // possible_type: vec![StreamDataType::TUPLE(vec![
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                // ])],
            },
            StreamInfo {
                name: "Ethernet::source".to_string(),
                possible_type: vec![StreamDataType::TUPLE(vec![
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                ])],
                // possible_type: vec![StreamDataType::TUPLE(vec![
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                // ])],
            },
            StreamInfo { name: "Ethernet::type".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "Ethernet::type".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo {
                name: "IPv4::source".to_string(),
                possible_type: vec![StreamDataType::TUPLE(vec![
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                ])],
                // possible_type: vec![StreamDataType::TUPLE(vec![
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                // ])],
            },
            StreamInfo {
                name: "IPv4::destination".to_string(),
                possible_type: vec![StreamDataType::TUPLE(vec![
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                ])],
                // possible_type: vec![StreamDataType::TUPLE(vec![
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                // ])],
            },
            StreamInfo { name: "IPv4::ihl".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv4::ihl".to_string(), possible_type: vec![StreamDataType::UINT8] },
            StreamInfo { name: "IPv4::dscp".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv4::dscp".to_string(), possible_type: vec![StreamDataType::UINT8] },
            StreamInfo { name: "IPv4::ecn".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv4::ecn".to_string(), possible_type: vec![StreamDataType::UINT8] },
            StreamInfo { name: "IPv4::length".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv4::length".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "IPv4::identification".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv4::identification".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "IPv4::fragment_offset".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv4::fragment_offset".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "IPv4::ttl".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv4::ttl".to_string(), possible_type: vec![StreamDataType::UINT8] },
            StreamInfo { name: "IPv4::protocol".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv4::protocol".to_string(), possible_type: vec![StreamDataType::UINT8] },
            StreamInfo { name: "IPv4::checksum".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv4::checksum".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "IPv4::flags::df".to_string(), possible_type: vec![StreamDataType::BOOL] },
            StreamInfo { name: "IPv4::flags::mf".to_string(), possible_type: vec![StreamDataType::BOOL] },
            StreamInfo {
                name: "IPv6::source".to_string(),
                possible_type: vec![StreamDataType::TUPLE(vec![
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                ])],
                // possible_type: vec![StreamDataType::TUPLE(vec![
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                // ])],
            },
            StreamInfo {
                name: "IPv6::destination".to_string(),
                possible_type: vec![StreamDataType::TUPLE(vec![
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                    StreamDataType::UINT64,
                ])],
                // possible_type: vec![StreamDataType::TUPLE(vec![
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                //     StreamDataType::UINT8,
                // ])],
            },
            StreamInfo { name: "IPv6::traffic_class".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv6::traffic_class".to_string(), possible_type: vec![StreamDataType::UINT8] },
            StreamInfo { name: "IPv6::flow_label".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv6::flow_label".to_string(), possible_type: vec![StreamDataType::UINT32] },
            StreamInfo { name: "IPv6::length".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv6::length".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "IPv6::hop_limit".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "IPv6::hop_limit".to_string(), possible_type: vec![StreamDataType::UINT8] },
            StreamInfo { name: "TCP::source".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "TCP::source".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "TCP::destination".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "TCP::destination".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "TCP::seq_number".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "TCP::seq_number".to_string(), possible_type: vec![StreamDataType::UINT32] },
            StreamInfo { name: "TCP::ack_number".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "TCP::ack_number".to_string(), possible_type: vec![StreamDataType::UINT32] },
            StreamInfo { name: "TCP::data_offset".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "TCP::data_offset".to_string(), possible_type: vec![StreamDataType::UINT8] },
            StreamInfo { name: "TCP::window_size".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "TCP::window_size".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "TCP::checksum".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "TCP::checksum".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "TCP::urgent_pointer".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "TCP::urgent_pointer".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "TCP::flags::ns".to_string(), possible_type: vec![StreamDataType::BOOL] },
            StreamInfo { name: "TCP::flags::cwr".to_string(), possible_type: vec![StreamDataType::BOOL] },
            StreamInfo { name: "TCP::flags::ece".to_string(), possible_type: vec![StreamDataType::BOOL] },
            StreamInfo { name: "TCP::flags::urg".to_string(), possible_type: vec![StreamDataType::BOOL] },
            StreamInfo { name: "TCP::flags::ack".to_string(), possible_type: vec![StreamDataType::BOOL] },
            StreamInfo { name: "TCP::flags::psh".to_string(), possible_type: vec![StreamDataType::BOOL] },
            StreamInfo { name: "TCP::flags::rst".to_string(), possible_type: vec![StreamDataType::BOOL] },
            StreamInfo { name: "TCP::flags::syn".to_string(), possible_type: vec![StreamDataType::BOOL] },
            StreamInfo { name: "TCP::flags::fin".to_string(), possible_type: vec![StreamDataType::BOOL] },
            StreamInfo { name: "UDP::source".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "UDP::source".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "UDP::destination".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "UDP::destination".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "UDP::length".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "UDP::length".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "UDP::checksum".to_string(), possible_type: vec![StreamDataType::UINT64] },
            // StreamInfo { name: "UDP::checksum".to_string(), possible_type: vec![StreamDataType::UINT16] },
            StreamInfo { name: "payload".to_string(), possible_type: vec![StreamDataType::STRING] },
            StreamInfo { name: "direction".to_string(), possible_type: vec![StreamDataType::STRING] },
            StreamInfo { name: "protocol".to_string(), possible_type: vec![StreamDataType::STRING] },
        ],
    };

    let j = serde_json::to_string(&file_result)?;
    let mut output = File::create("result")?;
    use std::io::Write;
    write!(output, "{}", j)?;
    return Ok(true);
}

fn can_parse_as(content: &str, possible_type: &StreamDataType) -> bool {
    if content == "0.0" {
        if *possible_type == StreamDataType::BOOL {
            return content.parse::<bool>().is_ok();
        }
        return true;
    }
    if content == "#" {
        return true;
    }
    match possible_type {
        StreamDataType::BOOL => content.parse::<bool>().is_ok(),
        StreamDataType::STRING => true,
        StreamDataType::INT64 => content.parse::<i64>().is_ok(),
        StreamDataType::UINT64 => content.parse::<u64>().is_ok(),
        StreamDataType::INT32 => content.parse::<i32>().is_ok(),
        StreamDataType::UINT32 => content.parse::<u32>().is_ok(),
        StreamDataType::INT16 => content.parse::<i16>().is_ok(),
        StreamDataType::UINT16 => content.parse::<u16>().is_ok(),
        StreamDataType::INT8 => content.parse::<i8>().is_ok(),
        StreamDataType::UINT8 => content.parse::<u8>().is_ok(),
        StreamDataType::FLOAT32 => content.parse::<f32>().is_ok(),
        StreamDataType::FLOAT64 => content.parse::<f64>().is_ok(),
        StreamDataType::TUPLE(_) => unreachable!(),
    }
}

fn read_time(content: &str) -> Result<SystemTime, ()> {
    match content.parse::<u64>() {
        Ok(nano_seconds) => Ok(UNIX_EPOCH + Duration::from_nanos(nano_seconds)),
        Err(_) => match content.parse::<f64>() {
            Ok(secs) => Ok(UNIX_EPOCH + Duration::from_secs_f64(secs)),
            Err(_) => match content.parse::<DateTime<Utc>>() {
                Ok(dt) => Ok(UNIX_EPOCH + Duration::from_nanos(dt.timestamp() as u64)),
                Err(_) => Err(()),
            },
        },
    }
}

fn try_to_analyze_as_csv() -> Result<bool, Box<dyn Error>> {
    let mut buffer = vec![];
    let mut file = File::open("input").unwrap();
    file.read_to_end(&mut buffer)?;
    std::mem::drop(file);
    let hash = Sha512::digest(buffer.as_slice())?;
    let hash_hex = hex::encode(hash);
    let csv_reader = csv::Reader::from_path("input");
    if csv_reader.is_err() {
        return Ok(false);
    }
    let mut csv_reader = csv_reader.unwrap();
    let mut file_result =
        output::Output { hash: hash_hex, file_type: FileType::CSV, possible_clocks: vec![], available_streams: vec![] };
    let headers = csv_reader.headers();
    if headers.is_err() {
        return Ok(false);
    }
    let headers = headers.unwrap();
    let mut time_info: Vec<Result<Option<SystemTime>, ()>> = vec![];
    for header in headers {
        file_result.available_streams.push(StreamInfo { name: header.to_string(), possible_type: available_types() });
        time_info.push(Ok(None))
    }

    for line in csv_reader.records() {
        if line.is_err() {
            return Ok(false);
        }
        let line = line.unwrap();
        if line.len() != file_result.available_streams.len() {
            return Ok(false);
        }
        for (index, content) in line.iter().enumerate() {
            file_result.available_streams[index]
                .possible_type
                .retain(|possible_type| can_parse_as(content, possible_type));
            if let Ok(previous) = time_info[index].clone() {
                let current = read_time(content);
                match current {
                    Ok(now) => match previous {
                        None => time_info[index] = Ok(Some(now)),
                        Some(other) => {
                            if other < now {
                                time_info[index] = Ok(Some(now))
                            } else {
                                time_info[index] = Err(())
                            }
                        }
                    },
                    Err(_) => time_info[index] = Err(()),
                }
            }
        }
    }
    for (index, stream) in file_result.available_streams.iter().enumerate() {
        if let Ok(_) = &time_info[index] {
            file_result.possible_clocks.push(stream.name.clone())
        }
    }
    let j = serde_json::to_string(&file_result)?;
    let mut output = File::create("result")?;
    use std::io::Write;
    write!(output, "{}", j)?;
    return Ok(true);
}

fn main() -> Result<(), Box<dyn Error>> {
    let is_pcap = try_to_analyze_as_pcap()?;
    if is_pcap {
        return Ok(());
    }
    let is_csv = try_to_analyze_as_csv()?;
    if is_csv {
        return Ok(());
    }

    let mut buffer = vec![];
    let mut file = File::open("input").unwrap();
    file.read_to_end(&mut buffer)?;
    std::mem::drop(file);
    let hash = Sha512::digest(buffer.as_slice())?;
    let hash_hex = hex::encode(hash);

    let file_result = output::Output {
        hash: hash_hex,
        file_type: FileType::UNKNOWN,
        possible_clocks: vec![],
        available_streams: vec![],
    };

    let j = serde_json::to_string_pretty(&file_result)?;
    print!("{}", j);
    Ok(())
}
