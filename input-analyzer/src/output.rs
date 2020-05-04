use serde::{Serialize, Serializer};

#[derive(Serialize)]
pub(crate) struct Output {
    pub(crate) hash: String,
    pub(crate) file_type: FileType,
    pub(crate) possible_clocks: Vec<String>,
    pub(crate) available_streams: Vec<StreamInfo>,
}

#[derive(Serialize)]
pub(crate) struct StreamInfo {
    pub(crate) name: String,
    pub(crate) possible_type: Vec<StreamDataType>,
}
pub(crate) enum FileType {
    CSV,
    PCAP,
    UNKNOWN,
}
#[derive(Eq, PartialEq)]
pub(crate) enum StreamDataType {
    BOOL,
    STRING,
    INT64,
    UINT64,
    #[allow(dead_code)]
    INT32,
    #[allow(dead_code)]
    UINT32,
    #[allow(dead_code)]
    INT16,
    #[allow(dead_code)]
    UINT16,
    #[allow(dead_code)]
    INT8,
    #[allow(dead_code)]
    UINT8,
    #[allow(dead_code)]
    FLOAT32,
    FLOAT64,
    TUPLE(Vec<StreamDataType>),
}

impl Serialize for FileType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            FileType::CSV => serializer.serialize_str("csv"),
            FileType::PCAP => serializer.serialize_str("pcap"),
            FileType::UNKNOWN => serializer.serialize_str("unknown"),
            #[allow(unreachable_patterns)]
            _ => unreachable!("unknown new file format"),
        }
    }
}

impl Serialize for StreamDataType {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        match self {
            StreamDataType::BOOL => serializer.serialize_str("Bool"),
            StreamDataType::STRING => serializer.serialize_str("String"),
            StreamDataType::INT64 => serializer.serialize_str("Int64"),
            StreamDataType::UINT64 => serializer.serialize_str("UInt64"),
            StreamDataType::INT32 => serializer.serialize_str("Int32"),
            StreamDataType::UINT32 => serializer.serialize_str("UInt32"),
            StreamDataType::INT16 => serializer.serialize_str("Int16"),
            StreamDataType::UINT16 => serializer.serialize_str("UInt16"),
            StreamDataType::INT8 => serializer.serialize_str("Int8"),
            StreamDataType::UINT8 => serializer.serialize_str("UInt8"),
            StreamDataType::FLOAT32 => serializer.serialize_str("Float32"),
            StreamDataType::FLOAT64 => serializer.serialize_str("Float64"),
            StreamDataType::TUPLE(entries) => {
                let entries: Vec<String> = entries
                    .iter()
                    .map(|t| match t {
                        StreamDataType::BOOL => "Bool".to_string(),
                        StreamDataType::STRING => "String".to_string(),
                        StreamDataType::INT64 => "Int64".to_string(),
                        StreamDataType::UINT64 => "UInt64".to_string(),
                        StreamDataType::INT32 => "Int32".to_string(),
                        StreamDataType::UINT32 => "UInt32".to_string(),
                        StreamDataType::INT16 => "Int16".to_string(),
                        StreamDataType::UINT16 => "UInt16".to_string(),
                        StreamDataType::INT8 => "Int8".to_string(),
                        StreamDataType::UINT8 => "UInt8".to_string(),
                        StreamDataType::FLOAT32 => "Float32".to_string(),
                        StreamDataType::FLOAT64 => "Float64".to_string(),
                        StreamDataType::TUPLE(_) => unreachable!(),
                    })
                    .collect();
                let result = format!("({})", entries.join(", "));
                serializer.serialize_str(result.as_str())
            }
        }
    }
}
