#[derive(AsStaticStr, Debug, Eq, PartialEq, Clone, Copy, Hash, EnumIter)]
pub enum BuiltinType {
    #[strum(serialize = "Int8")]
    Int8,
    #[strum(serialize = "Int16")]
    Int16,
    #[strum(serialize = "Int32")]
    Int32,
    #[strum(serialize = "Int64")]
    Int64,
    #[strum(serialize = "UInt8")]
    UInt8,
    #[strum(serialize = "UInt16")]
    UInt16,
    #[strum(serialize = "UInt32")]
    UInt32,
    #[strum(serialize = "UInt64")]
    UInt64,
    #[strum(serialize = "Bool")]
    Bool,
    #[strum(serialize = "Float32")]
    Float32,
    #[strum(serialize = "Float64")]
    Float64,
    #[strum(serialize = "String")]
    String,
}
