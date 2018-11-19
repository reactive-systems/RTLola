#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum BuiltinType {
    Int(u8),
    UInt(u8),
    Float(u8),
    String,
    Bool,
    InvocationTime,
}

impl BuiltinType {
    pub fn all() -> Vec<(&'static str, BuiltinType)> {
        vec![
            ("Int8", BuiltinType::Int(8)),
            ("Int16", BuiltinType::Int(16)),
            ("Int32", BuiltinType::Int(32)),
            ("Int64", BuiltinType::Int(64)),
            ("UInt8", BuiltinType::UInt(8)),
            ("UInt16", BuiltinType::UInt(16)),
            ("UInt32", BuiltinType::UInt(32)),
            ("UInt64", BuiltinType::UInt(64)),
            ("Float32", BuiltinType::Float(32)),
            ("Float64", BuiltinType::Float(64)),
            ("String", BuiltinType::String),
            ("Bool", BuiltinType::Bool),
            ("InvocationTime", BuiltinType::InvocationTime),
        ]
    }
}

#[derive(Debug)]
pub enum Type {
    BuiltIn(BuiltinType),
    Tuple(Vec<BuiltinType>),
    //    Composite(Vec<(String, Box<Type>)>),
}

// These MUST all be lowercase
// TODO add an static assertion for this
pub(crate) const KEYWORDS: [&str; 28] = [
    "input",
    "output",
    "trigger",
    "type",
    "self",
    "include",
    "invoke",
    "inv",
    "extend",
    "ext",
    "terminate",
    "ter",
    "unless",
    "if",
    "then",
    "else",
    "and",
    "or",
    "not",
    "forall",
    "exists",
    "any",
    "true",
    "false",
    "arithmetic_error",
    "error",
    "overflow_error",
    "conversion_error",
];
