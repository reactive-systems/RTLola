use ordered_float::NotNan;
use std::ops;
use streamlab_frontend::ir::Type;

use self::Value::*;

#[derive(Debug, PartialEq, Eq, PartialOrd, Hash, Clone)]
pub(crate) enum Value {
    None,
    Unsigned(u128),
    Signed(i128),
    Float(NotNan<f64>),
    Bool(bool),
    #[allow(dead_code)]
    Str(String),
}

impl Value {
    // TODO: -> Result<Option, ConversionError>
    pub(crate) fn try_from(source: &str, ty: &Type) -> Option<Value> {
        match ty {
            Type::Option(_) | Type::Function(_, _) => panic!("Cannot occur."),
            Type::String => unimplemented!(),
            Type::Tuple(_) => unimplemented!(),
            Type::Float(_) => source.parse::<f64>().ok().map(|f| Float(NotNan::new(f).unwrap())),
            Type::UInt(_) => {
                // TODO: This is just a quickfix!! Think of something more general.
                if source == "0.0" {
                    Some(Unsigned(0))
                } else {
                    source.parse::<u128>().map(Unsigned).ok()
                }
            }
            Type::Int(_) => source.parse::<i128>().map(Signed).ok(),
            Type::Bool => source.parse::<bool>().map(Bool).ok(),
        }
    }

    pub(crate) fn new_float(f: f64) -> Value {
        Value::Float(NotNan::new(f).unwrap())
    }

    pub(crate) fn is_bool(&self) -> bool {
        if let Value::Bool(_) = self {
            true
        } else {
            false
        }
    }

    pub(crate) fn get_bool(&self) -> bool {
        if let Value::Bool(b) = *self {
            b
        } else {
            panic!("failed to extract value")
        }
    }
}

impl ops::Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        match (self, other) {
            (Unsigned(v1), Unsigned(v2)) => Unsigned(v1 + v2),
            (Signed(v1), Signed(v2)) => Signed(v1 + v2),
            (Float(v1), Float(v2)) => Float(v1 + v2),
            _ => panic!("Incompatible types."),
        }
    }
}

impl ops::Sub for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        match (self, other) {
            (Unsigned(v1), Unsigned(v2)) => Unsigned(v1 - v2),
            (Signed(v1), Signed(v2)) => Signed(v1 - v2),
            (Float(v1), Float(v2)) => Float(v1 - v2),
            _ => panic!("Incompatible types."),
        }
    }
}

impl ops::Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        match (self, other) {
            (Unsigned(v1), Unsigned(v2)) => Unsigned(v1 * v2),
            (Signed(v1), Signed(v2)) => Signed(v1 * v2),
            (Float(v1), Float(v2)) => Float(v1 * v2),
            _ => panic!("Incompatible types."),
        }
    }
}

impl ops::Div for Value {
    type Output = Value;
    fn div(self, other: Value) -> Value {
        match (self, other) {
            (Unsigned(v1), Unsigned(v2)) => Unsigned(v1 / v2),
            (Signed(v1), Signed(v2)) => Signed(v1 / v2),
            (Float(v1), Float(v2)) => Float(v1 / v2),
            _ => panic!("Incompatible types."),
        }
    }
}

impl ops::Rem for Value {
    type Output = Value;
    fn rem(self, other: Value) -> Value {
        match (self, other) {
            (Unsigned(v1), Unsigned(v2)) => Unsigned(v1 % v2),
            (Signed(v1), Signed(v2)) => Signed(v1 % v2),
            (Float(v1), Float(v2)) => Float(v1 % v2),
            _ => panic!("Incompatible types."),
        }
    }
}

impl Value {
    pub fn pow(self, exp: Value) -> Value {
        match (self, exp) {
            (Unsigned(v1), Unsigned(v2)) => Unsigned(v1.pow(v2 as u32)),
            (Signed(v1), Signed(v2)) => Signed(v1.pow(v2 as u32)),
            (Float(v1), Float(v2)) => Value::new_float(v1.powf(v2.into())),
            (Float(v1), Signed(v2)) => Value::new_float(v1.powi(v2 as i32)),
            _ => panic!("Incompatible types."),
        }
    }
}

impl ops::BitOr for Value {
    type Output = Value;
    fn bitor(self, other: Value) -> Value {
        match (self, other) {
            (Bool(v1), Bool(v2)) => Bool(v1 || v2),
            _ => panic!("Incompatible types."),
        }
    }
}

impl ops::BitAnd for Value {
    type Output = Value;
    fn bitand(self, other: Value) -> Value {
        match (self, other) {
            (Bool(v1), Bool(v2)) => Bool(v1 && v2),
            _ => panic!("Incompatible types."),
        }
    }
}

impl ops::Not for Value {
    type Output = Value;
    fn not(self) -> Value {
        match self {
            Bool(v) => Bool(!v),
            _ => panic!("Incompatible types."),
        }
    }
}

impl ops::Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        match self {
            Signed(v) => Signed(-v), // TODO Check
            Float(v) => Float(-v),
            _ => panic!("Incompatible types."),
        }
    }
}
