use ordered_float::NotNan;
use std::cmp::Ordering;
use std::ops;
use streamlab_frontend::ir::Type;

use self::Value::*;

#[derive(Debug, Eq, Hash, Clone)]
pub(crate) enum Value {
    None,
    Bool(bool),
    Unsigned(u64),
    Signed(i64),
    Float(NotNan<f64>),
    Str(String),
}

impl Value {
    // TODO: -> Result<Option, ConversionError>
    pub(crate) fn try_from(source: &str, ty: &Type) -> Option<Value> {
        match ty {
            Type::Option(_) | Type::Function(_, _) => panic!("Cannot occur."),
            Type::String => Some(Value::Str(source.to_string())),
            Type::Tuple(_) => unimplemented!(),
            Type::Float(_) => source.parse::<f64>().ok().map(|f| Float(NotNan::new(f).unwrap())),
            Type::UInt(_) => {
                // TODO: This is just a quickfix!! Think of something more general.
                if source == "0.0" {
                    Some(Unsigned(0))
                } else {
                    source.parse::<u64>().map(Unsigned).ok()
                }
            }
            Type::Int(_) => source.parse::<i64>().map(Signed).ok(),
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
            (a, b) => panic!("Incompatible types: ({:?},{:?})", a, b),
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
            (a, b) => panic!("Incompatible types: ({:?},{:?})", a, b),
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
            (a, b) => panic!("Incompatible types: ({:?},{:?})", a, b),
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
            (a, b) => panic!("Incompatible types: ({:?},{:?})", a, b),
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
            (a, b) => panic!("Incompatible types: ({:?},{:?})", a, b),
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
            (a, b) => panic!("Incompatible types: ({:?},{:?})", a, b),
        }
    }
}

impl ops::BitOr for Value {
    type Output = Value;
    fn bitor(self, other: Value) -> Value {
        match (self, other) {
            (Bool(v1), Bool(v2)) => Bool(v1 || v2),
            (a, b) => panic!("Incompatible types: ({:?},{:?})", a, b),
        }
    }
}

impl ops::BitAnd for Value {
    type Output = Value;
    fn bitand(self, other: Value) -> Value {
        match (self, other) {
            (Bool(v1), Bool(v2)) => Bool(v1 && v2),
            (a, b) => panic!("Incompatible types: ({:?},{:?})", a, b),
        }
    }
}

impl ops::Not for Value {
    type Output = Value;
    fn not(self) -> Value {
        match self {
            Bool(v) => Bool(!v),
            a => panic!("Incompatible type: {:?}", a),
        }
    }
}

impl ops::Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        match self {
            Signed(v) => Signed(-v), // TODO Check
            Float(v) => Float(-v),
            a => panic!("Incompatible type: {:?}", a),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Bool(b1), Bool(b2)) => b1.eq(b2),
            (Unsigned(u1), Unsigned(u2)) => u1.eq(u2),
            (Signed(i1), Signed(i2)) => i1.eq(i2),
            (Float(f1), Float(f2)) => f1.eq(f2),
            (Str(s1), Str(s2)) => s1.eq(s2),
            (a, b) => panic!("Incompatible types: ({:?},{:?})", a, b),
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Unsigned(u1), Unsigned(u2)) => u1.cmp(u2),
            (Signed(i1), Signed(i2)) => i1.cmp(i2),
            (Float(f1), Float(f2)) => f1.cmp(f2),
            (Str(s1), Str(s2)) => s1.cmp(s2),
            (a, b) => panic!("Incompatible types: ({:?},{:?})", a, b),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn size_of_value() {
        let result = std::mem::size_of::<Value>();
        assert!(result == 32, "Size of `Value` should be 32 bytes, was `{}`", result);
    }
}
