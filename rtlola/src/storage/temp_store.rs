use super::Value;
use lola_parser::*;

pub(crate) struct TempStore {
    offsets: Vec<usize>,
    data: Vec<u8>,
    types: Vec<Type>,
}

impl TempStore {
    pub(crate) fn new(expr: &Expression) -> TempStore {
        let mut offsets = Vec::new();
        let mut size = 0;
        for ty in expr.temporaries.iter() {
            offsets.push(size as usize);
            let ty = match ty {
                Type::Option(t) => t, // We don't store options but resolve during lookup.
                _ => ty,
            };
            size += ty.size().unwrap().0;
        }

        let offsets = offsets;
        let size = size as usize;

        let data = vec![0; size];

        let types = expr
            .temporaries
            .clone()
            .into_iter()
            .map(|ty| if let Type::Option(t) = ty { *t.clone() } else { ty })
            .collect();

        TempStore { offsets, data, types }
    }

    pub(crate) fn get_value(&self, t: Temporary) -> Value {
        match self.types[t.0 as usize] {
            Type::UInt(_) => Value::Unsigned(self.get_unsigned(t)),
            Type::Int(_) => Value::Signed(self.get_signed(t)),
            Type::Bool => Value::Bool(self.get_bool(t)),
            _ => unimplemented!(),
        }
    }

    fn get_bounds(&self, t: Temporary) -> (usize, usize) {
        let lower = self.offsets[t.0 as usize];
        let ty = &self.types[t.0 as usize];
        let higher = lower
            + match self.types[t.0 as usize] {
                Type::UInt(_) => ty.size().unwrap().0 as usize,
                _ => panic!("Unexpected call to `TempStore::get_unsigned`."),
            };
        (lower, higher)
    }

    pub(crate) fn get_unsigned(&self, t: Temporary) -> u128 {
        // TODO: The check is not required, just for safety.
        if let Type::UInt(_) = self.types[t.0 as usize] {
            let (lower, higher) = self.get_bounds(t);
            Self::parse_bytes(&self.data[lower..higher]) as u128
        } else {
            panic!("Unexpected call to `TempStore::get_unsigned`.")
        }
    }

    pub(crate) fn get_signed(&self, t: Temporary) -> i128 {
        // TODO: The check is not required, just for safety.
        if let Type::Int(_) = self.types[t.0 as usize] {
            let (lower, higher) = self.get_bounds(t);
            Self::parse_bytes(&self.data[lower..higher]) as i128
        } else {
            panic!("Unexpected call to `TempStore::get_unsigned`.")
        }
    }

    pub(crate) fn write_value(&mut self, t: Temporary, v: Value) {
        match (&self.types[t.0], v) {
            (Type::UInt(_), Value::Unsigned(u)) => self.write_unsigned(t, u),
            (Type::Int(_), Value::Signed(i)) => self.write_signed(t, i),
            (Type::Bool, Value::Bool(b)) => self.write_bool(t, b),
            _ => unimplemented!(),
        }
    }

    pub(crate) fn write_unsigned(&mut self, t: Temporary, v: u128) {
        // TODO: The check is not required, just for safety.
        if let Type::UInt(_) = self.types[t.0 as usize] {
            let (lower, higher) = self.get_bounds(t);
            Self::write_bytes(&mut self.data[lower..higher], v)
        } else {
            panic!("Unexpected call to `TempStore::get_unsigned`.")
        }
    }

    pub(crate) fn write_signed(&mut self, t: Temporary, v: i128) {
        // TODO: The check is not required, just for safety.
        if let Type::Int(_) = self.types[t.0 as usize] {
            let (lower, higher) = self.get_bounds(t);
            Self::write_bytes(&mut self.data[lower..higher], v as u128)
        } else {
            panic!("Unexpected call to `TempStore::get_unsigned`.")
        }
    }

    pub(crate) fn get_bool(&self, t: Temporary) -> bool {
        self.data[self.offsets[t.0 as usize]] != 0
    }

    pub(crate) fn write_bool(&mut self, t: Temporary, v: bool) {
        self.data[self.offsets[t.0 as usize]] = v as u8
    }

    pub(crate) fn move_val(&mut self, src: Temporary, tar: Temporary) {
        if src == tar {
            return;
        }
        let (s_low, s_high) = self.get_bounds(src);
        let (t_low, t_high) = self.get_bounds(tar);
        assert_eq!(s_high - s_low, t_high - t_low);
        assert!(s_high <= t_low && t_high <= s_low);
        for i in 0..(s_high - s_low) {
            self.data[t_low + i] = self.data[s_low + 1];
        }
    }

    #[inline]
    fn write_bytes(data: &mut [u8], mut v: u128) {
        // Write least significant byte first.
        for i in (0..data.len()).rev() {
            data[i] = v as u8;
            v = v >> 8;
        }
    }

    #[inline]
    fn parse_bytes(d: &[u8]) -> u64 {
        assert!(d.len().is_power_of_two());
        let mut res = 0u64;
        for i in 0..d.len() {
            res = (res << 8) | (d[i] as u64);
        }
        res
    }
}
