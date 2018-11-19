use ::ast::*;
use super::super::common::BuiltinType;
use std::fmt::{Display, Result, Formatter};
use super::super::common::{Type as OType};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct NumConfig {
    pub(crate) width: u8,
    pub(crate) def_signed: bool, // definitely signed
    pub(crate) def_float: bool, // definitely float
}

impl NumConfig {
    pub fn new_float(w: Option<u8>) -> NumConfig {
        NumConfig{ width: w.unwrap_or(32), def_float: true, def_signed: true }
    }

    pub fn new_signed(w: Option<u8>) -> NumConfig {
        NumConfig{ width: w.unwrap_or(8), def_float: false, def_signed: true }
    }

    pub fn new_unsigned(w: Option<u8>) -> NumConfig {
        NumConfig{ width: w.unwrap_or(8), def_float: false, def_signed: false }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum Candidates {
    Numeric(NumConfig),
    Concrete(BuiltinType),
    Tuple(Vec<Candidates>),
//    Defined(Vec<Candidates>),
    Any,
    None,
}

impl Display for Candidates {

    fn fmt(&self, f: &mut Formatter) -> Result {
        use ::print::PrintHelper;
        match self {
            Candidates::Numeric(cfg) if cfg.def_float => write!(f, "Float≥{}", cfg.width),
            Candidates::Numeric(cfg) if cfg.def_signed => write!(f, "(Float|Int)≥{}", cfg.width),
            Candidates::Numeric(cfg) => write!(f, "(Float|Int|UInt)>{}", cfg.width),
            Candidates::Tuple(cands) => PrintHelper::write(f, cands, "{ ", " }", " ,"),
//            Candidates::Defined(fields) => PrintHelper::write(f, fields, "{ ", " }", " ,"),
            Candidates::Concrete(ty) => write!(f, "{:?}", ty), // TODO: implement Display for builtin types.
            Candidates::Any => write!(f, "⊤"),
            Candidates::None => write!(f, "⊥"),
        }
    }

}

impl Candidates {

    pub fn meet(&self, others: &Candidates) -> Candidates {
        match (self, others) {
            (Candidates::Numeric(cfg0), Candidates::Numeric(cfg1)) => {
                // We leverage the specificity-hierarchy here:
                // UInt < Int < Float by signedness and floatness resp.
                let cfg = NumConfig {
                    width: std::cmp::max(cfg0.width, cfg1.width),
                    def_signed: cfg0.def_signed || cfg1.def_signed,
                    def_float: cfg0.def_float || cfg1.def_float,
                };
                assert!(cfg.def_signed || !cfg.def_float); // There is no unsigned float.
                Candidates::Numeric(cfg)
            },
            (Candidates::Tuple(ref v0), Candidates::Tuple(ref v1)) if v0.len() == v1.len() => {
                let res = v0.iter().zip(v1).map(|(l, r)| l.meet(r));
                let res: Vec<Candidates> = res.collect();
                if res.iter().any(|c| *c == Candidates::None) {
                    Candidates::None
                } else {
                    Candidates::Tuple(res)
                }
            },
            (Candidates::Concrete(c), Candidates::Numeric(cfg)) | (Candidates::Numeric(cfg), Candidates::Concrete(c)) => {
                Candidates::meet_conc_abs(*c, *cfg)
            }
            (Candidates::Concrete(t0), Candidates::Concrete(t1)) => Candidates::meet_builtin(*t0, *t1),
            (left, right) if left == right => left.clone(),
            _ => Candidates::None,
        }
    }

    fn meet_conc_abs(conc: BuiltinType, abs: NumConfig) -> Candidates {
        match conc {
            BuiltinType::UInt(_) if (abs.def_signed || abs.def_float) => Candidates::None,
            BuiltinType::Int(_) if abs.def_float => Candidates::None,
            BuiltinType::UInt(w) | BuiltinType::Int(w) | BuiltinType::Float(w) if w < abs.width => Candidates::None,
            _ => Candidates::Concrete(conc),
        }
    }

    fn meet_builtin(t0: BuiltinType, t1: BuiltinType) -> Candidates {
        match (t0, t1) {
            (BuiltinType::Float(w0), BuiltinType::Float(w1)) => Candidates::Concrete(BuiltinType::Float(std::cmp::max(w0, w1))),
            (BuiltinType::Int(w0), BuiltinType::Int(w1)) => Candidates::Concrete(BuiltinType::Int(std::cmp::max(w0, w1))),
            (BuiltinType::UInt(w0), BuiltinType::UInt(w1)) => Candidates::Concrete(BuiltinType::UInt(std::cmp::max(w0, w1))),
            (a, b) if a == b => Candidates::Concrete(a),
            _ => Candidates::None,
        }
    }

    pub fn is_none(&self) -> bool {
        *self == Candidates::None
    }
    pub fn is_tuple(&self) -> bool {
        match self {
            Candidates::Tuple(_) => true,
            _ => false,
        }
    }
    pub fn is_numeric(&self) -> bool {
        match self {
            Candidates::Numeric(_) | Candidates::Concrete(BuiltinType::Float(_)) | Candidates::Concrete(BuiltinType::UInt(_)) | Candidates::Concrete(BuiltinType::Int(_)) => true,
            Candidates::Any => true, // TODO For type inference, we need to change this and propagate requirements backwards.
            _ => false,
        }
    }
    pub fn is_integer(&self) -> bool {
        match self {
            Candidates::Numeric(cfg) if !cfg.def_float => true,
            Candidates::Concrete(BuiltinType::UInt(_)) | Candidates::Concrete(BuiltinType::Int(_)) => true,
            Candidates::Any => true, // TODO For type inference, we need to change this and propagate requirements backwards.
            _ => false,
        }
    }
    pub fn is_float(&self) -> bool {
        match self {
            Candidates::Numeric(_) | Candidates::Concrete(BuiltinType::Float(_)) => true,
            Candidates::Any => true, // TODO For type inference, we need to change this and propagate requirements backwards.
            _ => false,
        }
    }

    pub fn is_unsigned(&self) -> bool {
        match self {
            Candidates::Numeric(cfg) if !cfg.def_signed=> true,
            Candidates::Concrete(BuiltinType::UInt(_)) => true,
            Candidates::Any => true, // TODO For type inference, we need to change this and propagate requirements backwards.
            _ => false,
        }
    }
    pub fn is_logic(&self) -> bool {
        match self {
            Candidates::Concrete(BuiltinType::Bool) => true,
            Candidates::Any => true, // TODO For type inference, we need to change this and propagate requirements backwards.
            _ => false,
        }
    }

    pub fn into_signed(self) -> Candidates {
        match self {
            Candidates::Numeric(cfg) => Candidates::Numeric(NumConfig { width: cfg.width, def_signed: true, def_float: cfg.def_float }),
            Candidates::Concrete(BuiltinType::UInt(w)) => Candidates::Concrete(BuiltinType::Int(w)),
            _ if self.is_numeric() => self.clone(),
            _ => panic!("A non-numeric type cannot be signed!"),
        }
    }

    pub fn width(&self) -> Option<u8> {
        match self {
            Candidates::Numeric(cfg) => Some(cfg.width),
            Candidates::Concrete(BuiltinType::UInt(w)) | Candidates::Concrete(BuiltinType::Int(w))
                | Candidates::Concrete(BuiltinType::Float(w)) => Some(*w),
            _ => None,
        }
    }

}

impl Default for Candidates {
    fn default() -> Self {
        Candidates::Any
    }
}

impl<'a> From<&'a Literal> for Candidates {

    fn from(lit: &'a Literal) -> Self {
        fn find_required_bits(i: i128) -> u8 {
            let exact = 128 - (if i >= 0 { i } else { !i }).leading_zeros();
            match exact { // I'm pretty sure this can be performed more efficiently.
                0...8 => 8,
                9...16 => 16,
                17...32 => 32,
                33...64 => 64,
                65...128 => 128,
                _ => unreachable!(),
            }
        }
        fn find_required_bits_f(_f: f64) -> u8 {
            // It could be both, so we default to the less specific option.
            32
        }
        match &lit.kind {
            LitKind::Str(_) => Candidates::Concrete(BuiltinType::String),
            LitKind::Int(i) => {
                let width = find_required_bits(*i);
                assert!(width.count_ones() == 1 && width <= 128 && width >= 8);
                if *i < 0 {
                    Candidates::Numeric(NumConfig::new_signed(Some(width)))
                } else {
                    Candidates::Numeric(NumConfig::new_unsigned(Some(width)))
                }
            }
            LitKind::Float(f) => Candidates::Numeric(NumConfig::new_float(Some(find_required_bits_f(*f)))),
            LitKind::Bool(_) => Candidates::Concrete(BuiltinType::Bool),
            LitKind::Tuple(lits) => Candidates::Tuple(lits.iter().map(Candidates::from).collect()),
        }
    }
}

impl<'a> From<&'a BuiltinType> for Candidates {
    fn from(t: &'a BuiltinType) -> Self {
        Candidates::Concrete(*t)
    }
}

// TODO: Discuss.
impl<'a> From<&'a Candidates> for OType {
    fn from(cand: &'a Candidates) -> Self {
        match cand {
            Candidates::Any => unreachable!(), // Explicit type annotations should make this case impossible.
            Candidates::Concrete(t) => OType::BuiltIn(*t),
            Candidates::Numeric(cfg) =>
                if cfg.def_signed {
                    let width = if cfg.width >= 32 {
                        cfg.width
                    } else { 32 };
                    OType::BuiltIn(BuiltinType::Float(width))
                } else if cfg.def_signed {
                    let width = if cfg.width >= 32 {
                        cfg.width
                    } else { 32 };
                    OType::BuiltIn(BuiltinType::Int(width))
                } else {
                    let width = if cfg.width >= 32 {
                        cfg.width
                    } else { 32 };
                    OType::BuiltIn(BuiltinType::Int(width))
                },
            Candidates::Tuple(v) => {
                // Just make sure there are no nested tuples.
                assert!(v.iter().all(|t| match t {
                    Candidates::Tuple(_) => false,
                    _ => true,
                }));
                let transformed: Vec<BuiltinType> = v.iter()
                    .map(|t| t.into())
                    .map(|ot| match ot {
                        OType::BuiltIn(t) => t,
                        OType::Tuple(_v) => unreachable!(),
                    })
                    .collect();
                OType::Tuple(transformed)
            },
            Candidates::None => unreachable!(),
        }
    }
}
