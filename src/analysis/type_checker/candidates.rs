use ::ast::*;
use super::super::common::BuiltinType;
use std::fmt::{Display, Result, Formatter};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct NumConfig {
    pub(crate) width: u8,
    pub(crate) def_signed: bool, // definitely signed
    pub(crate) def_float: bool, // definitely float
}

impl NumConfig {
    pub fn new_float(w: u8) -> NumConfig {
        NumConfig{ width: w, def_float: true, def_signed: true }
    }

    pub fn new_signed(w: u8) -> NumConfig {
        NumConfig{ width: w, def_float: false, def_signed: true }
    }

    pub fn new_any(w: u8) -> NumConfig {
        NumConfig{ width: w, def_float: false, def_signed: false }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum Candidates {
    Numeric(NumConfig),
    String,
    Tuple(Vec<Candidates>),
    Defined(Vec<Candidates>),
    Logic,
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
            Candidates::String => write!(f, "String"),
            Candidates::Tuple(cands) => PrintHelper::write(f, cands, "{ ", " }", " ,"),
            Candidates::Defined(fields) => PrintHelper::write(f, fields, "{ ", " }", " ,"),
            Candidates::Logic => write!(f, "Bool"),
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
            }
            (left, right) if left == right => left.clone(),
            _ => Candidates::None,
        }
    }

    pub fn is_none(&self) -> bool {
        *self == Candidates::None
    }
    pub fn is_numeric(&self) -> bool { self.num_cfg().is_some() }
    pub fn is_logic(&self) -> bool { *self == Candidates::Logic }

    pub fn num_cfg(&self) -> Option<NumConfig> {
        match self {
            Candidates::Numeric(cfg) => Some(*cfg),
            _ => None,
        }
    }

}

impl Default for Candidates {
    fn default() -> Self {
        Candidates::Any
    }
}

impl<'a> From<&'a Type> for Candidates {
    fn from(ty: &Type) -> Self {
        match &ty.kind {
            TypeKind::Tuple(elems) => unimplemented!(),
            TypeKind::Simple(name) => unimplemented!(),
            TypeKind::Malformed(string) => unreachable!(),
        }
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
        fn find_required_bits_f(f: f64) -> u8 {
            // It could be both, so we default to the less specific option.
            32
        }
        match &lit.kind {
            LitKind::Str(_) => Candidates::String,
            LitKind::Int(i) => {
                let width = find_required_bits(*i);
                assert!(width.count_ones() == 1 && width <= 128 && width >= 8);
                let config = if *i < 0 {
                    NumConfig::new_signed(width)
                } else {
                    NumConfig::new_any(width)
                };
                Candidates::Numeric(config)
            }
            LitKind::Float(f) => Candidates::Numeric(NumConfig::new_float(find_required_bits_f(*f))),
            LitKind::Bool(_) => Candidates::Logic,
            LitKind::Tuple(lits) => Candidates::Tuple(lits.iter().map(Candidates::from).collect()),
        }
    }
}

impl<'a> From<&'a Option<Type>> for Candidates {

    fn from(lit: &'a Option<Type>) -> Self {
        if let Some(t) = lit {
            Candidates::from(t)
        } else {
            Candidates::Any
        }
    }
}

impl<'a> From<&'a TypeDeclaration> for Candidates {
    fn from(td: &'a TypeDeclaration) -> Self {
        let subs: Vec<Candidates> = td.fields.iter().map(|field| Candidates::from(&field.ty)).collect();
        Candidates::Defined(subs)
    }
}

impl<'a> From<&'a BuiltinType> for Candidates {
    fn from(t: &'a BuiltinType) -> Self {
        match t {
            BuiltinType::String => Candidates::String,
            BuiltinType::Bool => Candidates::Logic,
            _ => unimplemented!(), // Wait for change in builtin types.
        }
    }
}
