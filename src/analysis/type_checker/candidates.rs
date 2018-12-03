use super::super::common::BuiltinType;
use super::super::common::Type as OType;
use ast::*;
use std::fmt::{Display, Formatter, Result};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct NumConfig {
    pub(crate) width: u8,
    pub(crate) def_signed: bool, // definitely signed
    pub(crate) def_float: bool,  // definitely float
}

impl NumConfig {
    pub fn new_float(w: Option<u8>) -> NumConfig {
        NumConfig {
            width: w.unwrap_or(32),
            def_float: true,
            def_signed: true,
        }
    }

    pub fn new_signed(w: Option<u8>) -> NumConfig {
        NumConfig {
            width: w.unwrap_or(8),
            def_float: false,
            def_signed: true,
        }
    }

    pub fn new_unsigned(w: Option<u8>) -> NumConfig {
        NumConfig {
            width: w.unwrap_or(8),
            def_float: false,
            def_signed: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimingInfo {
    EventBased,
    TimeBased,
    Unknown,
}

impl TimingInfo {
    pub(crate) fn meet_opt(lhs: Option<TimingInfo>, rhs: Option<TimingInfo>) -> Option<TimingInfo> {
        match (lhs, rhs) {
            (Some(left), Some(right)) => TimingInfo::meet(left, right),
            _ => None,
        }
    }

    pub(crate) fn meet(left: TimingInfo, right: TimingInfo) -> Option<TimingInfo> {
        match (left, right) {
            (TimingInfo::Unknown, s) | (s, TimingInfo::Unknown) => Some(s),
            (s0, s1) if s0 == s1 => Some(s0),
            _ => None,
        }
    }
}

impl Display for TimingInfo {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "{}",
            match self {
                TimingInfo::Unknown => "⊤",
                TimingInfo::EventBased => "Event",
                TimingInfo::TimeBased => "Time",
            }
        )
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) enum Candidates {
    Numeric(NumConfig, TimingInfo),
    Concrete(BuiltinType, TimingInfo),
    Tuple(Vec<Candidates>),
    //    Defined(Vec<Candidates>),
    Any(TimingInfo),
    None,
}

impl Candidates {
    pub(crate) fn timing_info(&self) -> Option<TimingInfo> {
        match self {
            Candidates::Numeric(_, ti) => Some(*ti),
            Candidates::Concrete(_, ti) => Some(*ti),
            Candidates::Tuple(v) => v
                .iter()
                .map(|c| c.timing_info())
                .fold(Some(TimingInfo::Unknown), TimingInfo::meet_opt),
            Candidates::Any(ti) => Some(*ti),
            Candidates::None => None,
        }
    }

    fn replace_ti(&self, ti: TimingInfo) -> Candidates {
        match self {
            Candidates::Numeric(cfg, _) => Candidates::Numeric(*cfg, ti),
            Candidates::Concrete(c, _) => Candidates::Concrete(*c, ti),
            Candidates::Tuple(v) => {
                let new_v = v.iter().map(|c| c.replace_ti(ti)).collect();
                Candidates::Tuple(new_v)
            }
            Candidates::Any(_) => Candidates::Any(ti),
            Candidates::None => Candidates::None,
        }
    }

    pub(crate) fn as_event_driven(&self) -> Candidates {
        self.replace_ti(TimingInfo::EventBased)
    }

    pub(crate) fn as_time_driven(&self) -> Candidates {
        self.replace_ti(TimingInfo::TimeBased)
    }

    pub(crate) fn top() -> Candidates {
        Candidates::Any(TimingInfo::Unknown)
    }

    pub(crate) fn bot() -> Candidates {
        Candidates::None
    }
}

impl Display for Candidates {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use print::PrintHelper;
        match self {
            Candidates::Numeric(cfg, ti) if cfg.def_float => {
                write!(f, "Float≥{}/{}", cfg.width, ti)
            }
            Candidates::Numeric(cfg, ti) if cfg.def_signed => {
                write!(f, "(Float|Int)≥{}/{}", cfg.width, ti)
            }
            Candidates::Numeric(cfg, ti) => write!(f, "(Float|Int|UInt)>{}/{}", cfg.width, ti),
            Candidates::Tuple(cands) => PrintHelper::write(f, cands, "{ ", " }", " ,"),
            // Candidates::Defined(fields) => PrintHelper::write(f, fields, "{ ", " }", " ,"),
            Candidates::Concrete(ty, ti) => write!(f, "{:?}/{}", ty, ti), // TODO: implement Display for builtin types.
            Candidates::Any(ti) => write!(f, "⊤/{}", ti),
            Candidates::None => write!(f, "⊥"),
        }
    }
}

impl Candidates {
    pub fn meet(&self, others: &Candidates) -> Candidates {
        let time_meet = TimingInfo::meet_opt(self.timing_info(), others.timing_info());
        if time_meet.is_none() {
            return Candidates::None;
        }
        let ti = time_meet.unwrap();
        match (self, others) {
            (Candidates::Numeric(cfg, _), Candidates::Any(_)) => Candidates::Numeric(*cfg, ti),
            (Candidates::Tuple(v), Candidates::Any(_)) => {
                Candidates::Tuple(v.iter().map(|c| c.meet(others)).collect())
            }
            (Candidates::Concrete(c, _), Candidates::Any(_)) => Candidates::Concrete(*c, ti),
            (Candidates::None, Candidates::Any(_)) => Candidates::None,
            (Candidates::Numeric(cfg0, _), Candidates::Numeric(cfg1, _)) => {
                // We leverage the specificity-hierarchy here:
                // UInt < Int < Float by signedness and floatness resp.
                let cfg = NumConfig {
                    width: std::cmp::max(cfg0.width, cfg1.width),
                    def_signed: cfg0.def_signed || cfg1.def_signed,
                    def_float: cfg0.def_float || cfg1.def_float,
                };
                assert!(cfg.def_signed || !cfg.def_float); // There is no unsigned float.
                Candidates::Numeric(cfg, ti)
            }
            (Candidates::Tuple(ref v0), Candidates::Tuple(ref v1)) => {
                if v0.len() == v1.len() {
                    let res = v0.iter().zip(v1).map(|(l, r)| l.meet(r));
                    let res: Vec<Candidates> = res.collect();
                    if res.iter().any(|c| *c == Candidates::None) {
                        Candidates::None
                    } else {
                        Candidates::Tuple(res)
                    }
                } else {
                    Candidates::None
                }
            }
            (Candidates::Concrete(c, _), Candidates::Numeric(cfg, _))
            | (Candidates::Numeric(cfg, _), Candidates::Concrete(c, _)) => {
                Candidates::meet_conc_abs(*c, *cfg, ti)
            }
            (Candidates::Concrete(t0, _), Candidates::Concrete(t1, _)) => {
                Candidates::meet_builtin(*t0, *t1, ti)
            }
            (Candidates::Any(_), Candidates::Any(_)) => Candidates::Any(ti),
            (Candidates::None, Candidates::None) => Candidates::None,
            _ => Candidates::None,
        }
    }

    fn meet_conc_abs(conc: BuiltinType, abs: NumConfig, ti: TimingInfo) -> Candidates {
        match conc {
            BuiltinType::UInt(w) | BuiltinType::Int(w) | BuiltinType::Float(w) if w < abs.width => {
                Candidates::None
            }
            BuiltinType::UInt(_) if !(abs.def_signed || abs.def_float) => {
                Candidates::Concrete(conc, ti)
            }
            BuiltinType::Int(_) if !abs.def_float => Candidates::Concrete(conc, ti),
            BuiltinType::Float(_) => Candidates::Concrete(conc, ti),
            _ => Candidates::None,
        }
    }

    fn meet_builtin(t0: BuiltinType, t1: BuiltinType, ti: TimingInfo) -> Candidates {
        match (t0, t1) {
            (BuiltinType::Float(w0), BuiltinType::Float(w1)) => {
                Candidates::Concrete(BuiltinType::Float(std::cmp::max(w0, w1)), ti)
            }
            (BuiltinType::Int(w0), BuiltinType::Int(w1)) => {
                Candidates::Concrete(BuiltinType::Int(std::cmp::max(w0, w1)), ti)
            }
            (BuiltinType::UInt(w0), BuiltinType::UInt(w1)) => {
                Candidates::Concrete(BuiltinType::UInt(std::cmp::max(w0, w1)), ti)
            }
            (a, b) if a == b => Candidates::Concrete(a, ti),
            _ => Candidates::None,
        }
    }

    pub fn is_none(&self) -> bool {
        *self == Candidates::None
    }
    #[allow(dead_code)]
    pub fn is_tuple(&self) -> bool {
        match self {
            Candidates::Tuple(_) => true,
            _ => false,
        }
    }
    pub fn is_numeric(&self) -> bool {
        match self {
            Candidates::Numeric(_, _)
            | Candidates::Concrete(BuiltinType::Float(_), _)
            | Candidates::Concrete(BuiltinType::UInt(_), _)
            | Candidates::Concrete(BuiltinType::Int(_), _) => true,
            Candidates::Any(_) => true, // TODO For type inference, we need to change this and propagate requirements backwards.
            _ => false,
        }
    }
    pub fn is_integer(&self) -> bool {
        match self {
            Candidates::Numeric(cfg, _) if !cfg.def_float => true,
            Candidates::Concrete(BuiltinType::UInt(_), _)
            | Candidates::Concrete(BuiltinType::Int(_), _) => true,
            Candidates::Any(_) => true, // TODO For type inference, we need to change this and propagate requirements backwards.
            _ => false,
        }
    }
    pub fn is_float(&self) -> bool {
        match self {
            Candidates::Numeric(_, _) | Candidates::Concrete(BuiltinType::Float(_), _) => true,
            Candidates::Any(_) => true, // TODO For type inference, we need to change this and propagate requirements backwards.
            _ => false,
        }
    }

    pub fn is_unsigned(&self) -> bool {
        match self {
            Candidates::Numeric(cfg, _) if !cfg.def_signed => true,
            Candidates::Concrete(BuiltinType::UInt(_), _) => true,
            Candidates::Any(_) => true, // TODO For type inference, we need to change this and propagate requirements backwards.
            _ => false,
        }
    }
    pub fn is_logic(&self) -> bool {
        match self {
            Candidates::Concrete(BuiltinType::Bool, _) => true,
            Candidates::Any(_) => true, // TODO For type inference, we need to change this and propagate requirements backwards.
            _ => false,
        }
    }
    pub fn into_signed(self) -> Candidates {
        match self {
            Candidates::Numeric(cfg, ti) => Candidates::Numeric(
                NumConfig {
                    width: cfg.width,
                    def_signed: true,
                    def_float: cfg.def_float,
                },
                ti,
            ),
            Candidates::Concrete(BuiltinType::UInt(w), ti) => {
                Candidates::Concrete(BuiltinType::Int(w), ti)
            }
            _ if self.is_numeric() => self.clone(),
            _ => panic!("A non-numeric type cannot be signed!"),
        }
    }

    pub fn width(&self) -> Option<u8> {
        match self {
            Candidates::Numeric(cfg, _) => Some(cfg.width),
            Candidates::Concrete(BuiltinType::UInt(w), _)
            | Candidates::Concrete(BuiltinType::Int(w), _)
            | Candidates::Concrete(BuiltinType::Float(w), _) => Some(*w),
            _ => None,
        }
    }
}

impl<'a> From<&'a Literal> for Candidates {
    fn from(lit: &'a Literal) -> Self {
        fn find_required_bits(i: i128) -> u8 {
            let exact = 128 - (if i >= 0 { i } else { !i }).leading_zeros();
            match exact {
                // I'm pretty sure this can be performed more efficiently.
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
            LitKind::Str(_) => Candidates::Concrete(BuiltinType::String, TimingInfo::Unknown),
            LitKind::Int(i) => {
                let width = find_required_bits(*i);
                assert!(width.count_ones() == 1 && width <= 128 && width >= 8);
                if *i < 0 {
                    Candidates::Numeric(NumConfig::new_signed(Some(width)), TimingInfo::Unknown)
                } else {
                    Candidates::Numeric(NumConfig::new_unsigned(Some(width)), TimingInfo::Unknown)
                }
            }
            LitKind::Float(f) => Candidates::Numeric(
                NumConfig::new_float(Some(find_required_bits_f(*f))),
                TimingInfo::Unknown,
            ),
            LitKind::Bool(_) => Candidates::Concrete(BuiltinType::Bool, TimingInfo::Unknown),
        }
    }
}

impl<'a> From<&'a BuiltinType> for Candidates {
    fn from(t: &'a BuiltinType) -> Self {
        Candidates::Concrete(*t, TimingInfo::Unknown)
    }
}

// TODO: Discuss.
impl<'a> From<&'a Candidates> for OType {
    fn from(cand: &'a Candidates) -> Self {
        match cand {
            Candidates::Any(_) => unreachable!(), // Explicit type annotations should make this case impossible.
            Candidates::Concrete(t, _) => OType::BuiltIn(*t),
            Candidates::Numeric(cfg, _) => if cfg.def_float {
                let width = if cfg.width >= 32 { cfg.width } else { 32 };
                OType::BuiltIn(BuiltinType::Float(width))
            } else if cfg.def_signed {
                let width = if cfg.width >= 32 { cfg.width } else { 32 };
                OType::BuiltIn(BuiltinType::Int(width))
            } else {
                let width = if cfg.width >= 32 { cfg.width } else { 32 };
                OType::BuiltIn(BuiltinType::UInt(width))
            },
            Candidates::Tuple(v) => {
                // Just make sure there are no nested tuples.
                assert!(v.iter().all(|t| match t {
                    Candidates::Tuple(_) => false,
                    _ => true,
                }));
                let transformed: Vec<BuiltinType> = v
                    .iter()
                    .map(|t| t.into())
                    .map(|ot| match ot {
                        OType::BuiltIn(t) => t,
                        OType::Tuple(_v) => unreachable!(),
                    }).collect();
                OType::Tuple(transformed)
            }
            Candidates::None => unreachable!(),
        }
    }
}
