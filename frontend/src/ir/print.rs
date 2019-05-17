use super::*;
use crate::ast::print::write_delim_list;
use std::fmt::{Display, Formatter, Result};

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Expression::LoadConstant(c) => write!(f, "{}", c),
            Expression::Function(name, args, ty) => {
                write!(f, "{}(", name)?;
                if let Type::Function(arg_tys, res) = ty {
                    let zipped: Vec<(&Type, &Expression)> = arg_tys.iter().zip(args.iter()).collect();
                    if let Some((last, prefix)) = zipped.split_last() {
                        prefix.iter().fold(Ok(()), |accu, (t, a)| accu.and_then(|()| write!(f, "{}: {}, ", a, t)))?;
                        write!(f, "{}: {}", last.1, last.0)?;
                    }
                    write!(f, ") -> {}", res)
                } else {
                    panic!("The type of a function needs to be a function.")
                }
            }
            Expression::Convert { from, to, expr } => write!(f, "{}.cast::<{},{}>()", expr, from, to),
            Expression::Tuple(elems) => write_delim_list(f, elems, "(", ")", ","),
            Expression::Ite { condition, consequence, alternative } => {
                write!(f, "if {} then {} else {}", condition, consequence, alternative)
            }
            Expression::ArithLog(op, args, ty) => {
                write_delim_list(f, args, &format!("{}(", op), &format!(") : [{}]", ty), ",")
            }
            Expression::WindowLookup(wr) => write!(f, "{}", wr),
            Expression::Default { expr, default } => write!(f, "{}.default({})", expr, default),
            Expression::OffsetLookup { target, offset } => write!(f, "{}.offset({})", target, offset),
            Expression::SyncStreamLookup(sr) => write!(f, "{}", sr),
            Expression::StreamAccess(sr, access) => match access {
                StreamAccessKind::Hold => write!(f, "{}.hold()", sr),
                StreamAccessKind::Optional => write!(f, "{}.get()", sr),
            },
            Expression::TupleAccess(expr, num) => write!(f, "{}.{}", expr, num),
        }
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Constant::Bool(b) => write!(f, "{}", b),
            Constant::UInt(u) => write!(f, "{}", u),
            Constant::Int(i) => write!(f, "{}", i),
            Constant::Float(fl) => write!(f, "{}", fl),
            Constant::Str(s) => write!(f, "{}", s),
        }
    }
}

impl Display for Offset {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Offset::PastDiscreteOffset(u) => write!(f, "{}", u),
            _ => unimplemented!(),
        }
    }
}

impl Display for ArithLogOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            ArithLogOp::Not => write!(f, "!"),
            ArithLogOp::Neg => write!(f, "~"),
            ArithLogOp::Add => write!(f, "+"),
            ArithLogOp::Sub => write!(f, "-"),
            ArithLogOp::Mul => write!(f, "*"),
            ArithLogOp::Div => write!(f, "/"),
            ArithLogOp::Rem => write!(f, "%"),
            ArithLogOp::Pow => write!(f, "^"),
            ArithLogOp::And => write!(f, "∧"),
            ArithLogOp::Or => write!(f, "∨"),
            ArithLogOp::Eq => write!(f, "="),
            ArithLogOp::Lt => write!(f, "<"),
            ArithLogOp::Le => write!(f, "≤"),
            ArithLogOp::Ne => write!(f, "≠"),
            ArithLogOp::Ge => write!(f, "≥"),
            ArithLogOp::Gt => write!(f, ">"),
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Type::Float(_) => write!(f, "Float{}", self.size().expect("Floats are sized.").0 * 8),
            Type::UInt(_) => write!(f, "UInt{}", self.size().expect("UInts are sized.").0 * 8),
            Type::Int(_) => write!(f, "Int{}", self.size().expect("Ints are sized.").0 * 8),
            Type::Function(args, res) => write_delim_list(f, args, "(", &format!(") -> {}", res), ","),
            Type::Tuple(elems) => write_delim_list(f, elems, "(", ")", ","),
            Type::String => write!(f, "String"),
            Type::Option(inner) => write!(f, "Option<{}>", inner),
            Type::Bool => write!(f, "Bool"),
        }
    }
}

impl Display for WindowReference {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "Win({})", self.ix)
    }
}

impl Display for StreamReference {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            StreamReference::OutRef(ix) => write!(f, "Out({})", ix),
            StreamReference::InRef(ix) => write!(f, "In({})", ix),
        }
    }
}
