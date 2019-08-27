use super::*;
use crate::ast::print::write_delim_list;
use std::fmt::{Display, Formatter, Result};

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self.kind {
            ExpressionKind::LoadConstant(c) => write!(f, "{}", c),
            ExpressionKind::Function(name, args, ty) => {
                write!(f, "{}(", name)?;
                if let Type::Function(arg_tys, res) = ty {
                    let zipped: Vec<(&Type, &Expression)> = arg_tys.iter().zip(args.iter()).collect();
                    if let Some((last, prefix)) = zipped.split_last() {
                        prefix.iter().fold(Ok(()), |accu, (t, a)| accu.and_then(|()| write!(f, "{}: {}, ", a, t)))?;
                        write!(f, "{}: {}", last.1, last.0)?;
                    }
                    write!(f, ") -> {}", res)
                } else {
                    unreachable!("The type of a function needs to be a function.")
                }
            }
            ExpressionKind::Convert { from, to, expr } => write!(f, "cast<{},{}>({})", from, to, expr),
            ExpressionKind::Tuple(elems) => write_delim_list(f, elems, "(", ")", ","),
            ExpressionKind::Ite { condition, consequence, alternative, .. } => {
                write!(f, "if {} then {} else {}", condition, consequence, alternative)
            }
            ExpressionKind::ArithLog(op, args, ty) => {
                write_delim_list(f, args, &format!("{}(", op), &format!(") : [{}]", ty), ",")
            }
            ExpressionKind::WindowLookup(wr) => write!(f, "{}", wr),
            ExpressionKind::Default { expr, default, .. } => write!(f, "{}.default({})", expr, default),
            ExpressionKind::OffsetLookup { target, offset } => write!(f, "{}.offset({})", target, offset),
            ExpressionKind::StreamAccess(sr, access) => match access {
                StreamAccessKind::Sync => write!(f, "{}", sr),
                StreamAccessKind::Hold => write!(f, "{}.hold()", sr),
                StreamAccessKind::Optional => write!(f, "{}.get()", sr),
            },
            ExpressionKind::TupleAccess(expr, num) => write!(f, "{}.{}", expr, num),
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
        use ArithLogOp::*;
        match self {
            Not => write!(f, "!"),
            Neg => write!(f, "~"),
            Add => write!(f, "+"),
            Sub => write!(f, "-"),
            Mul => write!(f, "*"),
            Div => write!(f, "/"),
            Rem => write!(f, "%"),
            Pow => write!(f, "^"),
            And => write!(f, "∧"),
            Or => write!(f, "∨"),
            Eq => write!(f, "="),
            Lt => write!(f, "<"),
            Le => write!(f, "≤"),
            Ne => write!(f, "≠"),
            Ge => write!(f, "≥"),
            Gt => write!(f, ">"),
            BitNot => write!(f, "~"),
            BitAnd => write!(f, "&"),
            BitOr => write!(f, "|"),
            BitXor => write!(f, "^"),
            Shl => write!(f, "<<"),
            Shr => write!(f, ">>"),
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
            Type::Bytes => write!(f, "Bytes"),
            Type::Option(inner) => write!(f, "Option<{}>", inner),
            Type::Bool => write!(f, "Bool"),
        }
    }
}

impl Display for WindowReference {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "Win({})", self.0)
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
