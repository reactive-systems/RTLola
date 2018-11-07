//! This module contains `Display` implementations for the AST.

use super::ast::*;
use super::parse::Ident;
use std::fmt::{Display, Formatter, Result};

struct PrintHelper {}
impl PrintHelper {
    fn write<T: Display>(f: &mut Formatter, v: &[T], pref: &str, suff: &str, join: &str) -> Result {
        write!(f, "{}", pref)?;
        if let Some(e) = v.first() {
            write!(f, "{}", e)?;
            for b in &v[1..] {
                write!(f, "{}{}", join, b)?;
            }
        }
        write!(f, "{}", suff)?;
        Ok(())
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "constant {}{} := {}",
            self.name,
            if let Some(ty) = &self.ty {
                format!(": {}", ty)
            } else {
                String::new()
            },
            self.literal
        )
    }
}

impl Display for Input {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "input {}: {}", self.name, self.ty)
    }
}

impl Display for Output {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "output {}{} := {}",
            self.name,
            if let Some(ty) = &self.ty {
                format!(": {}", ty)
            } else {
                String::new()
            },
            self.expression
        )
    }
}

impl Display for Trigger {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "trigger{} {}{}",
            if let Some(name) = &self.name {
                format!(" {} :=", name)
            } else {
                String::new()
            },
            self.expression,
            if let Some(msg) = &self.message {
                format!(" \"{}\"", msg)
            } else {
                String::new()
            }
        )
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.kind)
    }
}

impl Display for TypeKind {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match &self {
            TypeKind::Simple(name) => write!(f, "{}", name),
            TypeKind::Malformed(s) => write!(f, "{}", s),
            TypeKind::Tuple(types) => PrintHelper::write(f, types, "(", ")", ", "),
            TypeKind::UserDefined(fields) => PrintHelper::write(f, fields, "", "", ", "),
        }
    }
}

impl Display for TypeDeclField {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}: {}", &self.name, &self.ty)
    }
}

impl Display for TypeDeclaration {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "type {} {{ {} }}",
            if let Some(name) = &self.name {
                format!("{}", name)
            } else {
                String::new()
            },
            self.kind
        )
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match &self.kind {
            ExpressionKind::Lit(l) => write!(f, "{}", l),
            ExpressionKind::Ident(ident) => write!(f, "{}", ident),
            ExpressionKind::Default(expr, val) => write!(f, "{} ? {}", expr, val),
            ExpressionKind::Lookup(inst, offset, Some(aggr)) => {
                write!(f, "{}[{}, {}]", inst, offset, aggr)
            }
            ExpressionKind::Lookup(inst, offset, None) => write!(f, "{}[{}]", inst, offset),
            ExpressionKind::Binary(op, lhs, rhs) => write!(f, "{} {} {}", lhs, op, rhs),
            ExpressionKind::Unary(operator, operand) => write!(f, "{}{}", operator, operand),
            ExpressionKind::Ite(cond, cons, alt) => {
                write!(f, "if {} then {} else {}", cond, cons, alt)
            }
            ExpressionKind::ParenthesizedExpression(left, expr, right) => write!(
                f,
                "{}{}{}",
                if left.is_some() { "(" } else { "" },
                expr,
                if right.is_some() { ")" } else { "" }
            ),
            ExpressionKind::MissingExpression() => Ok(()),
            ExpressionKind::Tuple(exprs) => PrintHelper::write(f, exprs, "(", ")", ", "),
            ExpressionKind::Function(kind, args) => {
                write!(f, "{}", kind)?;
                PrintHelper::write(f, args, "(", ")", ", ")
            },
        }
    }
}

impl Display for StreamInstance {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.stream_identifier)?;
        if !self.arguments.is_empty() {
            PrintHelper::write(f, &self.arguments, "(", ")", ", ")
        } else {
            Ok(())
        }
    }
}

impl Display for FunctionKind {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "{}",
            match self {
                FunctionKind::NthRoot => "nroot",
                FunctionKind::Sqrt => "sqrt",
                FunctionKind::Projection => "π",
                FunctionKind::Sin => "sin",
                FunctionKind::Cos => "cos",
                FunctionKind::Tan => "tan",
                FunctionKind::Arcsin => "sin⁻¹",
                FunctionKind::Arccos => "cos⁻¹",
                FunctionKind::Arctan => "tan⁻¹",
                FunctionKind::Exp => "exp",
                FunctionKind::Floor => "floor",
                FunctionKind::Ceil => "ceil",
            }
        )
    }
}

impl Display for Offset {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            Offset::DiscreteOffset(e) => write!(f, "{}", e),
            Offset::RealTimeOffset(e, u) => write!(f, "{}{}", e, u),
        }
    }
}

impl Display for TimeUnit {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "{}",
            match self {
                TimeUnit::NanoSecond => "ns",
                TimeUnit::MicroSecond => "ms",
                TimeUnit::MilliSecond => "μs",
                TimeUnit::Second => "s",
                TimeUnit::Minute => "min",
                TimeUnit::Hour => "h",
                TimeUnit::Day => "d",
                TimeUnit::Week => "w",
                TimeUnit::Year => "a",
            }
        )
    }
}

impl Display for WindowOperation {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "{}",
            match self {
                WindowOperation::Sum => "Σ",
                WindowOperation::Product => "Π",
                WindowOperation::Average => "avg",
                WindowOperation::Count => "#",
                WindowOperation::Integral => "∫",
            }
        )
    }
}

impl Display for UnOp {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "{}",
            match self {
                UnOp::Not => "!",
                UnOp::Neg => "-",
            }
        )
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match &self.kind {
            LitKind::Bool(val) => write!(f, "{}", val),
            LitKind::Int(i) => write!(f, "{}", i),
            LitKind::Float(fl) => {
                if fl.fract() == 0.0 {
                    write!(f, "{:.1}", fl)
                } else {
                    write!(f, "{}", fl)
                }
            }
            LitKind::Str(s) => write!(f, "{}", s),
            LitKind::Tuple(vals) => PrintHelper::write(f, vals, "(", ")", ", "),
        }
    }
}

impl Display for Ident {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.name)
    }
}

impl Display for BinOp {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match &self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Rem => write!(f, "%"),
            BinOp::Pow => write!(f, "**"),
            BinOp::And => write!(f, "∧"),
            BinOp::Or => write!(f, "∨"),
            BinOp::Eq => write!(f, "="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Le => write!(f, "≤"),
            BinOp::Ne => write!(f, "!="),
            BinOp::Gt => write!(f, ">"),
            BinOp::Ge => write!(f, "≥"),
        }
    }
}

impl Display for LanguageSpec {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match &self {
            LanguageSpec::Classic => write!(f, "ClassicLola"),
            LanguageSpec::Lola2 => write!(f, "Lola 2.0"),
            LanguageSpec::RTLola => write!(f, "RTLola"),
        }
    }
}

impl Display for LolaSpec {
    fn fmt(&self, f: &mut Formatter) -> Result {
        if let Some(v) = &self.language {
            write!(f, "version {}", v)?
        }
        let mut first = true;
        if !self.type_declarations.is_empty() {
            PrintHelper::write(
                f,
                &self.type_declarations,
                if first { "" } else { " " },
                "",
                " ",
            )?;
            first = false;
        }
        if !self.constants.is_empty() {
            PrintHelper::write(f, &self.constants, if first { "" } else { " " }, "", " ")?;
            first = false;
        }
        if !self.inputs.is_empty() {
            PrintHelper::write(f, &self.inputs, if first { "" } else { " " }, "", " ")?;
            first = false;
        }
        if !self.outputs.is_empty() {
            PrintHelper::write(f, &self.outputs, if first { "" } else { " " }, "", " ")?;
            first = false;
        }
        if !self.trigger.is_empty() {
            PrintHelper::write(f, &self.trigger, if first { "" } else { " " }, "", " ")?
        }
        Ok(())
    }
}
