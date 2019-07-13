//! This module contains `Display` implementations for the AST.

use super::*;
use crate::parse::Ident;
use num::{BigInt, ToPrimitive, Zero};
use std::fmt::{Display, Formatter, Result};
use std::ops::Rem;

/// Writes out the joined vector `v`, enclosed by the given strings `pref` and `suff`.
pub(crate) fn write_delim_list<T: Display>(
    f: &mut Formatter<'_>,
    v: &[T],
    pref: &str,
    suff: &str,
    join: &str,
) -> Result {
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

/// Helper to format an Optional
fn format_opt<T: Display>(opt: &Option<T>, pref: &str, suff: &str) -> String {
    if let Some(ref e) = opt {
        format!("{}{}{}", pref, e, suff)
    } else {
        String::new()
    }
}

/// Formats an optional type
fn format_type(ty: &Option<Type>) -> String {
    format_opt(ty, ": ", "")
}

impl Display for Constant {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "constant {}{} := {}", self.name, format_type(&self.ty), self.literal)
    }
}

impl Display for Input {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "input {}", self.name)?;
        if !self.params.is_empty() {
            write_delim_list(f, &self.params, " <", ">", ", ")?;
        }
        write!(f, ": {}", self.ty)
    }
}

impl Display for Output {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "output {}", self.name)?;
        if !self.params.is_empty() {
            write_delim_list(f, &self.params, " <", ">", ", ")?;
        }
        match self.ty.kind {
            TypeKind::Inferred => {}
            _ => {
                write!(f, ": {}", self.ty)?;
            }
        }
        match &self.extend.expr {
            None => {}
            Some(expr) => {
                write!(f, " @ {}", expr)?;
            }
        }
        write!(f, "{} := {}", format_opt(&self.template_spec, " ", ""), self.expression)
    }
}

impl Display for TemplateSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut vec = Vec::new();
        if let Some(ref i) = self.inv {
            vec.push(format!("{}", i));
        }
        if let Some(ref e) = self.ext {
            vec.push(format!("{}", e));
        }
        if let Some(ref t) = self.ter {
            vec.push(format!("{}", t));
        }
        write_delim_list(f, &vec, "{ ", " }", " ")
    }
}

impl Display for Parameter {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.ty.kind {
            TypeKind::Inferred => write!(f, "{}", self.name),
            _ => write!(f, "{}: {}", self.name, self.ty),
        }
    }
}

impl Display for InvokeSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.condition {
            Some(ref c) => write!(f, "invoke {} {} {}", self.target, if self.is_if { "if" } else { "unless" }, c,),
            None => write!(f, "invoke {}", self.target),
        }
    }
}

impl Display for ExtendSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "extend {}", &self.target)
    }
}

impl Display for TimeSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let value = self.exact_period.to_integer();
        let abs_value = BigInt::from(value.abs());

        if !value.is_zero() {
            if (&abs_value).rem(10_u64.pow(9) * 60 * 60 * 24 * 365).is_zero() {
                let x: BigInt = abs_value / (10_u64.pow(9) * 60 * 60 * 24 * 365);
                return write!(f, "{}{:?}a", if value.is_negative() { "-" } else { "" }, x.to_u128().unwrap());
            }
            if (&abs_value).rem(10_u64.pow(9) * 60 * 60 * 24 * 7).is_zero() {
                let x: BigInt = abs_value / (10_u64.pow(9) * 60 * 60 * 24 * 7);
                return write!(f, "{}{:?}w", if value.is_negative() { "-" } else { "" }, x.to_u128().unwrap());
            }
            if (&abs_value).rem(10_u64.pow(9) * 60 * 60 * 24).is_zero() {
                let x: BigInt = abs_value / (10_u64.pow(9) * 60 * 60 * 24);
                return write!(f, "{}{:?}d", if value.is_negative() { "-" } else { "" }, x.to_u128().unwrap());
            }
            if (&abs_value).rem(10_u64.pow(9) * 60 * 60).is_zero() {
                let x: BigInt = abs_value / (10_u64.pow(9) * 60 * 60);
                return write!(f, "{}{:?}h", if value.is_negative() { "-" } else { "" }, x.to_u128().unwrap());
            }
            if (&abs_value).rem(10_u64.pow(9) * 60).is_zero() {
                let x: BigInt = abs_value / (10_u64.pow(9) * 60);
                return write!(f, "{}{:?}min", if value.is_negative() { "-" } else { "" }, x.to_u128().unwrap());
            }
            if (&abs_value).rem(10_u64.pow(9)).is_zero() {
                let x: BigInt = abs_value / (10_u64.pow(9));
                return write!(f, "{}{:?}s", if value.is_negative() { "-" } else { "" }, x.to_u128().unwrap());
            }
            if (&abs_value).rem(10_u64.pow(6)).is_zero() {
                let x: BigInt = abs_value / (10_u64.pow(6));
                return write!(f, "{}{:?}ms", if value.is_negative() { "-" } else { "" }, x.to_u128().unwrap());
            }
            if (&abs_value).rem(10_u64.pow(3)).is_zero() {
                let x: BigInt = abs_value / (10_u64.pow(3));
                return write!(f, "{}{:?}μs", if value.is_negative() { "-" } else { "" }, x.to_u128().unwrap());
            }
        }
        write!(f, "{}{:?}ns", if value.is_negative() { "-" } else { "" }, abs_value.to_u128().unwrap())
    }
}

impl Display for TerminateSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "terminate {}", self.target)
    }
}

impl Display for Trigger {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "trigger{} {}{}",
            format_opt(&self.name, " ", " :="),
            self.expression,
            format_opt(&self.message, " \"", "\""),
        )
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.kind)
    }
}

impl Display for TypeKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self {
            TypeKind::Simple(name) => write!(f, "{}", name),
            TypeKind::Tuple(types) => write_delim_list(f, types, "(", ")", ", "),
            TypeKind::Inferred => write!(f, "?"),
        }
    }
}

impl Display for TypeDeclField {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}: {}", &self.name, &self.ty)
    }
}

impl Display for TypeDeclaration {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "type {}", format_opt(&self.name, "", ""))?;
        write_delim_list(f, &self.fields, " { ", " }", ", ")
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self.kind {
            ExpressionKind::Lit(l) => write!(f, "{}", l),
            ExpressionKind::Ident(ident) => write!(f, "{}", ident),
            ExpressionKind::StreamAccess(expr, access) => match access {
                StreamAccessKind::Hold => write!(f, "{}.hold()", expr),
                StreamAccessKind::Optional => write!(f, "{}.get()", expr),
            },
            ExpressionKind::Default(expr, val) => write!(f, "{}.defaults(to: {})", expr, val),
            ExpressionKind::Offset(expr, val) => write!(f, "{}.offset(by: {})", expr, val),
            ExpressionKind::SlidingWindowAggregation { expr, duration, aggregation } => {
                write!(f, "{}.aggregate(over: {}, using: {})", expr, duration, aggregation)
            }
            ExpressionKind::Binary(op, lhs, rhs) => write!(f, "{} {} {}", lhs, op, rhs),
            ExpressionKind::Unary(operator, operand) => write!(f, "{}{}", operator, operand),
            ExpressionKind::Ite(cond, cons, alt) => write!(f, "if {} then {} else {}", cond, cons, alt),
            ExpressionKind::ParenthesizedExpression(left, expr, right) => {
                write!(f, "{}{}{}", if left.is_some() { "(" } else { "" }, expr, if right.is_some() { ")" } else { "" })
            }
            ExpressionKind::MissingExpression => Ok(()),
            ExpressionKind::Tuple(exprs) => write_delim_list(f, exprs, "(", ")", ", "),
            ExpressionKind::Function(name, types, args) => {
                write!(f, "{}", name.name)?;
                if !types.is_empty() {
                    write_delim_list(f, types, "<", ">", ", ")?;
                }
                let args: Vec<String> = args
                    .iter()
                    .zip(&name.arg_names)
                    .map(|(arg, arg_name)| match arg_name {
                        None => format!("{}", arg),
                        Some(arg_name) => format!("{}: {}", arg_name, arg),
                    })
                    .collect();
                write_delim_list(f, &args, "(", ")", ", ")
            }
            ExpressionKind::Field(expr, ident) => write!(f, "{}.{}", expr, ident),
            ExpressionKind::Method(expr, name, types, args) => {
                write!(f, "{}.{}", expr, name.name)?;
                if !types.is_empty() {
                    write_delim_list(f, types, "<", ">", ", ")?;
                }
                let args: Vec<String> = args
                    .iter()
                    .zip(&name.arg_names)
                    .map(|(arg, arg_name)| match arg_name {
                        None => format!("{}", arg),
                        Some(arg_name) => format!("{}: {}", arg_name, arg),
                    })
                    .collect();
                write_delim_list(f, &args, "(", ")", ", ")
            }
        }
    }
}

impl Display for FunctionName {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.name)?;
        let args: Vec<String> = self
            .arg_names
            .iter()
            .map(|arg_name| match arg_name {
                None => String::from("_:"),
                Some(arg_name) => format!("{}:", arg_name),
            })
            .collect();
        write_delim_list(f, &args, "(", ")", "")
    }
}

impl Display for Offset {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Offset::Discrete(val) => write!(f, "{}", val),
            Offset::RealTime(val, unit) => write!(f, "{}{}", val, unit),
        }
    }
}

impl Display for TimeUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{}",
            match self {
                TimeUnit::Nanosecond => "ns",
                TimeUnit::Microsecond => "μs",
                TimeUnit::Millisecond => "ms",
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
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{}",
            match self {
                WindowOperation::Sum => "Σ",
                WindowOperation::Product => "Π",
                WindowOperation::Average => "avg",
                WindowOperation::Count => "#",
                WindowOperation::Integral => "∫",
                WindowOperation::Min => "min",
                WindowOperation::Max => "max",
            }
        )
    }
}

impl Display for UnOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
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
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self.kind {
            LitKind::Bool(val) => write!(f, "{}", val),
            LitKind::Numeric(val, unit) => write!(f, "{}{}", val, unit.clone().unwrap_or_default()),
            LitKind::Str(s) => write!(f, "\"{}\"", s),
            LitKind::RawStr(s) => {
                // need to determine padding with `#`
                let mut padding = 0;
                while s.contains(&format!("{}\"", "#".repeat(padding))) {
                    padding += 1;
                }
                write!(f, "r{pad}\"{}\"{pad}", s, pad = "#".repeat(padding))
            }
        }
    }
}

impl Display for Ident {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.name)
    }
}

impl Display for BinOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
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
            BinOp::Ne => write!(f, "≠"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Ge => write!(f, "≥"),
        }
    }
}

impl Display for LanguageSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self {
            LanguageSpec::Classic => write!(f, "ClassicLola"),
            LanguageSpec::Lola2 => write!(f, "Lola 2.0"),
            LanguageSpec::RTLola => write!(f, "RTLola"),
        }
    }
}

impl Display for Import {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "import {}", self.name)
    }
}

impl Display for LolaSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if let Some(v) = &self.language {
            writeln!(f, "version {}", v)?
        }

        for import in &self.imports {
            writeln!(f, "{}", import)?;
        }
        for decl in &self.type_declarations {
            writeln!(f, "{}", decl)?;
        }
        for constant in &self.constants {
            writeln!(f, "{}", constant)?;
        }
        for input in &self.inputs {
            writeln!(f, "{}", input)?;
        }
        for output in &self.outputs {
            writeln!(f, "{}", output)?;
        }
        for trigger in &self.trigger {
            writeln!(f, "{}", trigger)?;
        }
        Ok(())
    }
}

impl FunctionName {
    pub(crate) fn as_string(&self) -> String {
        format!("{}", self)
    }
}
