//! This module contains `Display` implementations for the AST.

use super::ast::*;
use super::parse::Ident;
use std::fmt::{Display, Formatter, Result};

/// Writes out the joined vector `v`, enclosed by the given strings `pref` and `suff`.
pub(crate) fn write_delim_list<T: Display>(
    f: &mut Formatter,
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
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "constant {}{} := {}",
            self.name,
            format_type(&self.ty),
            self.literal
        )
    }
}

impl Display for Input {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "input {}", self.name)?;
        if !self.params.is_empty() {
            write_delim_list(f, &self.params, " <", ">", ", ")?;
        }
        write!(f, ": {}", self.ty)
    }
}

impl Display for Output {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "output {}", self.name)?;
        if !self.params.is_empty() {
            write_delim_list(f, &self.params, " <", ">", ", ")?;
        }
        write!(
            f,
            "{}{} := {}",
            format_type(&self.ty),
            format_opt(&self.template_spec, " ", ""),
            self.expression
        )
    }
}

impl Display for TemplateSpec {
    fn fmt(&self, f: &mut Formatter) -> Result {
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
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}{}", self.name, format_type(&self.ty))
    }
}

impl Display for InvokeSpec {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self.condition {
            Some(ref c) => write!(
                f,
                "invoke {} {} {}",
                self.target,
                if self.is_if { "if" } else { "unless" },
                c,
            ),
            None => write!(f, "invoke {}", self.target),
        }
    }
}

impl Display for ExtendSpec {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "extend {}{}",
            format_opt(&self.target, "", ""),
            format_opt(&self.freq, " @ ", ""),
        )
    }
}

impl Display for ExtendRate {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            ExtendRate::Frequency(e, fu) => write!(f, "{}{}", e, fu),
            ExtendRate::Duration(e, tu) => write!(f, "{}{}", e, tu),
        }
    }
}

impl Display for TerminateSpec {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "terminate {}", self.target)
    }
}

impl Display for Trigger {
    fn fmt(&self, f: &mut Formatter) -> Result {
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
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.kind)
    }
}

impl Display for TypeKind {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match &self {
            TypeKind::Simple(name) => write!(f, "{}", name),
            TypeKind::Malformed(s) => write!(f, "{}", s),
            TypeKind::Tuple(types) => write_delim_list(f, types, "(", ")", ", "),
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
        write!(f, "type {}", format_opt(&self.name, "", ""))?;
        write_delim_list(f, &self.fields, " { ", " }", ", ")
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
            ExpressionKind::Tuple(exprs) => write_delim_list(f, exprs, "(", ")", ", "),
            ExpressionKind::Function(kind, args) => {
                write!(f, "{}", kind)?;
                write_delim_list(f, args, "(", ")", ", ")
            }
        }
    }
}

impl Display for StreamInstance {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.stream_identifier)?;
        if !self.arguments.is_empty() {
            write_delim_list(f, &self.arguments, "(", ")", ", ")
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
                FunctionKind::Arcsin => "arcsin",
                FunctionKind::Arccos => "arccos",
                FunctionKind::Arctan => "arctan",
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

impl Display for FreqUnit {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "{}",
            match self {
                FreqUnit::MicroHertz => "μHz",
                FreqUnit::MilliHertz => "mHz",
                FreqUnit::Hertz => "Hz",
                FreqUnit::KiloHertz => "kHz",
                FreqUnit::MegaHertz => "MHz",
                FreqUnit::GigaHertz => "GHz",
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
            BinOp::Ne => write!(f, "≠"),
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
            writeln!(f, "version {}", v)?
        }

        for decl in &self.type_declarations {
            writeln!(f, "{}", decl);
        }
        for constant in &self.constants {
            writeln!(f, "{}", constant);
        }
        for input in &self.inputs {
            writeln!(f, "{}", input);
        }
        for output in &self.outputs {
            writeln!(f, "{}", output);
        }
        for trigger in &self.trigger {
            writeln!(f, "{}", trigger);
        }
        Ok(())
    }
}
