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

    fn format_opt<T: Display>(opt: &Option<T>, pref: &str, suff: &str) -> String {
        if let Some(ref e) = opt {
            format!("{}{}{}", pref, e, suff)
        } else {
            String::new()
        }
    }

    fn format_type(ty: &Option<Type>) -> String {
        PrintHelper::format_opt(ty, ": ", "")
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "constant {}{} := {}",
            self.name,
            PrintHelper::format_type(&self.ty),
            self.literal
        )
    }
}

impl Display for Input {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "input {}", self.name)?;
        if !self.params.is_empty() {
            PrintHelper::write(f, &self.params, " <", ">", ", ")?;
        }
        write!(f, ": {}", self.ty)
    }
}

impl Display for Output {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "output {}", self.name)?;
        if !self.params.is_empty() {
            PrintHelper::write(f, &self.params, " <", ">", ", ")?;
        }
        write!(
            f,
            "{}{} := {}",
            PrintHelper::format_type(&self.ty),
            PrintHelper::format_opt(&self.template_spec, " ", ""),
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
        PrintHelper::write(f, &vec, "{ ", " }", " ")
    }
}

impl Display for Parameter {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}{}", self.name, PrintHelper::format_type(&self.ty))
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
            PrintHelper::format_opt(&self.target, "", ""),
            PrintHelper::format_opt(&self.freq, " @ ", ""),
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
            PrintHelper::format_opt(&self.name, " ", " :="),
            self.expression,
            PrintHelper::format_opt(&self.message, " \"", "\""),
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
        write!(f, "type {}", PrintHelper::format_opt(&self.name, "", ""))?;
        PrintHelper::write(f, &self.fields, " { ", " }", ", ")
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

impl Display for Offset {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            Offset::DiscreteOffset(e) => write!(f, "{}", e),
            Offset::RealTimeOffset(e, u) => write!(f, "{}{}", e, u),
        }
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
