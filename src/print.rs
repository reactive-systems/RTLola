//! This module contains `Display` implementations for the AST.

use super::ast::*;
use super::parse::Ident;
use std::fmt::{Display, Formatter, Result};

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
        match &self.kind {
            TypeKind::Simple(name) => write!(f, "{}", name),
            _ => unimplemented!(),
        }
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match &self.kind {
            ExpressionKind::Lit(l) => write!(f, "{}", l),
            ExpressionKind::Ident(ident) => write!(f, "{}", ident),
            ExpressionKind::Binary(op, lhs, rhs) => write!(f, "{} {} {}", lhs, op, rhs),
            ExpressionKind::ParenthesizedExpression(left, expr, right) => write!(
                f,
                "{}{}{}",
                if left.is_some() { "(" } else { "" },
                expr,
                if right.is_some() { ")" } else { "" }
            ),
            _ => unimplemented!(),
        }
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match &self.kind {
            LitKind::Bool(val) => write!(f, "{}", val),
            LitKind::Int(i) => write!(f, "{}", i),
            LitKind::Float(fl) => write!(f, "{}", fl),
            _ => unimplemented!(),
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
            BinOp::Eq => write!(f, "="),
            BinOp::Ne => write!(f, "!="),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
            _ => unimplemented!(),
        }
    }
}
