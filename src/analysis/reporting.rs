//! This module contains helper to report messages (warnings/errors)

use self::Level::*;
use ast_node::Span;
use std::cell::RefCell;

/// A handler is responsible for emitting warnings and errors
pub(crate) struct Handler {
    error_count: RefCell<usize>,
    emitter: RefCell<Box<dyn Emitter>>,
}

impl Handler {
    pub(crate) fn new() -> Self {
        Handler {
            error_count: RefCell::new(0),
            emitter: RefCell::new(Box::new(StderrEmitter::new())),
        }
    }

    pub(crate) fn contains_error(&self) -> bool {
        *self.error_count.borrow() > 0
    }

    pub(crate) fn emitted_errors(&self) -> usize {
        *self.error_count.borrow()
    }

    /// Displays diagnostic to user
    fn emit(&self, diagnostic: Diagnostic) {
        if diagnostic.is_error() {
            let mut count = self.error_count.borrow_mut();
            *count += 1;
        }
        self.emitter.borrow_mut().emit(&diagnostic)
    }

    pub(crate) fn error(&self, message: &str) {
        self.emit(Diagnostic {
            level: Error,
            message: message.to_owned(),
            span: None,
            children: vec![],
        });
    }

    pub(crate) fn error_with_span(&self, message: &str, span: Span) {
        self.emit(Diagnostic {
            level: Error,
            message: message.to_owned(),
            span: Some(span),
            children: vec![],
        });
    }
}

/// Emitter trait for emitting errors.
pub(crate) trait Emitter {
    /// Emit a structured diagnostic.
    fn emit(&mut self, diagnostic: &Diagnostic);
}

/// Emits errors to stderr
struct StderrEmitter {}

impl StderrEmitter {
    fn new() -> Self {
        StderrEmitter {}
    }
}

impl Emitter for StderrEmitter {
    fn emit(&mut self, diagnostic: &Diagnostic) {
        eprintln!("{}: {}", diagnostic.level.to_str(), diagnostic.message);
        if let Some(span) = diagnostic.span {
            // TODO: actually map back to source code
            eprintln!("{:?}", span);
        }
        for child in &diagnostic.children {
            eprintln!("| {}: {}", child.level.to_str(), child.message);
            if let Some(span) = child.span {
                // TODO: actually map back to source code
                eprintln!("| {:?}", span);
            }
        }
        eprintln!("");
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Level {
    /// A compiler bug
    Bug,
    /// A fatal error, immediate exit afterwards
    Fatal,
    Error,
    Warning,
    Note,
    Help,
}

/// A structured representation of a user-facing diagnostic.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: Level,
    pub message: String,
    pub span: Option<Span>,
    pub children: Vec<SubDiagnostic>,
}

impl Diagnostic {
    fn is_error(&self) -> bool {
        match self.level {
            Bug | Fatal | Error => true,
            Warning | Note | Help => false,
        }
    }
}

/// For example a note attached to an error.
#[derive(Debug, Clone)]
pub struct SubDiagnostic {
    pub level: Level,
    pub message: String,
    pub span: Option<Span>,
}

impl Level {
    pub fn to_str(self) -> &'static str {
        match self {
            Bug => "error: internal compiler error",
            Fatal | Error => "error",
            Warning => "warning",
            Note => "note",
            Help => "help",
        }
    }
}
