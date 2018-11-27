//! This module contains helper to report messages (warnings/errors)

use self::Level::*;
use ast_node::Span;
use crate::parse::SourceMapper;
use std::cell::RefCell;
use std::io::Write;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

/// A handler is responsible for emitting warnings and errors
pub(crate) struct Handler {
    error_count: RefCell<usize>,
    emitter: RefCell<Box<dyn Emitter>>,
    mapper: SourceMapper,
}

impl Handler {
    pub(crate) fn new(mapper: SourceMapper) -> Self {
        Handler {
            error_count: RefCell::new(0),
            emitter: RefCell::new(Box::new(StderrEmitter::new())),
            mapper,
        }
    }

    pub(crate) fn contains_error(&self) -> bool {
        self.emitted_errors() > 0
    }

    pub(crate) fn emitted_errors(&self) -> usize {
        *self.error_count.borrow()
    }

    /// Displays diagnostic to user
    fn emit(&self, diagnostic: &Diagnostic) {
        if diagnostic.is_error() {
            let mut count = self.error_count.borrow_mut();
            *count += 1;
        }
        self.emitter.borrow_mut().emit(&self.mapper, &diagnostic)
    }

    pub(crate) fn error(&self, message: &str) {
        self.emit(&Diagnostic {
            level: Error,
            message: message.to_owned(),
            span: None,
            children: vec![],
        });
    }

    pub(crate) fn error_with_span(&self, message: &str, span: Span) {
        self.emit(&Diagnostic {
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
    fn emit(&mut self, mapper: &SourceMapper, diagnostic: &Diagnostic);
}

/// Emits errors to stderr
struct StderrEmitter {}

impl StderrEmitter {
    fn new() -> Self {
        StderrEmitter {}
    }
}

impl Emitter for StderrEmitter {
    fn emit(&mut self, mapper: &SourceMapper, diagnostic: &Diagnostic) {
        let mut stderr = StandardStream::stderr(ColorChoice::Always);
        for line in self.render(mapper, diagnostic) {
            for part in &line.strings {
                stderr
                    .set_color(&part.color)
                    .expect("cannot set output color");
                write!(&mut stderr, "{}", part.string);
            }
            writeln!(&mut stderr, "");
        }
        stderr.reset().expect("cannot reset output color");
    }
}

impl StderrEmitter {
    fn render(&mut self, mapper: &SourceMapper, diagnostic: &Diagnostic) -> Vec<ColoredLine> {
        let mut lines = Vec::new();

        // write header, e.g., `error: some error message`
        let mut line = ColoredLine::new();
        line.push(&diagnostic.level.to_str(), diagnostic.level.to_color());
        line.push(": ", ColorSpec::new());
        line.push(&diagnostic.message, ColorSpec::new().set_bold(true).clone());
        lines.push(line);

        // output source code
        if let Some(span) = diagnostic.span {
            // map span back to source code
            if let Some(line) = mapper.get_line(span) {
                let line_number_length = format!("{}", line.line_number).len();

                // path information
                let mut rendered_line = ColoredLine::new();
                rendered_line.push(&" ".repeat(line_number_length), ColorSpec::new());
                rendered_line.push("--> ", ColorSpec::new().set_fg(Some(Color::Blue)).clone());
                rendered_line.push(
                    &format!(
                        "{}:{}:{}",
                        line.path.display(),
                        line.line_number,
                        line.column_number,
                    ),
                    ColorSpec::new(),
                );
                lines.push(rendered_line);

                // source code snippet
                let mut rendered_line = ColoredLine::new();
                rendered_line.push(
                    &format!("{} | ", " ".repeat(line_number_length)),
                    ColorSpec::new().set_fg(Some(Color::Blue)).clone(),
                );
                lines.push(rendered_line);

                let mut rendered_line = ColoredLine::new();
                rendered_line.push(
                    &format!("{} | ", line.line_number),
                    ColorSpec::new().set_fg(Some(Color::Blue)).clone(),
                );
                rendered_line.push(&line.line, ColorSpec::new());
                lines.push(rendered_line);

                let mut rendered_line = ColoredLine::new();
                rendered_line.push(
                    &format!("{} | ", " ".repeat(line_number_length)),
                    ColorSpec::new().set_fg(Some(Color::Blue)).clone(),
                );
                rendered_line.push(
                    &format!(
                        "{}{}",
                        " ".repeat(line.highlight.start),
                        "^".repeat(line.highlight.end - line.highlight.start)
                    ),
                    diagnostic.level.to_color(),
                );
                lines.push(rendered_line);
            }
        }
        /*for child in &diagnostic.children {
            eprintln!("| {}: {}", child.level.to_str(), child.message);
            if let Some(span) = child.span {
                // TODO: actually map back to source code
                eprintln!("| {:?}", span);
            }
        }*/
        lines.push(ColoredLine::new());
        lines
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

    pub(crate) fn to_color(self) -> ColorSpec {
        let mut colorspec = ColorSpec::new();
        colorspec.set_intense(true).set_bold(true);
        match self {
            Bug | Fatal | Error => colorspec.set_fg(Some(Color::Red)),
            Warning => colorspec.set_fg(Some(Color::Yellow)),
            Note => colorspec.set_fg(Some(Color::Green)),
            Help => colorspec.set_fg(Some(Color::Cyan)),
        };
        colorspec
    }
}

#[derive(Debug)]
struct ColoredString {
    string: String,
    color: ColorSpec,
}

#[derive(Debug)]
struct ColoredLine {
    strings: Vec<ColoredString>,
}

impl ColoredLine {
    fn new() -> ColoredLine {
        ColoredLine {
            strings: Vec::new(),
        }
    }

    fn push(&mut self, string: &str, color: ColorSpec) {
        self.strings.push(ColoredString {
            string: string.to_owned(),
            color,
        })
    }
}
