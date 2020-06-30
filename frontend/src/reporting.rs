//! This module contains helper to report messages (warnings/errors)

use self::Level::*;
use crate::parse::Span;
use crate::parse::{CodeLine, SourceMapper};
use std::cell::RefCell;
#[cfg(not(test))]
use std::io::Write;
use termcolor::{Color, ColorSpec};
#[cfg(not(test))]
use termcolor::{ColorChoice, StandardStream, WriteColor};

/// A handler is responsible for emitting warnings and errors
#[derive(Debug)]
pub(crate) struct Handler {
    error_count: RefCell<usize>,
    warning_count: RefCell<usize>,
    emitter: RefCell<Box<dyn Emitter>>,
    mapper: SourceMapper,
}

impl Handler {
    pub(crate) fn new(mapper: SourceMapper) -> Self {
        Handler {
            error_count: RefCell::new(0),
            warning_count: RefCell::new(0),
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

    #[allow(dead_code)]
    pub(crate) fn emitted_warnings(&self) -> usize {
        *self.warning_count.borrow()
    }

    /// Displays diagnostic to user
    fn emit(&self, diagnostic: &Diagnostic) {
        if diagnostic.is_error() {
            let mut count = self.error_count.borrow_mut();
            *count += 1;
        }
        if diagnostic.is_warning() {
            let mut count = self.warning_count.borrow_mut();
            *count += 1;
        }
        self.emitter.borrow_mut().emit(&self.mapper, &diagnostic)
    }

    #[allow(dead_code)]
    pub(crate) fn warn(&self, message: &str) {
        self.emit(&Diagnostic {
            level: Warning,
            message: message.to_owned(),
            span: Vec::new(),
            children: vec![],
            sort_spans: true,
        });
    }

    pub(crate) fn warn_with_span(&self, message: &str, span: LabeledSpan) {
        self.emit(&Diagnostic {
            level: Warning,
            message: message.to_owned(),
            span: vec![span],
            children: vec![],
            sort_spans: true,
        });
    }

    pub(crate) fn error(&self, message: &str) {
        self.emit(&Diagnostic {
            level: Error,
            message: message.to_owned(),
            span: Vec::new(),
            children: vec![],
            sort_spans: true,
        });
    }

    pub(crate) fn error_with_span(&self, message: &str, span: LabeledSpan) {
        self.emit(&Diagnostic {
            level: Error,
            message: message.to_owned(),
            span: vec![span],
            children: vec![],
            sort_spans: true,
        });
    }

    pub(crate) fn build_error_with_span(&self, message: &str, span: LabeledSpan) -> DiagnosticBuilder<'_> {
        let mut builder = DiagnosticBuilder::new(&self, Error, message);
        builder.add_labeled_span(span);
        builder
    }

    pub(crate) fn build_diagnostic(&self, message: &str, level: Level) -> DiagnosticBuilder<'_> {
        DiagnosticBuilder::new(&self, level, message)
    }

    #[allow(dead_code)]
    pub(crate) fn bug_with_span(&self, message: &str, span: LabeledSpan) {
        self.emit(&Diagnostic {
            level: Bug,
            message: message.to_owned(),
            span: vec![span],
            children: vec![],
            sort_spans: true,
        });
    }
}

/// Emitter trait for emitting errors.
pub(crate) trait Emitter: std::fmt::Debug {
    /// Emit a structured diagnostic.
    fn emit(&mut self, mapper: &SourceMapper, diagnostic: &Diagnostic);
}

/// Emits errors to stderr
#[derive(Debug)]
struct StderrEmitter {}

impl StderrEmitter {
    fn new() -> Self {
        StderrEmitter {}
    }
}

impl Emitter for StderrEmitter {
    /// Implement two versions of emit in order to suppress output when testing.

    /// standard emit implementation
    #[cfg(not(test))]
    fn emit(&mut self, mapper: &SourceMapper, diagnostic: &Diagnostic) {
        let mut stderr = StandardStream::stderr(ColorChoice::Always);
        for line in self.render(mapper, diagnostic) {
            for part in &line.strings {
                stderr.set_color(&part.color).expect("cannot set output color");
                write!(&mut stderr, "{}", part.string).expect("writing to stderr failed");
            }
            writeln!(&mut stderr).expect("writing to stderr failed");
        }
        stderr.reset().expect("cannot reset output color");
        stderr.flush().expect("flushing stderr failed");
    }

    /// test emit implementation
    #[cfg(test)]
    fn emit(&mut self, _mapper: &SourceMapper, _diagnostic: &Diagnostic) {}
}

impl StderrEmitter {
    #[allow(dead_code)]
    fn render(&mut self, mapper: &SourceMapper, diagnostic: &Diagnostic) -> Vec<ColoredLine> {
        let mut lines = Vec::new();

        // write header, e.g., `error: some error message`
        let mut line = ColoredLine::new();
        line.push(&diagnostic.level.to_str(), diagnostic.level.to_color());
        line.push(": ", ColorSpec::new());
        line.push(&diagnostic.message, ColorSpec::new().set_bold(true).clone());
        lines.push(line);

        // output source code snippet with annotations
        // first, try to get code lines from spans
        let mut snippets: Vec<(CodeLine, Option<String>, bool)> = diagnostic
            .span
            .iter()
            .flat_map(|s| mapper.get_line(s.span).map(|l| (l, s.label.clone(), s.primary)))
            .collect();

        if !snippets.is_empty() && snippets.len() == diagnostic.span.len() {
            let line_number_length =
                snippets.iter().map(|(s, _, _)| format!("{}", s.line_number).len()).fold(0, std::cmp::max);

            // we assume the first span is the main one, i.e., we output path information
            let path = {
                let (main, _, _) = snippets.first().unwrap();

                // emit path information
                let mut rendered_line = ColoredLine::new();
                rendered_line.push(&" ".repeat(line_number_length), ColorSpec::new());
                rendered_line.push("--> ", ColorSpec::new().set_fg(Some(Color::Blue)).clone());
                rendered_line.push(
                    &format!("{}:{}:{}", main.path.display(), main.line_number, main.column_number,),
                    ColorSpec::new(),
                );
                lines.push(rendered_line);
                main.path.clone()
            };

            // we sort the code lines, i.e., earlier lines come first
            if diagnostic.sort_spans {
                snippets.sort_unstable();
            }

            let mut prev_line_number = None;
            let mut num_messages = 0;

            for (snippet, label, primary) in snippets {
                fn render_source_line(snippet: &CodeLine) -> ColoredLine {
                    let mut rendered_line = ColoredLine::new();
                    rendered_line.push(
                        &format!("{} | ", snippet.line_number),
                        ColorSpec::new().set_fg(Some(Color::Blue)).clone(),
                    );
                    rendered_line.push(&snippet.line, ColorSpec::new());
                    rendered_line
                }

                assert_eq!(path, snippet.path, "assume snippets to be in same source file, use `SubDiagnostic` if not");

                // source code snippet
                if let Some(prev_line_number) = prev_line_number {
                    //                    assert!(prev_line_number.unwrap() <= snippet.line_number);
                    if diagnostic.sort_spans && prev_line_number + 1 < snippet.line_number {
                        // print ...
                        let mut rendered_line = ColoredLine::new();
                        rendered_line.push("...", ColorSpec::new().set_fg(Some(Color::Blue)).clone());
                        lines.push(rendered_line);
                    }

                    if prev_line_number != snippet.line_number {
                        // do not print line twice
                        lines.push(render_source_line(&snippet));
                    }
                } else {
                    // print leading space
                    let mut rendered_line = ColoredLine::new();
                    rendered_line.push(
                        &format!("{} | ", " ".repeat(line_number_length)),
                        ColorSpec::new().set_fg(Some(Color::Blue)).clone(),
                    );
                    lines.push(rendered_line);

                    lines.push(render_source_line(&snippet));
                }
                prev_line_number = Some(snippet.line_number);

                let color = if primary {
                    diagnostic.level.to_color()
                } else {
                    let mut colorspec = ColorSpec::new();
                    colorspec.set_intense(true).set_bold(true).set_fg(Some(Color::Blue));
                    colorspec
                };

                if num_messages > 0 {
                    // add an empty line
                    let mut empty_line = ColoredLine::new();
                    empty_line.push(
                        &format!("{} | ", " ".repeat(line_number_length)),
                        ColorSpec::new().set_fg(Some(Color::Blue)).clone(),
                    );
                    empty_line.push(&format!("{}|", " ".repeat(snippet.highlight.start)), color.clone());
                    lines.push(empty_line);
                }

                let mut rendered_line = ColoredLine::new();
                rendered_line.push(
                    &format!("{} | ", " ".repeat(line_number_length)),
                    ColorSpec::new().set_fg(Some(Color::Blue)).clone(),
                );
                let highlight_char: String =
                    if primary && num_messages == 0 { String::from("^") } else { "-".repeat(num_messages + 1) };

                rendered_line.push(
                    &format!(
                        "{}{}",
                        " ".repeat(snippet.highlight.start),
                        highlight_char.repeat(snippet.highlight.end - snippet.highlight.start)
                    ),
                    color.clone(),
                );
                if let Some(label) = label {
                    rendered_line.push(&format!(" {}", label), color);
                }
                lines.push(rendered_line);
                num_messages += 1;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Level {
    /// A compiler bug
    #[allow(dead_code)]
    Bug,
    /// A fatal error, immediate exit afterwards
    #[allow(dead_code)]
    Fatal,
    Error,
    Warning,
    #[allow(dead_code)]
    Note,
    #[allow(dead_code)]
    Help,
}

/// A structured representation of a user-facing diagnostic.
#[derive(Debug, Clone)]
pub(crate) struct Diagnostic {
    pub(crate) level: Level,
    pub(crate) message: String,
    pub(crate) span: Vec<LabeledSpan>,
    pub(crate) children: Vec<SubDiagnostic>,
    pub(crate) sort_spans: bool,
}

impl Diagnostic {
    fn is_error(&self) -> bool {
        match self.level {
            Bug | Fatal | Error => true,
            Warning | Note | Help => false,
        }
    }
    fn is_warning(&self) -> bool {
        match self.level {
            Bug | Fatal | Error | Note | Help => false,
            Warning => true,
        }
    }
}

/// For example a note attached to an error.
#[derive(Debug, Clone)]
pub(crate) struct SubDiagnostic {
    pub(crate) level: Level,
    pub(crate) message: String,
    pub(crate) span: Option<Span>,
}

impl Level {
    pub(crate) fn to_str(self) -> &'static str {
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

/// Show a label (message) next to the position in source code
#[derive(Debug, Clone)]
pub(crate) struct LabeledSpan {
    span: Span,
    label: Option<String>,
    primary: bool,
}

impl LabeledSpan {
    pub(crate) fn new(span: Span, label: &str, primary: bool) -> Self {
        LabeledSpan { span, label: Some(label.to_string()), primary }
    }
}

/// Sometimes diagnostics cannot be emitted directly as important information is still missing.
/// `DiagnosticBuilder` helps in this situations by allowing to incrementally build diagnostics.
#[derive(Debug)]
pub(crate) struct DiagnosticBuilder<'a> {
    handler: &'a Handler,
    diagnostic: Diagnostic,
    status: DiagnosticBuilderStatus,
}

impl<'a> DiagnosticBuilder<'a> {
    fn new(handler: &'a Handler, level: Level, messgage: &str) -> Self {
        DiagnosticBuilder {
            handler,
            diagnostic: Diagnostic {
                level,
                message: messgage.to_string(),
                span: Vec::new(),
                children: Vec::new(),
                sort_spans: true,
            },
            status: DiagnosticBuilderStatus::Building,
        }
    }

    pub(crate) fn emit(&mut self) {
        assert_eq!(self.status, DiagnosticBuilderStatus::Building);
        self.handler.emit(&self.diagnostic);
        self.status = DiagnosticBuilderStatus::Emitted;
    }

    #[allow(dead_code)]
    pub(crate) fn cancel(&mut self) {
        assert_eq!(self.status, DiagnosticBuilderStatus::Building);
        self.status = DiagnosticBuilderStatus::Cancelled;
    }

    pub(crate) fn prevent_sorting(&mut self) {
        assert_eq!(self.status, DiagnosticBuilderStatus::Building);
        self.diagnostic.sort_spans = false;
    }

    pub(crate) fn add_span_with_label(&mut self, span: Span, label: &str, primary: bool) {
        self.diagnostic.span.push(LabeledSpan::new(span, label, primary))
    }

    pub(crate) fn add_labeled_span(&mut self, span: LabeledSpan) {
        self.diagnostic.span.push(span)
    }
}

impl<'a> Drop for DiagnosticBuilder<'a> {
    fn drop(&mut self) {
        // make sure that diagnostic is either emitted or canceled
        if self.status == DiagnosticBuilderStatus::Building {
            panic!("Diagnostic was build but was neither emitted, nor cancelled.");
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum DiagnosticBuilderStatus {
    Building,
    Emitted,
    Cancelled,
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
        ColoredLine { strings: Vec::new() }
    }

    fn push(&mut self, string: &str, color: ColorSpec) {
        self.strings.push(ColoredString { string: string.to_owned(), color })
    }
}
