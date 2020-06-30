# RTLola

RTLola is a monitoring framework.  It consist of a parser, analyzer, and interpreter for the RTLola specification language.

## Modules

The implementation is split into the following modules

* `frontend`: parsing, type checking, analysis, and lowering into an Intermediate Representation
* `interpreter`: an interpreter that runs a monitor based on the Intermediate Representation from the `frontend` 

## Documentation

* Syntax `doc/syntax.md`
* Development `doc/development.md`
* Rust API: `target/doc/rtlola_frontend/index.html`, `target/doc/rtlola_interpreter/index.html` (Note: if you want to build the documentation, use `cargo doc`)