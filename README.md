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

## Contributors
This project is based on research by the following people:
* [Jan Baumeister](https://www.react.uni-saarland.de/people/baumeister.html)
* [Peter Faymonville](https://www.react.uni-saarland.de/people/faymonville.html)
* [Bernd Finkbeiner](https://www.react.uni-saarland.de/people/finkbeiner.html)
* Florian Kohn
* [Malte Schledjewski](https://www.react.uni-saarland.de/people/schledjewski.html)
* [Maximilian Schwenger](https://www.react.uni-saarland.de/people/schwenger.html)
* [Leander Tentrup](https://www.react.uni-saarland.de/people/tentrup.html)
* [Hazem Torfah](https://www.react.uni-saarland.de/people/torfah.html)

Find out more on [rtlola.org](http://rtlola.org).
