# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Evaluator: Add pcap interface (see `ids` subcommand) (requires libpcap dependency)
- Frontend: Add pcap interface (see `ids` subcommand) (requires libpcap dependency)
- Language: Add bitwise operators and `&`, or `|`, xor `^`, left shift `<<`, and right shift `>>`
- Language: Add `Bytes` data type
- Language: Add method `Bytes.at(index:)`, e.g., `bytes.at(index: 0)`

## [0.2.0] - 2019-08-23
### Added
- Language: it is now possible to annotate optional types using the syntax `T?` to denote optional of type `T`, e.g., `Int64?`
- Language: support `min` and `max` as aggregations for sliding windows, e.g., `x.aggregate(over: 1s, using: min)`

### Fixed
- Frontend: Fix parsing problem related to keywords, e.g., `output outputxyz` is now be parsed (was refused before)
- Frontend: Ignore [BOM](https://de.wikipedia.org/wiki/Byte_Order_Mark) at the start of specification file
- Interpreter: Fix sliding window aggregation bug


## [0.1.0] - 2019-08-12
### Added
- Initial public release: parsing, type checking, memory analysis, lowering, and evaluation of StreamLAB specifications

