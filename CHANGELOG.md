# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Language: type annotation `T?` to denote optional of type `T`, e.g., `Int64?`

### Fixed
- Frontend: Fix parsing problem, e.g., `output outputxyz` can now be parsed
- Frontend: Ignore [BOM](https://de.wikipedia.org/wiki/Byte_Order_Mark) at the start of a Lola specification
- Interpreter: Fix sliding window aggregation bug


## [0.1.0] - 2019-08-12
### Added
- Initial public release: parsing, type checking, memory analysis, lowering, and evaluation of StreamLAB specifications

