# Development

## Environment

* Latest stable Rust compiler (<https://rustup.rs>)
* Code formating using `rustfmt` (<https://github.com/rust-lang-nursery/rustfmt>)

Recommended:

* Linting using `clippy` (<https://github.com/rust-lang-nursery/rust-clippy>)
* Visual Studio Code (<https://code.visualstudio.com>) has a pretty good Rust extension (<https://github.com/rust-lang-nursery/rls-vscode>)

## Workflow

Development happens in local branches and are merged into `master` using pull-requests.
The `master` branch should be stable, i.e., all tests should pass.
This is guaranteed by Continous Integration (CI) testing on pull-requests.

## Continous Integration

Currently, we use Gitlab CI to execute the following check:

* `cargo test` against stable and nightly compiler (nightly is allowed to fail)
* correct formatting using `rustfmt`
* linting using `clippy` (is allowed to fail)

For details, see `/.gitlab-ci.yml`.
