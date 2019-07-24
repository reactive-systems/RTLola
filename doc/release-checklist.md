# Release Checklist

Here is a pretty good overview of steps recommended for Rust CLI tools https://dev.to/sharkdp/my-release-checklist-for-rust-programs-1m33.

## Building

We provide binaries for Linux, macOS, and Windows (x86-64).

* Linux: Static build with musl (TODO: Marvin)
* Mac: `cargo build --release`
* Windows: (TODO: Malte)

## Packaging

We currently provide a single zip containing

* binaries (`streamlab-linux`, `streamlab-mac`, and `streamlab-windows`)
* evaluator readme (`/evaluator/readme.md`)
* syntax description (`/doc/syntax.md`)

using the naming convention `streamlab-TAG.zip`, e.g., `streamlab-0.1.0.zip`.
