# Release Checklist

Here is a pretty good overview of steps recommended for Rust CLI tools https://dev.to/sharkdp/my-release-checklist-for-rust-programs-1m33.

## Building

We provide binaries for Linux, macOS, and Windows (x86-64).
See also [CLI Workgroup Tutorial on Packaging Rust](https://rust-lang-nursery.github.io/cli-wg/tutorial/packaging.html).

### Linux

Docker image: `rust:latest`

```bash
apt install musl-dev musl-tools clang
export TARGET=x86_64-unknown-linux-musl
rustup target add $TARGET
cargo build --release --target $TARGET --locked --features public
```

### macOS

```bash
export MACOSX_DEPLOYMENT_TARGET=10.7
export TARGET=x86_64-apple-darwin
cargo build --release --target $TARGET --locked --features public
```

### Windows

```bash
set RUSTFLAGS=-C target-feature=+crt-static 
rustup run stable-msvc cargo build --release --target x86_64-pc-windows-msvc --locked --features public
```
Or
```bash
rustup run stable-gnu cargo build --release --target x86_64-pc-windows-gnu --locked --features public
```

## Packaging

We currently provide a single zip containing

* binaries (`streamlab-linux`, `streamlab-mac`, and `streamlab-windows.exe`)
* evaluator readme (`/evaluator/readme.md`)
* syntax description (`/doc/syntax.md`)
* changelog (`/CHANGELOG.md`)
* ids description (`/doc/streamlab-ids.md`)

using the naming convention `streamlab-TAG.zip`, e.g., `streamlab-0.1.0.zip`.