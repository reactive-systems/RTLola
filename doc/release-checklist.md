# Release Checklist

Here is a pretty good overview of steps recommended for Rust CLI tools https://dev.to/sharkdp/my-release-checklist-for-rust-programs-1m33.

## Building

We provide binaries for Linux, macOS, and Windows (x86-64).
See also [CLI Workgroup Tutorial on Packaging Rust](https://rust-lang-nursery.github.io/cli-wg/tutorial/packaging.html).

### Linux

```shell
export TARGET=x86_64-unknown-linux-musl
rustup target add $TARGET
cargo build --release --target $TARGET --locked --features public
```

### macOS

```shell
export MACOSX_DEPLOYMENT_TARGET=10.7
export TARGET=x86_64-apple-darwin
cargo build --release --target $TARGET --locked --features public
```

### Windows

TODO(@malte)

## Packaging

We currently provide a single zip containing

* binaries (`streamlab-linux`, `streamlab-mac`, and `streamlab-windows.exe`)
* evaluator readme (`/evaluator/readme.md`)
* syntax description (`/doc/syntax.md`)

using the naming convention `streamlab-TAG.zip`, e.g., `streamlab-0.1.0.zip`.
