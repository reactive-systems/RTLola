#!/usr/bin/env bash

rm -rf in out
mkdir in
mkdir out
cp ../tests/specs/* in/
RUSTFLAGS='-C codegen-units=1' cargo +nightly afl build
cargo afl fuzz -i in -o out target/debug/afl-fuzz-frontend
