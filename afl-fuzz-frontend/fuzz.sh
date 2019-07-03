#!/usr/bin/env bash

echo "building afl..."
RUSTFLAGS='-C codegen-units=1' cargo afl build
cargo afl fuzz -i in -o out target/debug/afl-fuzz-frontend
