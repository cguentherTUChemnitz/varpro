#!/usr/bin/env bash
# echo on
cargo fmt
cargo clippy --all-targets --features lapack-netlib -- -D warnings
cargo test
