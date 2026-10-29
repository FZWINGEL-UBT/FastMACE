#!/usr/bin/env bash
# The Modal job's source tarball: a git archive of COMMIT with .source_commit, plus tc_gemm_lib/ (the CUTLASS library
# built by tools/bench/tc_gemm/build.sh in BUILD, with its CUTLASS commit and nvcc version).
#   bash make_tarball.sh REPO COMMIT BUILD OUT.tar.gz
set -euo pipefail
REPO=$1 COMMIT=$2 BUILD=$3 OUT=$4
stage=$(mktemp -d "${TMPDIR:-$HOME}/tc-gemm-tarball.XXXXXX")
git -C "$REPO" -c core.autocrlf=false archive "$COMMIT" | tar -x -C "$stage"
git -C "$REPO" rev-parse "$COMMIT" > "$stage/.source_commit"
mkdir "$stage/tc_gemm_lib"
cp "$BUILD/libfm_tc_gemm.so" "$BUILD/cutlass_commit.txt" "$BUILD/nvcc.txt" "$stage/tc_gemm_lib/"
tar -czf "$OUT" -C "$stage" .
rm -rf "$stage"
ls -la "$OUT"
