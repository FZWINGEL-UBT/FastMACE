#!/usr/bin/env bash
# Build the OPT-T01 CUTLASS 3xTF32 bench on the Colab A100 (header-only CUTLASS at the newest release tag).
set -eu
W=/content/cutlass_bench
cd "$W/src"
if [ ! -d "$W/cutlass" ]; then
  TAG=$(git ls-remote --tags --refs https://github.com/NVIDIA/cutlass | awk -F/ '{print $3}' \
        | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1)
  git clone -q --depth 1 --branch "$TAG" https://github.com/NVIDIA/cutlass "$W/cutlass"
  echo "$TAG $(git -C "$W/cutlass" rev-parse HEAD)" > "$W/cutlass_version.txt"
fi
cat "$W/cutlass_version.txt"
NVCC="nvcc -arch=sm_80 -O3 -std=c++17 -DNDEBUG -I$W/cutlass/include -I$W/cutlass/tools/util/include -I$W/src"
echo "$NVCC -c <each .cu>; nvcc -arch=sm_80 -o bench *.o -lcublas" > "$W/build_command.txt"
mkdir -p "$W/obj"
ls *.cu | xargs -P 12 -I{} sh -c "$NVCC -c {} -o $W/obj/{}.o > $W/obj/{}.log 2>&1 || { echo FAILED {}; $NVCC -DCUTLASS_BENCH_STUB -c {} -o $W/obj/{}.o; }"
for f in $W/obj/*.log; do [ -s "$f" ] && { echo "== $f"; tail -20 "$f"; }; done
nvcc -arch=sm_80 -o "$W/bench" $W/obj/*.o -lcublas
echo built "$W/bench"
