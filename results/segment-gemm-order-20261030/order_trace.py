"""Which tile order segment_gemm's launches get inside an A/B run: wraps ``segment_gemm._gemm`` and counts, per
TILE_ORDER in force at the call, the launches with the columns-together order on and off and their row counts.
Kernels replayed from a CUDA graph do not pass through Python, so the counts are the captures and eager calls.
    python order_trace.py <ab_rewrites.py arguments>
"""

import collections
import runpy
import sys

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
from fastmace.kernels.triton import segment_gemm as sg
from fastmace.kernels.triton import tile_rule

calls = collections.Counter()
original = sg._gemm


def traced(*args):
    choice, rows = args[11], args[9]
    calls[(tile_rule.TILE_ORDER, choice.column_fastest, rows)] += 1
    return original(*args)


sg._gemm = traced
tool = sys.argv[1]
sys.argv = sys.argv[1:]
try:
    runpy.run_path(tool, run_name="__main__")
finally:
    for key, count in sorted(calls.items(), key=str):
        print(f"TRACE order={key[0]} column_fastest={key[1]} rows={key[2]}: {count}", flush=True)
