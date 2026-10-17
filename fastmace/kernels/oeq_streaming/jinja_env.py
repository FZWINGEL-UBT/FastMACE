"""Jinja environment for the vendored receiver-streaming template (CUDA only).

Adapted from OpenEquivariance pull request 223, templates/jinja_utils.py
(BSD 3-Clause, see NOTICE.md): only the CUDA globals the template uses.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader


def cpp_scalar_type(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return "float"
    if dtype == np.dtype(np.float64):
        return "double"
    raise ValueError("generated kernels support f32 and f64 scalar types")


def cpp_scalar_literal(value, scalar):
    if scalar not in ("float", "double"):
        raise ValueError(f"Unsupported generated scalar type: {scalar}")
    suffix = "f" if scalar == "float" else ""
    return f"static_cast<scalar_t>({float(value).hex()}{suffix})"


@lru_cache(maxsize=8)
def get_jinja_environment(backend="cuda", warp_size=32):
    if backend != "cuda":
        raise ValueError("only the CUDA backend is vendored")
    env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)), extensions=["jinja2.ext.do"])
    env.globals.update(
        enumerate=enumerate, cpp_scalar_literal=cpp_scalar_literal, is_hip=False, is_sycl=False,
        warp_size=warp_size, syncwarp="__syncwarp()", atomic_add="atomicAdd",
        shfl_down=lambda val, offset: f"__shfl_down_sync(FULL_MASK, {val}, {offset})",
        shfl_down_width=lambda val, offset: f"__shfl_down_sync(FULL_MASK, {val}, {offset}, {warp_size})")
    return env
