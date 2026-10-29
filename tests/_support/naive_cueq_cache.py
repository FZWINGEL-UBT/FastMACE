"""A per-process cache of cuEquivariance's naive CPU kernels for the test suite (suite-speed, 27 October 2026).

Every cuEq conversion on the CPU builds its products with cuEq's naive method
(``fastmace.model.compile.naive_cueq``): for each segmented tensor product,
``cuequivariance_torch.primitives.segmented_polynomial_naive._tensor_product_fx`` traces an FX graph and
optimises its einsums (``opt_einsum_fx.optimize_einsums_full``). On murphy that took 0.4–0.6 s of the
0.6–0.8 s of one tiny-model conversion, 83 % of it in the einsum optimisation, and the CPU suite converts
about 1,300 times. The result depends only on the descriptor and the arguments, so this cache keeps the
first result per key and returns a deep copy of it: every caller still gets its own module and buffers
(no state is shared between models, SPEC §8.7), with the same graph and bitwise the same constants as a
fresh build. The key holds the descriptor (cuEq compares its paths' coefficients exactly), the
coefficients' dtypes, the device, the math dtype, the einsum flag and torch's default dtype, so a
conversion in one dtype never takes another dtype's result (the 2-T14 concern).

Tests only; the library is unchanged. ``FASTMACE_TEST_NAIVE_CUEQ_CACHE=0`` turns it off. Child processes
that tests start do not inherit it.
"""

from __future__ import annotations

import copy
import functools
import os

_installed = False


def _key(descriptor, device, math_dtype, optimize_einsums):
    import torch

    dtypes = tuple(str(path.coefficients.dtype) for path in descriptor.paths)
    return (descriptor, dtypes, str(device), str(math_dtype), bool(optimize_einsums), str(torch.get_default_dtype()))


def install():
    """Wrap ``_tensor_product_fx`` once per process; a no-op without cuEquivariance or when switched off."""
    global _installed
    if _installed or os.environ.get("FASTMACE_TEST_NAIVE_CUEQ_CACHE", "1") == "0":
        return
    try:
        from cuequivariance_torch.primitives import segmented_polynomial_naive as naive
    except ImportError:
        return
    build = naive._tensor_product_fx
    cache = {}

    @functools.wraps(build)
    def cached(descriptor, device, math_dtype, optimize_einsums):
        key = _key(descriptor, device, math_dtype, optimize_einsums)
        module = cache.get(key)
        if module is None:
            module = cache[key] = build(descriptor, device, math_dtype, optimize_einsums)
        return copy.deepcopy(module)

    cached.cache = cache
    cached.uncached = build
    naive._tensor_product_fx = cached
    _installed = True
