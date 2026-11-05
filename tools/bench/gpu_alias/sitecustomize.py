"""Run a GPU under another GPU's per-GPU plan (owner, 5 November 2026: the A100 settings on the H200).

Put this directory first on ``PYTHONPATH`` and set ``FASTMACE_GPU_ALIAS="REAL NAME=ALIAS NAME"``, for example
``FASTMACE_GPU_ALIAS="NVIDIA H200=NVIDIA A100-SXM4-80GB"``. Every Python process (compile workers included) then sees
``torch.cuda.get_device_name()`` return the alias on that GPU, so the decision-66 sets keyed by device name choose
the alias's plan. Compute capabilities stay real: rewrites that need sm_80 exactly (the CUTLASS TF32x3 cubin) still
refuse on other architectures. For measurements only; provenance records the alias name, so label results by the
job's ``nvidia-smi`` line. Without the variable this file does nothing.
"""

import os

_spec = os.environ.get("FASTMACE_GPU_ALIAS", "")
if "=" in _spec:
    _real, _alias = (part.strip() for part in _spec.split("=", 1))
    try:
        import torch.cuda as _cuda

        _original = _cuda.get_device_name

        def get_device_name(device=None, _original=_original, _real=_real, _alias=_alias):
            name = _original(device)
            return _alias if name == _real else name

        _cuda.get_device_name = get_device_name
        import torch as _torch

        _torch.cuda.get_device_name = get_device_name
    except Exception as _error:  # noqa: BLE001  (never break interpreter start-up)
        import sys

        print(f"gpu_alias: not applied ({_error!r})", file=sys.stderr)
