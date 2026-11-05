"""Run a GPU under another GPU's per-GPU plan (owner, 5 November 2026: the A100 settings on the H200).

Put this directory first on ``PYTHONPATH`` and set ``FASTMACE_GPU_ALIAS="REAL NAME=ALIAS NAME"``, for example
``FASTMACE_GPU_ALIAS="NVIDIA H200=NVIDIA A100-SXM4-80GB"``. Every Python process (compile workers included) then sees
``torch.cuda.get_device_name()`` and ``torch.cuda.get_device_properties().name`` (the planner's ``detect_hardware``)
return the alias on that GPU, so the decision-66 sets keyed by device name choose the alias's plan. The first H200
run (5 November 2026) patched only ``get_device_name`` and so ran the generic plan. Compute capabilities stay real: rewrites that need sm_80 exactly (the CUTLASS TF32x3 cubin) still
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

        _original_properties = _cuda.get_device_properties

        class _AliasedProperties:
            """The device's properties with ``name`` replaced (the planner's ``detect_hardware`` reads it)."""

            def __init__(self, properties):
                object.__setattr__(self, "_properties", properties)

            def __getattr__(self, attribute):
                if attribute == "name" and self._properties.name == _real:
                    return _alias
                return getattr(self._properties, attribute)

            def __repr__(self):
                return repr(self._properties).replace(_real, _alias)

        def get_device_properties(device=None, _original=_original_properties):
            return _AliasedProperties(_original(device) if device is not None else _original())

        _cuda.get_device_name = get_device_name
        _cuda.get_device_properties = get_device_properties
        import torch as _torch

        _torch.cuda.get_device_name = get_device_name
        _torch.cuda.get_device_properties = get_device_properties
    except Exception as _error:  # noqa: BLE001  (never break interpreter start-up)
        import sys

        print(f"gpu_alias: not applied ({_error!r})", file=sys.stderr)
