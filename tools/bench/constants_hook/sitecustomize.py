"""Set FastMACE module constants in every Python process (round 5, 5 November 2026: the force/stress gate and other
tools that start their own interpreters, run with an A/B switch on).

Put this directory first on ``PYTHONPATH`` and set ``FASTMACE_CONSTANTS="MODULE:NAME=VALUE;..."``, for example
``FASTMACE_CONSTANTS="fastmace.kernels.oeq_streaming.fused_forward:WGMMA=True"``; values are Python literals. The
modules are imported at start-up (``fastmace`` first: soft file locks before torch). For measurements only; record
the variable with the results. Without the variable this file does nothing.
"""

import os

_spec = os.environ.get("FASTMACE_CONSTANTS", "")
if _spec.strip():
    try:
        import ast
        import importlib

        import fastmace  # noqa: F401

        for _item in filter(None, (part.strip() for part in _spec.split(";"))):
            _target, _value = _item.split("=", 1)
            _module, _name = _target.split(":", 1)
            _mod = importlib.import_module(_module.strip())
            if not hasattr(_mod, _name.strip()):
                raise AttributeError(f"{_module} has no {_name}")
            setattr(_mod, _name.strip(), ast.literal_eval(_value.strip()))
    except Exception as _error:  # noqa: BLE001  (never break interpreter start-up)
        import sys

        print(f"constants_hook: not applied ({_error!r})", file=sys.stderr)
