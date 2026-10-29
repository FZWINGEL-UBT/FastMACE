"""Run a tool with the GEMM backend of ``FASTMACE_SCREEN_TC_GEMM`` installed (``cutlass_tc.install_from_env``).

    FASTMACE_SCREEN_TC_GEMM=cutlass-rn3 python run_with_backend.py TOOL.py ARGS...

As a launcher prefix it also reaches a tool's child processes, e.g. the gate's production child:
``force_stress_gate.py --production-command "python run_with_backend.py"`` (the child's argv is the command's words,
then the script, then its job).
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    import fastmace  # noqa: F401  (before torch, SPEC §14)

    sys.path.insert(0, str(HERE))
    import cutlass_tc

    value = cutlass_tc.install_from_env()
    print(f"run_with_backend: {cutlass_tc.ENV}={value}", file=sys.stderr, flush=True)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(Path(script).resolve().parent))
    runpy.run_path(script, run_name="__main__")
