"""Run one child body of a FastMACE GPU test file (its PRELUDE plus the named string) in this interpreter.

    python run_child.py tests/gpu/test_kernels_oeq_streaming_cuda.py FOLDED_CHUNKED

The streaming-kernel tests run their bodies in a child of the test's interpreter; locally the hybrid
environment has OpenEquivariance but no pytest, so this runs the same body directly.
"""

import ast
import sys
import textwrap

path, name = sys.argv[1], sys.argv[2]
tree = ast.parse(open(path, encoding="utf-8").read())
bodies = {node.targets[0].id: node.value.value for node in tree.body
          if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)}
source = textwrap.dedent(bodies["PRELUDE"]) + textwrap.dedent(bodies[name])
exec(compile(source, f"{path}:{name}", "exec"), {"__name__": "__main__"})
