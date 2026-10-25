import importlib.util
import sys

import mace.modules  # noqa: F401

for n in ("block_linear", "reassociate"):
    s = importlib.util.spec_from_file_location("mace.modules." + n, f"mace/modules/{n}.py")
    m = importlib.util.module_from_spec(s)
    sys.modules["mace.modules." + n] = m
    s.loader.exec_module(m)
import pytest  # noqa: E402

sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "tests/unit/test_block_linear.py",
                      "tests/unit/test_reassociate.py"]))
