"""2-T19: run setup_stages.py with MACE imported before fastmace (1b-T23 finding 8). Same arguments."""

import runpy
import sys
from pathlib import Path

import mace.calculators  # noqa: F401  (first, on purpose)

sys.argv = [str(Path(__file__).with_name("setup_stages.py"))] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
