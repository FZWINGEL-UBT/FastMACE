#!/usr/bin/env bash
set -eu
cd /mnt/d/Margraf/fm-dmqe-ideas
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
py="$HOME/environments/fastmace_cpu/bin/python"
"$py" -m pytest -q tests/cpu/test_dmqe_species_lab.py
"$py" -m ruff check tools/dmqe_outer_lab.py tools/dmqe_species_lab.py tools/dmqe_species_force_lab.py \
 tools/dmqe_geometry_lab.py tools/local/dmqe_workstation_guard.py tests/cpu/test_dmqe_species_lab.py
