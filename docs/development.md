# Development

## CPU development venv

CPU development and the install check use `~/environments/fastmace_cpu` in
WSL2 Ubuntu on the PC. It is FastMACE's own environment and may be rebuilt at
any time; the CPU CI follows the same recipe. It installs the main checkout
in editable mode, so it outlives any worktree and always sees the code that
is checked out there.

Keep pip caches in `~/pip-cache` and temporary files in `~/tmp`:

```bash
export PIP_CACHE_DIR=~/pip-cache TMPDIR=~/tmp
python3.10 -m venv ~/environments/fastmace_cpu
~/environments/fastmace_cpu/bin/python -m pip install --upgrade pip
~/environments/fastmace_cpu/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
~/environments/fastmace_cpu/bin/python -m pip install -e "/mnt/d/Margraf/FastMACE[dev]"
~/environments/fastmace_cpu/bin/python -m pip check
~/environments/fastmace_cpu/bin/python -m pip freeze > ~/environments/fastmace_cpu.requirements.txt
```

Torch comes from the PyTorch CPU index first, so the editable install keeps
it and does not pull a CUDA build from PyPI. The installed versions are
listed in `~/environments/fastmace_cpu.requirements.txt`.

Check that it works:

```bash
~/environments/fastmace_cpu/bin/python -c "import fastmace, torch; print(fastmace.__version__, torch.__version__)"
```

Run the CPU suite from the repository root (at most 8 workers locally, at
most 2 while other sessions run tests):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ~/environments/fastmace_cpu/bin/python -m pytest -n 8 tests/cpu
```

Never install FastMACE into the GPU environments; GPU runs use
`PYTHONPATH=<checkout>` with the environment's Python (AGENTS.md).
