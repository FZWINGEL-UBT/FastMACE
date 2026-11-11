"""e3nn MACE force check across torch versions (no FastMACE code; 11 November 2026).

Report to check: torch 2.13.x / 2.14.x give wrong forces with plain e3nn MACE above about 5k atoms.
Usage: python tools/bench/torch_force_check.py DTYPE 222,223,... > results.jsonl  (models from $FASTMACE_MODELS)

A rattled 512-atom NaCl cell is tiled k x k x k. By periodicity every copy must carry the small cell's forces, so
max |F_super - tile(F_small)| measures how wrong the large system's forces are, with no external reference.
Each case runs three times to catch run-to-run differences. Output: one JSON line per case.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from ase.build import bulk
from mace.calculators import MACECalculator

_M = os.environ.get("FASTMACE_MODELS", "/mnt/homes/zwingel/ResearchModule/Models")
MODELS = {"omat0-medium": f"{_M}/mace-omat-0-medium.model", "mp0b3-medium": f"{_M}/mace-mp-0b3-medium.model"}
dtype = sys.argv[1]  # float32 / float64
factors = [tuple(int(c) for c in x) for x in sys.argv[2].split(",")]  # e.g. 222,223

base = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((4, 4, 4))  # 512 atoms
rng = np.random.default_rng(0)
base.positions += rng.uniform(-0.08, 0.08, base.positions.shape)

meta = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), dtype=dtype)
for name, path in MODELS.items():
    calc = MACECalculator(model_paths=path, device="cuda", default_dtype=dtype)
    a = base.copy()
    calc.reset()
    a.calc = calc
    f_small = a.get_forces().copy()
    e_small = a.get_potential_energy() / len(a)
    for k in factors:
        sup = base.repeat(k)
        ref = np.tile(f_small, (int(np.prod(k)), 1))  # ASE repeat: copies outer, atoms inner
        for rep in range(3):
            sup = sup.copy()
            calc.reset()
            sup.calc = calc
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t = time.perf_counter()
                f = sup.get_forces().copy()
                e = sup.get_potential_energy() / len(sup)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc):
                    raise
                print(json.dumps(dict(meta, model=name, atoms=len(sup), rep=rep, oom=True)), flush=True)
                torch.cuda.empty_cache()
                break
            err = np.abs(f - ref)
            bad = int((np.linalg.norm(f - ref, axis=1) > 1e-2).sum())
            print(json.dumps(dict(meta, model=name, atoms=len(sup), rep=rep, s=round(dt, 2),
                                  max_abs_dF=float(err.max()), rms_dF=float(np.sqrt((err ** 2).mean())),
                                  atoms_dF_gt_10meV=bad, peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2), dE_meV_atom=1e3 * (e - e_small),
                                  max_abs_F=float(np.abs(ref).max()))), flush=True)
            sup.calc = None
            torch.cuda.empty_cache()
    del calc
    torch.cuda.empty_cache()
