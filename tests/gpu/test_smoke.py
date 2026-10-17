"""GPU smoke test: CUDA is visible and a tiny model gives the CPU's float64 result on the GPU."""

import numpy as np
import torch

from _support.structures import nacl
from _support.tiny_models import tiny_mace_calculator


def test_cuda_is_available():
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1


def test_tiny_model_forward_on_gpu():
    atoms = nacl(rattle=0.05, seed=1)
    results = {}
    for device in ("cpu", "cuda"):
        calc = tiny_mace_calculator([11, 17], device=device)
        assert next(calc.models[0].parameters()).device.type == device
        copy = atoms.copy()
        copy.calc = calc
        results[device] = copy.get_potential_energy(), copy.get_forces()
    (e_cpu, f_cpu), (e_gpu, f_gpu) = results["cpu"], results["cuda"]
    assert np.isfinite(e_gpu) and np.isfinite(f_gpu).all()
    assert abs(e_gpu - e_cpu) < 1e-9
    np.testing.assert_allclose(f_gpu, f_cpu, atol=1e-10)
