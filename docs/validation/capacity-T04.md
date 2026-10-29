# CAP-T04: fragmentation at the peak

Plan task CAP-T04 ([plans/capacity.md](../../plans/capacity.md)), owner go-ahead on 29 October 2026, measured on one
structure (NbCl5) as the owner asked. CAP-T02 ([capacity-T02.md](capacity-T02.md)) left the NbCl5 float32 limit at
30,600 atoms on the RTX 5060 Ti. There, PyTorch's peak reserved memory exceeded its peak allocated memory by
1.2 GiB.

## Verdict

**There is no allocator fragmentation to recover.**

- PyTorch's allocator reports 0 inactive split bytes in every phase. With expandable segments (FastMACE's default)
  a block is not split into unusable pieces.
- The gap between peak reserved and peak allocated (1.19 GiB at 32,400 atoms) is cached, unused memory. The two
  peaks fall at different moments: the first force capture's eager warm-up runs outside the graph pool, then the
  capture runs inside it.
- None of the engine's capture options changes the memory. At NbCl5, 32,400 atoms, these settings give
  identical peaks (12.58 GiB allocated, 13.77 reserved in the build, 14.02 in the warm-up):
  - `warmup_in_graph_pool=false`;
  - `empty_cache_before_capture=false`;
  - `force_graphs=one`.
- Nothing is changed in the library.

**The NbCl5 float32 limit is 32,400 atoms (5×5×18, about 0.85M edges), 308–313 ms per step.** The capacity scan's
isotropic sizes jump from 30,600 to 36,720 atoms and had skipped this size. At 34,200 atoms (5×5×19) the run
fails at step 211:

- The heated system's edge count outgrew the captured padded size.
- `fit` released the force graph, and the larger one needed more memory than was free.
- One force evaluation at that edge count does not fit, so this is a real limit.

**What remains, in order of size:**

- the activations themselves (CAP-T07, CAP-T08): 5.9 kB per edge and 268 kB per atom in float32;
- under WSL2, 0.3–1.0 GiB of stack reserved for spilling kernels (Inductor's fused kernels up to 5.3 KB per
  thread; [capacity-T03.md](capacity-T03.md));
- the registry snapshot around each capture (CAP-T05, about 0.5 GiB);
- the Langevin noise batch (CAP-T06, about 0.3 GiB).

## Measurements

NbCl5 float32, MH-1, default plan (decision 67), RTX 5060 Ti under WSL2. Columns are GiB: peak allocated / peak
reserved / outside the allocator, per phase.

| Point | Build | Warm-up | Steady | ms per step | Result |
| --- | --- | --- | --- | --- | --- |
| 32,400 atoms, default | 12.58 / 13.77 / 2.27 | 12.80 / 14.02 / 2.13 | 0.99 / 13.80 / 2.13 | 313 | passes |
| 32,400, `warmup_in_graph_pool=false` | 12.58 / 13.77 / 2.26 | 12.80 / 14.02 / 2.13 | 0.99 / 13.80 / 2.13 | 313 | passes |
| 32,400, `empty_cache_before_capture=false` | 12.58 / 13.77 / 2.16 | 12.80 / 14.02 / 2.13 | 0.99 / 13.80 / 2.13 | 308 | passes |
| 32,400, `force_graphs=one` | 12.58 / 13.77 / 2.13 | 12.80 / 14.02 / 2.13 | 0.99 / 13.80 / 2.13 | 311 | passes |
| 34,200 atoms, default | 13.26 / 14.53 / 1.88 | 13.49 / 14.79 / 1.40 | | | fails at step 211 (force graph for a larger edge count) |

- Allocator statistics (`torch.cuda.memory_stats`): the peaks of `inactive_split_bytes` are 0 in every phase, and
  there were no allocation retries.
- In the steady phase at 34,200 atoms the card is full: 0 free, 14.53 GiB reserved and 1.40 GiB outside the
  allocator. That is more than its 15.93 GiB, so the WSL2 driver overcommits there.

## Setup

- **Machine:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130),
  `tools/ci/local_gpu.py run` reservations with the idle check.
- **FastMACE:** `4dc601be` (the options) and `fc605006` (allocator statistics added to
  `tools/bench/capacity_scan.py`); the library code is the same in both.
- **Run:** 20 warm-up and 200 timed steps of Langevin NVT, one process per point.
- **Raw data:** `results/cap-t04-20261029/points/`. The suffixes are:
  - `-base`: default plan;
  - `-nopoolwarm`, `-noempty`, `-one`: the three options above;
  - `-stats`: with allocator statistics;
  - `-trace`: with the memory trace at 34,200 atoms.
