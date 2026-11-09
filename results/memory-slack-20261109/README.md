# Reserved but unused memory at the Ada limit: the neighbour-list rebuild and the graph pool (9 November 2026)

The conductor's item 1 after the memory profile ([memory-profile-20261109](../memory-profile-20261109/README.md)):
find what keeps reserved memory that holds no live tensor, and free or size it; bitwise forces and a not-slower MD
A/B on the RTX 4000 Ada and the RTX 5060 Ti; then the limits again. Also the graph pool's packing slack: can the
capture order or the pool be arranged so that freed blocks are reused?

Source: branch `fastmace/memory-profile-ada` at `d4d7a08a` (main `8cb960ae` + this change). GPUs: leto's NVIDIA RTX
4000 Ada Generation (20 GB, NAS `mace_cueq_oeq`, idle check before every job) and the local RTX 5060 Ti (WSL2, inside
a `tools/ci/local_gpu.py` reservation). OMAT-0 medium, float32, hybrid backend, default plan.

## Summary

- **The neighbour list's rebuild sorts its keys alone now (`skin.SORT_KEYS_ONLY`, default True).** `build_gpu` wrote
  an int64 order key per screened pair into a new tensor, sorted it with its indices (`torch.sort`: sorted values,
  an index tensor and CUB's key-value buffers), and gathered the screen codes through the indices. The key is unique
  per pair and holds the receiver, the sender (group and length bin where sorted) and the image, so the rebuild now
  writes each key over its pair's screen code (the screen's buffer is scratch until the next screen), sorts the keys
  alone (`torch.unique(sorted=True)`, CUB's keys-only sort) and decodes the pairs from the sorted keys. The same
  pairs in the same order.
  - **The rebuild's peak above the list's buffers: 1,019 → 351 MiB** at SrTiO3 233,280 atoms (21.6 M candidates),
    the same build time (2.33–2.36 s either way), every buffer of the list and of `compact()`'s edge list bitwise
    equal ([n1/nl_sort-36.json](n1/nl_sort-36.json)).
  - **MD at SrTiO3 36³: peak reserved 16.49 → 15.82 GiB (75.9 → 72.8 kB/atom)**; the default pool's live peak 3.37 →
    2.75 GiB. The steady phase is no longer the peak (15.80 GiB); the build is ([n2/](n2/) against [s1/](s1/)).
  - **Ada SrTiO3 limit 253,265 → 274,360 atoms** (38³, 1,746 ms/step, steady peak reserved 18.68 GiB). 39³
    (296,595) fails in the build's first force call on the dense `[atoms, 5120]` message gradient of rescue rung
    (B), as NaCl at 36³ does: rung (B) now sets both Ada limits (item 2).
  - **Speed: ties.** Benchmark_loup (SrTiO3 10×10×10), b = keys only: Ada b/a 1.0004 / 0.9999, 5060 Ti 1.0027 /
    0.9986 (ab / ba).
- **The graph pool's packing slack (13.8 kB/atom at 36³) is placement loss of the caching allocator during the
  capture; no setting we tried reduces it.** The pool is one expandable segment; at its live peak (9.99 GiB, in the
  force backward) the segment spans 11.27 GiB with six holes (1.28 GiB, the largest 797 MiB, left by the spline
  forward's message blocks), and after the peak an 1,822 MiB request of the backward finds 5.0 GiB free but no hole
  that large, so the segment grows by 1.74 GiB to 12.97. Every hole is written during the replay, so nothing can be
  released.
  - `compose_skip.JOINED_LAYOUT` (the species products written in place, no `torch.cat` of the blocks): the same
    peaks to the byte ([s2/](s2/)). No change.
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`: 36³ runs out of memory in the build (19.17 GiB reserved;
    57 GiB of reserve growth while the request fitted in the free bytes) ([n3/](n3/)). REJECTED.
  - What would reduce it: fewer or smaller transients between the saved forward tensors (each removed one also removes
    its hole), or the recompute rung of item 2, which drops saved tensors. A placement-aware arena for the capture
    (a pluggable allocator replaying planned offsets) could reach the live peak but is not worth its risk here.

## Measurements

| Case (SrTiO3 36³, MD, 5 warm-up + 5 steps) | Peak reserved | kB/atom | Default pool live peak | Graph pool reserved / live |
|---|---:|---:|---:|---:|
| Defaults (`d4d7a08a` with `SORT_KEYS_ONLY=False`; equals r3 of the profile) | 16.49 GiB | 75.9 | 3.37 GiB | 12.97 / 9.99 GiB |
| `JOINED_LAYOUT` on | 16.49 GiB | 75.9 | 3.37 GiB | 12.97 / 9.99 GiB |
| **`SORT_KEYS_ONLY` (new default)** | **15.82 GiB** | **72.8** | **2.75 GiB** | 12.97 / 9.99 GiB |
| `expandable_segments:False` | out of memory (19.17 GiB in the build) | | | |

Capacity (`capacity_scan.py point`, 20 warm-up and 200 timed steps, alone on the GPU):

| GPU | Crystal | Before | With `SORT_KEYS_ONLY` | First failing size |
|---|---|---:|---:|---|
| RTX 4000 Ada (leto) | SrTiO3 | 253,265 | **274,360** (38³, 1,746 ms, 18.68 GiB) | 296,595: build, rung (B)'s dense `[N, 5120]` gradient (5.66 GiB) |
| RTX 4000 Ada (leto) | NaCl | 343,000 | 343,000 (not re-run: its failure is rung (B)'s buffer, not the rebuild) | 373,248 |
| RTX 5060 Ti (local) | SrTiO3 | 163,840 | 163,840 (unchanged) | 179,685: build, Triton module load out of memory |
| RTX 5060 Ti (local) | NaCl | 195,112 | 195,112 (unchanged) | 216,000: build, a driver allocation out of memory |

On the 5060 Ti the first failing sizes fail in the build on allocations outside PyTorch's allocator (Triton's module
loads, the driver's own), with WSL2 between: host or driver memory pressure, not tensors (noted; lower priority).

## Gates

- **Bitwise**: `tests/cpu/test_neighbors.py::test_keys_only_sort_builds_the_same_list` (8 cases: receivers sorted or
  not, sender groups or not, length bins or not; ragged chunks) compares every buffer of both builds and of
  `compact()`; the GPU check at 36³ compares them on the Ada. A neighbour list's buffers equal bit for bit give the
  same forces.
- **Speed**: the A/Bs above ([ab3/](ab3/); 5060 Ti in [5060ti/](5060ti/)).
- **CPU tests** (leto): `test_neighbors.py`, `test_neighbors_lean.py`, `test_rule_11.py`, `test_layering.py` pass.

## Jobs

[jobs/](jobs/): `nl_sort.py` (the rebuild check), `abnl.sh` / `abnl_local.sh` (A/Bs), `leto.sh` + `profile.sh`
(profiles; `EXTRA`, `TAG`), `points.sh`. `tools/bench/memory_profile.py --constant MODULE:NAME=VALUE` sets module
constants before the build.
