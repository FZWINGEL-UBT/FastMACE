# Memory at the limit, and the first padded edge size (ideas 4 and 5)

The owner's ideas 4 and 5 of 30 October 2026, through the conductor, after CAP-T09 ([capacity-nlist.md](capacity-nlist.md)):

- **Idea 4:** break down the peak reserved memory at the new limit, NbCl5 float32 at 51,840 atoms on the RTX 5060 Ti,
  by owner. Rank where the next GiB could come from. This is measurement only.
- **Idea 5:** size the first padded edge count from the candidates near the limit, to avoid the warm-up re-capture.
  The falsifier comes first.

## Verdict

**Idea 4: the graph pool holds 13.37 of the card's 15.93 GiB.**

- **Pool slack.** The pool's own allocations peak at 11.49 GiB live. The other **1.88 GiB is packing slack** inside
  one expandable segment. Nothing in it is deferred or held behind another stream.
- **Ranking.** That slack is the largest single item above what is live, ahead of the Langevin noise batch
  (0.59 GiB) and of everything outside the pool together.
- **Inside the live peak.** The per-atom activations dominate (CAP-T08): the `segment_gemm` joined products (3.16 GiB)
  and the symmetric contraction's inputs (1.98 GiB) are the two largest.

**Idea 5: falsified; no change.**

- **Where 52,920 atoms fails.** The run fails at the warm-up's force-graph capture, at the larger padded size: the
  first force graph was released first (`force_graphs="fit"`). The fresh capture still ran out of memory.
- **The test.** A first padded size large enough to need no re-capture (`edge_bucket_fraction=0.15`) fails at build
  instead, in the first force capture.
- **So the re-capture is not the cost.** The edge count grows about 8 % under thermal motion, and the largest padded
  size it reaches sets the peak either way.

## Idea 4: the ledger at 51,840 atoms

**Tool:** `tools/bench/memory_ledger.py`, new. It builds the default plan's `fm.Simulation` as
`capacity_scan.py point` does. It then runs 20 warm-up and 20 steady steps and records:

- **The allocator at the steady state.** It separates the graph pool's segments from the default pool. The default
  pool's live blocks are matched by address to their owners: the engine's registered buffers, the noise batch, the
  neighbour list and its screen buffer, and the model's parameters.
- **Memory outside the allocator** (`cudaMemGetInfo` minus reserved) at each phase. It is measured again after every
  graph is released and the cache emptied.
- **One more step that captures the stages again into a fresh pool**, with the allocator's history. That gives the
  live peak and the allocating sites, both overall and for the pool's own allocations.

**Cell:** NbCl5 6×6×20, MH-1 float32, default plan. 1.40M edges at the cutoff, padded to 1,367,040 and then
1,435,392. Device 15.93 GiB. Peak reserved 14.64 GiB in the steady phase, and 14.77 GiB at the warm-up's second force
capture.

| Rank | Where | GiB | What could come out of it |
| ---: | --- | ---: | --- |
| 1 | **Graph pool: packing slack** (reserved 13.37, the pool's own allocations peak at 11.49 live) | **1.88** | Some of it, by placing the few multi-GiB blocks so that freed holes are reused. Two `segment_gemm` products of 1.58 GiB each and two symmetric-contraction inputs of 0.99 GiB each make up most of the pool. They could share one fixed workspace, or be allocated largest first. Not tried. |
| 2 | Graph pool, live at its peak: per-atom activations of the force call | 11.49 | Idea 1, atom-chunked segments, is with the capacity worker. The largest sites are listed below. |
| 3 | Outside the allocator: CUDA context, libraries, kernels, and other processes on the GPU (the desktop) | 1.18 | Native Linux: CAP-T03 measured 0.3 GiB there for the same points. Nothing to do in FastMACE. |
| 4 | **Langevin noise batch** (`state.rng.batch`, 256 steps × 2 × N × 3 float64) | **0.59** | Idea 2, a smaller batch, is with the capacity worker. About 0.5 GiB. |
| 5 | Default pool: cached, not live | 0.22 | Mostly build-time leftovers; part of it is reusable by rebuilds. |
| 6 | Default pool: live but not a named buffer | 0.21 | Small pieces: model-copy temporaries (`deepcopy`, `convert`, 0.05 GiB), Inductor constants, eager-router buffers. |
| 7 | Neighbour list (buffers 115 MB, screen 49 MB) | 0.16 | Done in CAP-T09. |
| 8 | Outside the allocator: graph executables (the drop when every graph is released) | 0.11 | — |
| 9 | Model parameters and buffers | 0.07 | — |
| 10 | MD state, integrator and engine buffers | 0.03 | — |

The rows add up to the card: 13.37 + 1.18 + 0.59 + 0.22 + 0.21 + 0.16 + 0.11 + 0.07 + 0.03 = 15.94 GiB. Free memory was
0 at the steady state.

**Outside the allocator, by phase:**

| Phase | Reserved | Outside |
| --- | ---: | ---: |
| Start (context created) | 0 | 1.13 |
| After the build and the first captures | 14.46 | 1.47 |
| After the warm-up steps | 14.64 | 1.29 |
| Graphs released, cache emptied | 1.06 | 1.18 |

Windows' `nvidia-smi` showed 1.4–2.1 GiB in use before the process started, which includes the desktop.

**What fills the graph pool** (the re-capture's live peak, 11.55 GiB above 1.00 GiB live before, by allocating site):

| Site | GiB | kB per atom |
| --- | ---: | ---: |
| `fastmace/kernels/triton/segment_gemm.py` `joined_products_op` (2 blocks) | 3.16 | 65.5 |
| cuEquivariance `uniform_1d` (the symmetric contraction's inputs, 2 blocks) | 1.98 | 41.0 |
| `fastmace/kernels/oeq_streaming/fused_forward.py` `fused_folded_message` | 0.99 | 20.5 |
| Inductor buffers of the compiled model (about 20 blocks of 0.01–0.7 GiB) | 4.63 | 95.8 |
| `fastmace/kernels/triton/radial_table.py` `radial_table_forward` (the radial basis `a`) | 0.35 | 7.2 |
| other | 0.44 | 9.1 |

The live peak is about 240 kB per atom, almost all of it per atom, as CAP-T08 found. The only per-edge item above 0.1 GiB is the
radial basis, at 260 B per edge.

**Pool details:**

- one expandable segment of 13.36 GiB, plus one of 6 MiB;
- 579 allocation events in the re-capture, all on one stream;
- the live peak is the same whether a block counts until its free is requested or until it completes, so no free is
  deferred behind another stream.

The slack therefore comes from block placement in the segment: a hole left by a freed block is too small for a later,
larger block, so the segment grows at its end.

## Idea 5: the first padded edge size

**The mechanism:**

- The padded edge size is the cutoff edge count rounded up to a bucket of 5 % of the initial count.
- The initial count is the build's, of the thermalised start. At 300 K the count grows: at 51,840 atoms it went from
  1.36M to 1.40M edges within the 20 warm-up steps.
- So the warm-up captures a second force graph at the next bucket. With `force_graphs="fit"`, the first force graph is
  released first when both would not fit.

**Falsifier,** `tools/bench/capacity_scan.py point --trace` (FastMACE's log lines with times), NbCl5 5×7×21, 52,920
atoms:

| Run | Captures in the log | Failure |
| --- | --- | --- |
| Default, 5 % bucket | ACP; then B at 1,392,640 at build (first capture). At step 2: edges overflowed and the step was redone; one force graph released (fit) before capturing B at 1,462,272 | Out of memory in the warm-up, capturing B at 1,462,272 (`stage: capture`), after the release |
| `--neighbours edge_bucket_fraction=0.15`: the first padded size already covers the growth, so no re-capture | ACP | Out of memory at build, capturing the first B |

- **The failing capture was fresh.** In the default run the old force graph and its pool were gone when B at
  1,462,272 was captured. The failing capture was a fresh one, at the larger size, not two graphs at once.
- **Avoiding the re-capture moves the failure earlier,** from the warm-up to the build.
- **So a first padded size from the candidates cannot raise the limit:**
  - the peak belongs to the largest padded size the run reaches, whenever it is captured;
  - the cutoff + skin candidates would make the first size larger, not smaller, than the edges the run reaches.
- **No rule is added.**

**Smaller buckets, in passing.** A finer bucket would trim the padding above the edge count. That padding is at most
5 % of the edges, and the per-edge memory is now small: the radial basis is the largest per-edge item, at 260 B per
edge. That is about 20 MB at 1.4M edges, so it is not worth the extra captures. Not tried.

## What changed

- `tools/bench/memory_ledger.py` (new): the ledger above.
- `tools/bench/capacity_scan.py`:
  - `--neighbours FIELD=VALUE` (`NeighbourConfig` overrides), with the point's record;
  - decimal values parse as floats.
- `results/cap-ledger-20261030/`: the run script `local.sh`, the ledger record, and the two traced points.
- No library code changed.

## Checks

- Ruff is clean on both tools.
- **CPU tests:** `test_layering.py` and `test_ab_rewrites_tool.py`, commit `eeff1079`: **34 passed**. They ran on coraline
  (GPU hidden, `-n 2`), because leto had another user's job on its GPU.

## Setup

- **Machine:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130).
- **GPU access:** `tools/ci/local_gpu.py run` reservations, with the idle check before every job.
- **Source:** pinned copies of FastMACE `fastmace/cap-ledger`, based on `main` `fbd8a6ed`:
  - the ledger at `93f9bd21`;
  - the points at `b43757b4`.
- **Model:** MH-1 (`mh1-r2scan`), hybrid backend, float32, Langevin NVT at 300 K.
- **Raw records** in `results/cap-ledger-20261030/`:
  - `ledger/ledger-NbCl5-float32-51840.json`;
  - `points/point-NbCl5-float32-52920-trace.json` and `points/point-NbCl5-float32-52920b15-trace.json`;
  - `jobs.log`.
