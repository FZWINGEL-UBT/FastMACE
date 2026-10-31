# The message divided inside the chunked region (M3 target 1, 31 October 2026)

**Verdict: GO.**

- **The change.** With `recompute_node_region`, each interaction's message normalization (the message divided by
  `density × beta + alpha`) runs inside the region's atom chunks. The divided message is never kept for the backward.
- **Memory.** At NbCl5 21,600 and 43,200 atoms, the force call's peak drops by 32.6–32.7 kB per atom, to 0.75 times
  the region's before. The call takes 0.98 times as long.
- **Capacity.** On the 16 GB RTX 5060 Ti:
  - NbCl5 goes from 101,088 to 114,048 atoms (+13 %);
  - NaCl goes from 97,336 to 115,200 atoms (+18 %).
- **Planning.** Same record, no new rung. The memory term `_NODE_REGION_SHARE` goes from 0.11 to 0.18 of the
  stock per-node floats.
- **Exactness.** Rounding class: bitwise equal in eager on the CPU. The gate passed.

The target and its ceiling estimate are in [m3-ledger.md](m3-ledger.md), target 1.

## The change

- **`mh1.modules` interaction.** When its `gated_rows` chain is a joined region (`region_divides`), the interaction
  passes the raw message and the divisor (`density × beta + alpha`, one column) to the chain instead of the divided
  message. Otherwise it divides as before.
- **`segments.modules`.**
  - `joined_inputs` concatenates the divisor as a last input column.
  - `region` divides the message by it first.
  - `region_forward` splits the divisor per atom chunk with the rest.
  - `join_region` marks the chain `region_divides`.
- **Self-check.** `_region_cases` probes the region with a divisor of `1 + |probe|`. It compares against the stored
  chain applied to the divided message, with the divisor among the gradient leaves.
- **Memory term.** `_NODE_REGION_SHARE = 0.18` of the stock per-node floats (57.3 kB per atom of MH-1's 317,440 B).
  It is fitted on the peaks below: separate segments 3.126 GiB against the region's 1.974 at NbCl5 21,600 atoms,
  gate_epilogue off.

## Falsifier: the force call's peak

`results/chunked-region-20261031/region_probe.py`, as in [chunked-region.md](chunked-region.md): one eager call of
the compiled force call, the allocator's peak above the call's start, and the median of 5 calls. Combined region in
4 chunks; leto, RTX 4000 Ada, NAS `mace_cueq_oeq`. Main 4d4b2121 (message divided outside the region) against the
branch 36ced3e2 (divided inside). GO criterion: at least 20 kB per atom lower, at most 1.05 times the call.

| NbCl5 atoms | Main: peak, call | Branch: peak, call | Peak saved | Call |
|---|---|---|---|---|
| 21,600 | 2.642 GiB, 294.2 ms | 1.985 GiB, 288.8 ms | 32.6 kB per atom | 0.98× |
| 43,200 | 5.283 GiB, 579.0 ms | 3.969 GiB, 567.6 ms | 32.7 kB per atom | 0.98× |

The ledger's estimate was 28.7 kB per atom (interaction 1's divided message, 20.5, and interaction 0's, 8.2). The
measured saving is slightly larger.

## gate_epilogue and the separate segments

Main's region peak at 21,600 atoms was 2.642 GiB, against 2.417 in [chunked-region.md](chunked-region.md) the
morning before. The separate segments' peak went from 3.126 to 3.857 GiB. The cause is `gate_epilogue`, which
decision 66 turned on by default for MH-1 on the RTX 5060 Ti and RTX 4000 Ada (speed-def E2).

The same probe was run on the branch (rebased onto main 3748766c, d2c20b03), NbCl5 21,600 atoms, with gate_epilogue
on and off:

| Layout | GPU | gate_epilogue on | off | Difference |
|---|---|---:|---:|---:|
| Separate segments | leeloo, RTX 4000 Ada | 3.857 GiB | 3.126 GiB | 36.3 kB per atom |
| Separate segments | trillian, RTX A4000 (`wprime_backward` forced on) | 3.857 GiB | 3.126 GiB | 36.3 kB per atom |
| Combined region, 4 chunks | leeloo | 1.985 GiB | 1.974 GiB | 0.5 kB per atom |
| Combined region, 4 chunks | trillian | 1.985 GiB | 1.974 GiB | 0.5 kB per atom |

- **With the separate segments,** gate_epilogue keeps 36.3 kB per atom more at the force call's peak.
- **In the chunked region,** its extra buffer is rebuilt per chunk, and the difference vanishes.
- **Consequence.** The region's memory term holds with gate_epilogue on or off.

### gate_epilogue as a memory-rescue rung (fastmace/ge-rescue)

Without the node segments (trillian, `recompute_node_segments` off, `recompute_conv_weights` on, NbCl5 21,600
atoms), gate_epilogue keeps less:

| Records | gate_epilogue on | off | Difference |
|---|---:|---:|---:|
| conv_weights | 4.786 GiB | 4.539 GiB | 12.3 kB per atom |
| conv_weights, separate segments | 3.857 GiB | 3.126 GiB | 36.3 kB per atom |
| conv_weights, segments, chunked region | 1.985 GiB | 1.974 GiB | 0.5 kB per atom |

Without `recompute_conv_weights`, 21,600 atoms do not fit the A4000's 16 GB.

At the limit (trillian, NbCl5 114,048 atoms, every region record on, 200 steps), the steady peak reserved is
12.52 GiB with gate_epilogue on and 11.97 GiB with it off. That is 5 kB per atom of reserved memory.

**The rung's place.** Each rung frees, per percent of step time:

| Rung | Frees | Step time | Per percent |
|---|---|---|---|
| `recompute_conv_weights` | about 135 kB per atom (5,120 B per edge) | 1.01–1.04× | 35–135 kB |
| `recompute_first_table` | about 7 kB per atom (260 B per edge) | 1.02–1.04× | 2–4 kB |
| `recompute_node_segments` | 64 kB per atom | 1.095× | 7 kB |
| gate_epilogue off, without the segments | 12 kB per atom | 1.00–1.05× | from 2.5 kB |
| gate_epilogue off, with the segments | 36 kB per atom | 1.00–1.05× | from 7 kB |
| `recompute_node_region` | 93 kB per atom (with gate_epilogue on) | about 1.2× | about 5 kB |

- gate_epilogue's speed gain is speed-def E2's MD step: NbCl5 and NaNbO3 1.2–4.7 %, NaCl none.
- So gate_epilogue off pays most once the separate segments are on. It goes after `recompute_node_segments` and
  before `recompute_node_region`:
  `MEMORY_RESCUE = (recompute_conv_weights, recompute_first_table, recompute_node_segments, gate_epilogue,
  recompute_node_region)`. It is the one rung the rescue turns off (`planner.memory.RESCUE_OFF`).
- **Its term** is 0.039 of the stock per-node floats (12.3 kB per atom), 0.114 with the separate segments
  (36.3 kB), and none with the chunked region. `planner.memory.CONTEXT_TERMS` picks it from the enabled records.
- **Not done:** once `recompute_node_region` is on, gate_epilogue costs memory no longer and could be turned back on
  for its speed. The rescue only adds rungs, so it stays off.

## Correctness

- **CPU:** `tests/cpu/test_recompute_node_segments.py` covers the dividing region as before:
  - the region's self-check;
  - energy and forces bitwise equal to the record off, in eager, for 1 and 3 chunks.
- **The targeted CPU files:** layering, segments, rewrites, memory, planner, config, docs options and report, on
  leeloo's CPUs (GPU hidden), 1f77378a: 511 passed, 32 skipped.
- **Gate:** `tools/force_stress_gate.py` (leeloo, RTX 4000 Ada) with `recompute_conv_weights`,
  `recompute_node_segments` and `recompute_node_region` on, branch d2c20b03: passed.

| Structure | Energy ratio | Forces ratio | Stress ratio |
|---|---|---|---|
| NaCl | 0.693 | 0.939 | 0.983 |
| NbCl5 | 0.113 | 1.023 | 0.380 |
| NaNbO3 | 0.283 | 0.989 | 0.541 |

## Capacity: limit points on the RTX 5060 Ti (16 GB)

Setup:
- `tools/bench/capacity_scan.py point`, default plan, 200 steps.
- `results/chunked-region-20261031/points.sh` with REGION=on: `recompute_conv_weights`, `recompute_node_segments`
  and `recompute_node_region` on.
- Local WSL2, `mace_cueq_oeq`, branch 36ced3e2. gate_epilogue is on by default on this GPU.
- Every passing point satisfies the capacity check:
  - at step 0, exactly the reference edge count;
  - after the steps, the live count equal to a fresh ASE rebuild;
  - nonzero forces.

| Cell | Atoms | Status | ms/step | Peak reserved: build, warm-up, steady (GiB) |
|---|---:|---|---:|---|
| NbCl5 6×6×41 | 106,272 | ok | 1,336.7 | 12.89, 13.11, 13.14 |
| NbCl5 6×6×44 | 114,048 | ok | 1,482.3 | 13.83, 14.02, 14.08 |
| NbCl5 6×6×47 | 121,824 | out of memory at build | | |
| NbCl5 6×6×50 | 129,600 | out of memory at build | | |
| NaCl 23×23×24 | 101,568 | ok | 1,299.6 | 12.28, 12.51, 12.08 |
| NaCl 24×24×24 | 110,592 | ok | 1,459.9 | 13.32, 13.58, 13.58 |
| NaCl 24×24×25 | 115,200 | ok | 1,505.6 | 13.82, 14.14, 13.63 |
| NaCl 25×25×25 | 125,000 | out of memory at build | | |

- **NbCl5:** 101,088 atoms before, 114,048 now (+13 %). The ledger's ceiling was 120,000–129,000. The step time
  per atom stays at 13.0 µs.
- **NaCl:** 97,336 atoms before, at least 115,200 now (+18 %). 120,000 atoms was not tried.
- **The A4000 too** (trillian, 16 GB, `wprime_backward` forced on): NbCl5 114,048 atoms passes with gate_epilogue
  on, 12.52 GiB steady peak reserved.

## Fix 2: the second B graph is not a cost

The ledger ([m3-ledger.md](m3-ledger.md)) listed 1.34 GiB of graph executables outside the allocator on NaCl against
0.31 on NbCl5 and named two kept B graphs as the cause. Measured on native Linux, that is not a cost:

- **The ledger's own records** show one B graph plus ACP at steady state in both structures. The second B capture
  (the next edge bucket) replaced the first (`force_graph_releases = 1`). NbCl5's 0.31 GiB is clipped: its free
  memory was 0 when the ledger read it, so the difference has no meaning there.
- **The per-key probe** (`exec_probe`, leeloo, RTX 4000 Ada, native Linux): NaCl at 21,952 and 46,656 atoms, NbCl5
  at 21,600 and 43,200.
  - These smaller cells keep two B graphs, one per edge bucket, plus ACP.
  - Dropping each graph in turn frees nothing outside the allocator: 0.30 GiB before and after every drop.
  - Both B graphs share the graph pool, so the second one adds no pool memory either (the ledger's capture records
    show no growth at its capture).
- **The WSL2 figure.** The 1.25 GiB that NaCl showed on the local RTX 5060 Ti has no counterpart on native Linux.
  The probe was not run under WSL2, so what the WSL driver holds there is not split by key. It is not a second
  B graph at the limit, where only one is kept.
- **Decision (conductor):** recorded as a ledger artefact; no fix.

## Records

`results/region-norm-20261031/`:
- `falsifier-leto/`: main and branch region_probe records;
- `leeloo/`: the gate summaries and the gate_epilogue on/off probe records;
- `points-5060ti/`: the limit points' JSON and log;
- `exec-probe-leeloo/`: the per-key graph-executable probe;
- `trillian/`: the A4000's gate_epilogue probes (with and without the segments) and its limit points.
