# The near-limit memory probe (1 November 2026)

The conductor's decision (ii) after [estimate-band.md](estimate-band.md):

- probe only where rule 12's choice is ambiguous: the chosen stage's estimate is within the reserve's spread of
  the free memory, or the next-cheaper stage might fit;
- probe the chosen stage and, if there is room, the next-cheaper one, on a replica of 15,000–20,000 atoms with a
  warm compiler cache;
- choose from the measured reserved memory, and record the probe and the decision in the plan report;
- an `EngineConfig` option: `"auto"` (near the limit only, the default), `"off"` and `"always"`.

Gates: s0, s3 and s4 within 1.00–1.10 where the probe runs; the probe's cost with a warm cache; the leeloo ladder
re-run.

## Verdict

**Where the probe ran, rule 12's estimate is 1.02–1.07 times the measured peak.** Before the probe it was
1.01–1.24 on the same cells. The cells are eight default plans on leeloo, nothing forced, NbCl5 and SrTiO3 from
s0 to the region:

| Stage | Before the probe | With the probe |
|---|---:|---:|
| s0 (NbCl5 32,400) | 1.229 | 1.047 |
| s1 (SrTiO3 46,305) | 1.010 | 1.021 |
| s3 (NbCl5 72,576; SrTiO3 78,125) | 1.039; 1.041 | 1.049; 1.030 |
| s4 (NbCl5 85,536) | 1.240 | 1.047 |
| region (NbCl5 103,680, 121,824; SrTiO3 109,760) | 1.106, 1.108, 1.081 | 1.071, 1.074, 1.026 |

None ran out of memory or fell back. The probe changed no rung on these cells: rule 12's choices were right, and
its estimates were high.

**Cost, with a warm cache.**
- Each probed stage takes 38–43 s: a fresh child, the replica's build and 20 steps.
- The region stage takes 51–93 s. Its child builds the chunked region, with more compiled graphs.
- A plan probes one or two stages, 40–126 s in all, recorded as `setup: memory_probe`.
- The step time is unchanged: 305, 822, 1,015, 1,369, 1,618, 733, 1,384 and 2,134 ms, against 304–2,137 ms
  without the probe.

## How it works

`fastmace/planner/memory_probe.py`:

- **When** (`stages_to_probe`):
  - `"near-limit"` probes the chosen stage when its estimate × `PROBE_SPREAD` (1.25, the rescue matrix's highest
    stage band) reaches decision 51's limit;
  - it probes the next-cheaper stage (the chosen one without its last rung) when that stage's estimate ÷ 1.25 is
    within the limit;
  - `"always"` probes both, `"off"` neither;
  - `"auto"` resolves to `"near-limit"` for MD with CUDA graphs, else `"off"` (auto rule
    `near_limit_for_md_on_cuda`).
- **Rule 12's choice** (`Plan.memory_ladder`, `planner.memory.MemoryLadder`): rule 12 records the free memory, the
  chosen stage and the next-cheaper one. Each comes with the `RewriteConfig` values of its rescue records, its
  estimate and the estimate's size-independent part (weights, resident buffers, workspace).
- **The replica** (`replica`, `repeat_unit`):
  - the structure's own translational period along each cell vector, found from the positions (one atom's nearest
    same-species image, checked for the whole cell);
  - repeated to 15,000–20,000 atoms, the most isotropic choice;
  - the cell itself when it is no larger;
  - a structure without a periodic repeat unit (a thermalised start) is not probed, and the report says why.
- **The measurement** (`ChildProbe`, `SubprocessRunner.memory_probe`):
  - a fresh child per stage builds the run on the replica, with the stage's rescue records set and nothing else
    changed, and with the probe off;
  - it runs 20 MD steps and returns rule 12's estimate of the replica and its peak reserved and allocated memory.
- **The correction** (`memory.probed`):
  - the fixed part stays;
  - the rest is scaled by the replica's measured over its estimated peak, both less the fixed part, and kept
    `PROBE_HEADROOM` (1.05) above;
  - rule 12 re-plans with the probed stages corrected, and the other stages keep their estimates;
  - "short by far" divides a corrected estimate by the headroom instead of `RESCUE_OVERSTATEMENT`.
- **The record** (`Plan.memory_probe`, `MemoryProbeRecord`): every probed stage, with the replica, the measured and
  the corrected bytes, the cost, and the choice before and after. It appears:
  - in the plan text (`memory probe:`);
  - in the report (`memory:`);
  - in `metadata.json` (`plan.memory_probe`);
  - in the setup times.
- **Resume:** the scales go to `simulation.json` (`memory_probe_scales`), so a resume plans with them and never
  probes.
- **Never probed:** a relaxation, a plan that runs in a child process, a resume, or a probe's own child.
- **A failed child** ends the probing. The rules' plan stays, and the report warns.

**The headroom, offline** (`probe_offline.py`, `probe_offline.out`):
- **Method:** on the rescue matrix, each crystal, GPU and stage's smallest cell of at least 15,000 atoms played the
  replica for its larger cells.
- **Result:** corrected/measured came out at 0.95–1.05 for 59 of 60 cells.
- **Headroom:** hence 1.05, for 1.00–1.10.
- **The exception:** NbCl5 s4 at 43,200 atoms, 0.87, a build transient (a force graph's first warm-up outside the
  pool; estimate-band.md) that a replica does not see. With the headroom it is 0.91, still inside decision 51's
  10 % margin of the free memory.

## Records

`results/estimate-probe-20261101/`:
- `ladder.sh`, the GPU run, with `ladder-leeloo/run.log` and `ladder-leeloo/ladder.json`;
- `ladder.out`, `read_ladder.py`'s summary, from `read_ladder.py` (reads the points);
- `probe_offline.py`, `probe_offline.out` and `probe_offline.json`, the offline check;
- `debug_probe.py` and `debug_sim.py`, the trace that found the first run's bug;
- `ladder-leeloo/run-before-fix.log`, the first run.

The first run's probes never ran: the child read `options.engine`, which is `None` when the caller passes no
`EngineConfig`. It now takes the planner's engine; `tests/cpu/test_memory_probe.py` covers that path with a real
child.

Run details:
- leeloo, RTX 4000 Ada (20 GB), `mace_cueq_oeq`, hybrid, float32, MH-1;
- FastMACE `fastmace/estimate-probe` 5ff880c3 on `main` 8e7797be;
- `capacity_scan.py point`, 60 steps after 20;
- compiler cache warm from the earlier runs: `region-norm-leeloo/cache`.
