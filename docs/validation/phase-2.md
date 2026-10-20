# Phase 2 acceptance report

**Status: runs in progress, 20 October 2026.** Plan task 2-T23, branch `fastmace/2-T23`.

<!-- VERDICT -->

The library in every 2-T23 run is main `85360cc`: each run imports a `git archive` of a 2-T23 tools
commit whose `fastmace/` tree equals `85360cc`'s (`d885b12`, `37c4014`; only `tools/`, `tests/` and
`docs/` differ). Evidence from the merged phase 2 tasks is reused where the plan allows (conductor's
go message); 2-T23 reran what needs final main: item 1's GPU check, the acceptance and full GPU
tiers, and item 8.

## Verdict

<!-- TABLE -->

## Runs

<!-- RUNS -->

## Item 8 on murphy, NaCl: the rule for three more processes (written before they ran)

The first three murphy A/B processes (run `2-T23-murphy-20261020`, 15:42–16:05 UTC) gave main/1b on NaCl
1.00096, 1.00164 and 1.00042 against the 0.16 % floor. The 1.00164 process (ABAB, 1b built first) is
0.004 percentage points over the floor, with every pair above 1. This is the same process dependence
that 1b's open NaCl item showed ([phase-1b.md](phase-1b.md), finding 3). On 20 October 2026 the conductor
fixed this rule before more data was taken:

- three more NaCl-only A/B processes on murphy, timed and quiet, right after the acceptance tier and
  before item 4's repeat: ABAB with 1b built first, ABAB with main built first, ABBA with 1b built first
  (suite `regression2-nacl`);
- **murphy NaCl passes** if the median main/1b ratio over all six processes is ≤ 1.0016 (the floor), no
  single process exceeds 1.0032 (twice the floor), and the one-version controls show main not slower;
- otherwise it is reported as failing for the owner, with every number.

## Item 4, hybrid: the repeat's protocol (written before it ran)

**Why there is a repeat.** The first comparison (run `2-T23-murphy-20261020-runner2`, tools `4b35b60`)
followed 1b-T14's measure: MH-1 NaCl 3×3×3, hybrid float32, Langevin 2 fs, target ramp 300 → 600 K over
20,000 steps in two `run()` calls, two seeds, from the 0 K crystal. It failed narrowly on the mean
potential energy (0.73 and 0.92 meV/atom against a limit of 0.65), and the whole difference came from
the heating transient of the first ~1,000 steps (section [Item 4](#4-subprocess-runner)). On 20 October
2026, after that result, **the owner decided: "Repeat, equilibrated"**. Item 4's hybrid clause is
judged on the repeat; the first result stays in this report as it was.

**Protocol of the repeat** (`tools/runner_statistics.py`, suite `runner-repeat`, murphy, after the timed
window):

1. **Start state, made once** (`runner_statistics.py equilibrate`): MH-1 `matpes_r2scan`, NaCl 3×3×3
   (216 atoms) from the 0 K crystal, one in-process hybrid float32 `Simulation`, Langevin at a constant
   300 K target, friction 0.02 × `ase.units.fs` per fs (as the first comparison and 2-T07):
   - 4,000 steps at 0.5 fs (2 ps; the production step, which has no transient);
   - then 5,000 steps at 2 fs (10 ps).
   The positions, momenta and cell of the last step are written to `start.extxyz`. The mean temperature
   and potential energy over the second half of each stage are recorded, to show the state is
   equilibrated at 300 K.
2. **Runs**: both cases start from `start.extxyz`, with its momenta, and are otherwise the first
   comparison's protocol: 2 fs, the target ramp 300 → 600 K over 20,000 steps in two `run()` calls of
   10,000 (one hand-off), thermo rows every 20 steps.
   - `fresh`: the hybrid plan in this process;
   - `runner`: a plain-cuEq `Simulation` runs first (its compile fixes the process's family), then the
     hybrid plan (`backend="hybrid"`) runs in subprocess-runner children.
   - **Four seeds**: 20261001, 20261101, 20261201, 20261301; each case runs each seed.
3. **Limit**, 1b-T14's rule extended to four seeds. Per seed, for the mean temperature, the mean
   potential energy and the rms over 50 K target bins of the bin-mean potential energy:
   |runner − fresh| ≤ max(2 × D, floor), where D is the median of that quantity's difference over the six
   pairs of fresh seeds (with two seeds, D is their one difference: 1b-T14's rule), and the floors are
   1b-T14's: 1 K, 0.05 meV/atom, 0.05 meV/atom. Every row from step 20 to 20,000 counts; nothing is
   excluded.
4. **Pass**: all four seeds within the limit for all three quantities, every runner run reports
   `plan.runner == "subprocess"` and every fresh run `"in-process"`, and every run's backend is hybrid.

## Reproducing

From a FastMACE checkout (`$FM`) with the tag `fastmace-1b`:

```bash
python -m fastmace.bench.versions export --repository $FM --ref fastmace-1b --name fastmace_1b --destination DIR
python3 $FM/tools/ci/local_gpu.py run --purpose "phase 2 acceptance" -- \
  python3 $FM/tools/acceptance/run.py regression2 acceptance-tier trajectory2 runner-statistics full-tier \
  --machine local --run-id <id> --fastmace <pinned copy> --versions DIR
python3 $FM/tools/acceptance/summarize2.py ~/fastmace-acceptance/<id> --machine local
```

On murphy the same `run.py` line runs detached (`tools/acceptance/launch.sh`), with nothing on the GPU.
