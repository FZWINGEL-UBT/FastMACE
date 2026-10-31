# Audit of the `--a-constant` / `--b-constant` A/Bs (30 October 2026)

The segment_gemm tile-order A/B found a bug in `tools/ab_rewrites.py`. The tool set a side's module constants
only around that side's `planner.build` and `engine.prepare`; afterwards the module held the default again.
Code that read a constant later saw the default on both sides. That includes:

- graphs captured again in warm-up (new edge sizes);
- every graph of a run without `--prepare`;
- host code run on every neighbour-list rebuild.

A trace of `segment_gemm._gemm` (results/segment-gemm-order-20261030/order_trace.py) showed side a
capturing with TILE_ORDER "auto". The fix (d16d41c6, `ConstantsSide`) sets each side's constants before every
step. With it, NbCl5 at 9,216 atoms on the RTX 4000 Ada went from b/a 1.001 to 0.919.

`--prepare` came with CAP-T05 (6275833a). Before it, every graph was captured lazily in warm-up, outside the
constants block. The records cannot show warm-up re-captures and recompiles:

- `engine_counters_timed` and `compile_during_timing` cover only the timed blocks;
- the observables differ between the sides of every run at chaotic-MD level.

So each verdict rests on where the constant is read:

- **build:** in `planner.build`, a rewrite, a constructor or the build's first force call. Safe.
- **capture:** in a custom op's Python, every time a graph is captured. Not safe after the build.
- **host:** in host code, on every rebuild. Not safe after the build.

## The 28–29 October fp32-profile screens (no `--prepare`)

| Screen | Constant | Read | Verdict | Decision | Only this A/B? | Rerun |
|---|---|---|---|---|---|---|
| O7 | `fused_forward:BLOCKS_PER_SM` | build (`FusedFoldedForward.__init__`, in the rewrite) | valid | rejected | yes (local and Ada) | no |
| tiles128 | `tile_rule:TRITON_BLOCK_M=128`, `TRITON_NUM_WARPS=8` | capture (`segment_gemm_choice` in the `joined_products` op) | **invalid** | rejected as neutral (fp32-profile-20261028.md §5.2) | no | **yes** |
| aggrfusion | `torch._inductor.config:aggressive_fusion` | Inductor compile, in the build's first force call | valid (small risk) | rejected, neutral | yes (local) | no (conductor) |
| asyncrb | `skin:ASYNC_EARLY_BUILD` | build (the neighbour list's `__init__`) | valid | rejected, neutral | yes; GPU tests checked correctness only | no |
| tailhead | `species:TAIL_HEAD` | build (the readout fold) | valid | measurement; the prototype was not adopted | no: `tail_head_check.py` (exactness) | no |

- **tiles128.**
  - Side b never replayed 128 rows with 8 warps. Its b/a of 0.9993–1.0013 is the bug's signature.
  - The kernel bench (tools/bench/segment_gemm_tiles.py) contradicts the rejection: bit-identical and 3.8 % faster
    on sm_89, neutral on sm_120.
  - The rerun runs on main and on top of the tile order: rows order against rows + 128/8, and auto order against
    auto + 128/8. Ada first, then sm_120; the A100 only if one of them wins.
- **O7.** NaNbO3's b/a of 1.027–1.031 shows that side b ran 4 blocks per SM.
- **tailhead.** NbCl5's 0.991–0.996, in both build orders on two GPUs, shows that the constant was in force.

## Other runs

| Run | Constant | Read | `--prepare` | Verdict | Decision | Rerun |
|---|---|---|---|---|---|---|
| cap-nlist | `skin:LEAN_BUFFERS=False` | build | yes | valid | adopted | no |
| cap-outputs | `compile:UNUSED_OUTPUTS=()` | build (cache key, output filter) | yes | valid | adopted | no |
| cap-t05 | `graph:CAPTURE_SNAPSHOT_OFFLOAD_BYTES` | capture (snapshot only, never replay) | yes | valid for step time | offload adopted | no |
| cap-t06 | `langevin:NOISE_BATCH_BYTES` | build (the integrator's `__init__`) | yes | valid | adopted | no |
| cap-t07b | `radial:RECOMPUTE_KERNEL='chunked'` | build (`attach_recompute`) | no | valid (b/a 0.933–0.944) | adopted | no |
| cap-t08 region | `segments:SEGMENT_REGION='separate'` | build (`recompute_segments`) | yes | valid (b/a 1.24–1.25) | separate regions kept | no |
| setup-fixes | `forces:SPLIT_FORCE_FRAME`, `radial:PIECE_BUCKET` | build | yes | valid | F1–F5 adopted | no |
| setup-fixes | `compile:MARK_ATOMS_DYNAMIC=False` | host (`use_edges` on every rebuild) | yes | **at risk** | "step time unchanged" (setup-fixes.md) | **short recheck** |
| speed-small | `ab_constexpr_d:SIDE` | per step (the tool sets it itself) | yes | valid | constexpr D not adopted | no |

- **cap-t05.** Re-captures at new edge sizes used the default offload size. That changes only what the capture
  keeps aside, never the replayed kernels, so the step-time claim stands.
- **setup-fixes, MARK_ATOMS_DYNAMIC.**
  - After the build, side a's rebuilds marked the atom dimension dynamic again (the default). That probably
    recompiled it in warm-up, but unlogged.
  - The step-time A/B (b/a 0.995–1.005) is consistent with either the bug or no effect.
  - The setup-time numbers are not affected: `with_constants.py` set the constants for the whole process.
- **Not covered.** No other constant A/Bs exist in results/, plans/, docs/ or tools/ on main. Uncommitted runs
  under WSL `~/fastmace-runs` were not fully scanned.

## Reruns (conductor, 30 October)

1. **tiles128,** with the fixed tool, as described above.
2. **MARK_ATOMS_DYNAMIC=False,** a short step-time recheck: NaCl and NbCl5, both orders, one GPU.
3. **aggrfusion:** no rerun.

## Rerun results (31 October 2026)

### MARK_ATOMS_DYNAMIC: a real step-time cost on small cells

The switch was removed from main after the setup fixes (c97e9dc8; F5 is unconditional since). The recheck ran from
f246625b, the last commit with the switch, with only `tools/ab_rewrites.py` replaced by the fixed one (d16d41c6; the
only file that differs). Setup:

- local RTX 5060 Ti, `mace_cueq_oeq`, MH-1 hybrid float32 NVT 300 K;
- `--prepare`, 12 blocks of 300 steps;
- side a `fastmace.model.compile:MARK_ATOMS_DYNAMIC=False`, side b the adopted default (the atom dimension dynamic).

| Cell | Order ab: a / b ms, b/a | Order ba: a / b ms, b/a |
|---|---|---|
| NaCl 3×3×3, 216 atoms | 2.413 / 2.446, **1.014** | 2.454 / 2.496, **1.017** |
| NbCl5 3×3×3, 1,944 atoms | 18.050 / 18.041, 0.9995 | 18.230 / 18.311, 1.0045 |

- **The verdict changes.** F5's per-atom dynamic marking makes the NaCl step 1.4–1.7 % slower in both build orders.
  On NbCl5 the step is unchanged. The setup-fixes report's "step time unchanged" (b/a 0.995–1.005, NbCl5 only) had
  measured the default against itself.
- **What it bought** ([setup-fixes.md](setup-fixes.md), "F5's first-build cost"):
  - −8.6 s whenever a process meets an atom count new to the compile cache (27.3 against 18.7 s, NbCl5 3×3×4 on
    3×3×3's cache);
  - +2.0 s on every fresh process's first build.
- **Trade-off:** 1.5 % of a 2.4 ms step is 36 µs per step, so the 8.6 s saved on a new atom count are spent again
  after about 240,000 NaCl steps (4 minutes of MD at 216 atoms). On cells of about 2,000 atoms and more there is no
  step cost.
- **Decision:** the conductor puts the trade-off to the owner, and the code is unchanged.
- **Records:** results/ab-constants-audit-20261030/mark-atoms-dynamic/.

### tiles128 (128 rows and 8 warps against 64 and 4)

Same-process A/Bs with the fixed tool, MH-1 hybrid float32 NVT 300 K, side b `TRITON_BLOCK_M=128`,
`TRITON_NUM_WARPS=8`. Each side ran in one tile order: the old row-first order, or the new column order.
`results/segment-gemm-order-20261030/tiles128.sh`. Figures are b/a, build order ab / ba.

| Cell | Tile order | RTX 4000 Ada (leto) | RTX 5060 Ti (local) | A100-SXM4-40GB (Modal) |
|---|---|---|---|---|
| NbCl5 4×4×8, 9,216 atoms | rows | 0.999 / 0.996 | 1.001 / 1.002 | 1.021 / 1.021 |
| | columns | **0.987 / 0.986** | 0.996 / 0.994 | 1.021 / 1.021 |
| NbCl5 3×3×3, 1,944 atoms | rows | **0.990 / 0.989** | 1.017 / 1.004 | 1.000 / 1.000 |
| | columns | **0.989 / 0.989** | 1.020 / 1.007 | 1.000 / 1.000 |
| NaNbO3 3×3×3, 1,080 atoms | rows | **0.994 / 0.994** | 1.004 / 0.992 | 0.999 / 0.999 |
| | columns | **0.995 / 0.994** | 1.001 / 0.996 | 0.999 / 1.000 |

- **The 28 October verdict was the bug.** The screen called the tiles neutral, but on the RTX 4000 Ada they gain
  0.5–1.4 % on every cell in both build orders.
- **Elsewhere they do not.** The RTX 5060 Ti loses at NbCl5 1,944 atoms in both build orders. The A100 loses 2.1 %
  at 9,216 atoms.
- **The default (decision 66):** 128/8 is on for `NVIDIA RTX 4000 Ada Generation` only (`tile_rule.LARGE_TILE_GPUS`,
  8b074e61). Every other GPU keeps 64/4. The products are bitwise equal: the new GPU test
  `test_large_tiles_equal_the_default_tiles_bitwise` passed on the Ada.
- **The A100:** Modal served an A100-SXM4-40GB for this run. The tile-order A/Bs had run on the 80 GB card, which
  is the one `wprime_backward` defaults to.
- **Records:** results/ab-constants-audit-20261030/tiles128/.
