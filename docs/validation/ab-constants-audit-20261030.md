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

The results go into this report when they are back.
