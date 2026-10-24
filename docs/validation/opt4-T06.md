# OPT4-T06: the OEq streaming forward on NbCl5 (Nsight Compute verdict)

Plan: `plans/opt-4.md` §3 candidate 6. The plan's question: the NbCl5 streaming forward ran at 50 %
occupancy and "about 47 % of DRAM" in OPT-T10's ncu table. Is there a lever a shape rule can set (edges per
block, warps per block), apart from the research engine's rejected register caps and receiver-owned layout?

**Verdict: no. Nothing built.** The kernel runs at **94 % of DRAM peak** (414 GB/s). Its 50 % occupancy
does not limit it: raising occupancy with a register cap makes it slower, and no block size changes it.
The most any change could gain is about 56 µs per force call, 0.26 % of the NbCl5 step. The "47 % of DRAM"
in the plan comes from ncu's clocks, not from the kernel (see "Why the plan saw 47 %").

## Run

- **Machine:** local PC, RTX 5060 Ti (sm_120, 36 SMs), driver 610.88, WSL2.
- **Environment:** `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130), Nsight Compute 2026.2.0.
- **Source:** FastMACE main `13a1983`, from a pinned export at `~/fastmace-opt4/src-13a1983`. No baseline
  was needed, because this task has no A/B.
- **Model and plan:** MH-1 `matpes_r2scan` (`mace-mh-1.model`, multi-head), hybrid float32, NbCl5 3×3×3,
  with every plan default (`tools/step_breakdown.py`'s `build_engine`).
- **GPU time:** one `tools/ci/local_gpu.py` reservation, 10:00–10:02 UTC, 24 October 2026. Untimed: not a
  quiet window. All variants ran in one process on the same inputs.
- **Method:** `results/opt4-T06/probe_forward.py`, driven by `results/opt4-T06/t06.sh`.
  1. One force call records the interaction-2 forward's real inputs.
  2. The probe rebuilds the kernel for each launch variant. Each variant is timed with CUDA events: 10
     warm-up launches, then the median of 7 blocks of 50.
  3. Each variant's output is compared bitwise with the default's.
  4. The same probe then runs once more under `ncu --set full --clock-control none --cache-control all`,
     with one launch per variant.
- **Records:** `results/opt4-T06/local/`: `probe.json`, `ncu_details.txt`, `ncu_raw.csv` and the logs. The
  `.ncu-rep` file stays at `~/fastmace-opt4/T06/`.

**Inputs** (real, current main):

| Tensor | Shape | Bytes |
| --- | --- | ---: |
| Node features `x` | `[1944, 512]` | 4.0 MB |
| Edge attributes `sh` | `[51084, 16]` | 3.3 MB |
| Radial weights `w` | `[51084, 1280]` | 261.6 MB |
| Output | `[1944, 5120]` | 39.8 MB |

- The layout is `ir_mul` with 128 channels.
- Receiver degrees are 24–30 (mean 26.3).
- The edge count is 51,084, the bound edge count. OPT-T10's ncu row showed the padded 56,320.

## Nsight Compute (default launch: 128 threads, 80 registers)

| Metric | Value |
| --- | --- |
| Duration (ncu) / CUDA events | 813 µs / 802 µs (SM clock 2.82 GHz) |
| **DRAM throughput** | **93.85 % of peak, 414 GB/s** (read 310.2 MB, write 26.5 MB) |
| Compute (SM) throughput | 16.2 %; FP32 at 8 % of peak; IPC 0.65; issue slots busy 14 % |
| Occupancy | theoretical 50 % (register limit: 6 blocks of 4 warps), achieved 49.2 %; 9.0 waves |
| Warp stalls (sampled) | **long scoreboard 92 %**; then wait 3 %, selected 2.5 %, dispatch 1 % |
| Scheduler | 5.9 active warps, 0.20 eligible per cycle |
| Caches | L1 hit 22 %, L2 hit 20 %; for L2 misses, 25.5 of each 32-byte sector are used |
| Spills | none |

**What limits it: DRAM bandwidth.** The warps wait on global loads (long scoreboard), and they wait because
DRAM is full, not because too few loads are in flight. The time is close to what the bytes cost:

- The minimum traffic is 308.7 MB: read `w`, `x` and `sh` once, write the output once. At the kernel's
  414 GB/s that takes **746 µs**.
- A pure read of `w` on the same buffer (`w.sum()`) runs at 404 GB/s (648 µs).
- The kernel moves 336.6 MB in 802 µs. About 41 MB of its reads are more than the minimum, most likely `x`
  and `sh` read a second time because the streamed `w` pushes them out of L2.
- **Ceiling: 802 − 746 ≈ 56 µs per force call, 0.26 % of NbCl5's 21.93 ms step** (`plans/opt-4.md` §1).
  The plan estimated ≤ 2 %.

## Launch sweep (same process, same inputs, CUDA events; every variant bitwise equal to the default)

| Variant | Registers | Blocks per SM | Theoretical occupancy | Waves | Median µs | vs default |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default, 128 threads | 80 | 6 | 50 % | 9.0 | 801.9 | – |
| 32 threads | 80 | 24 | 50 % | 9.0 | 802.6 | +0.1 % |
| 64 threads | 80 | 12 | 50 % | 9.0 | 800.8 | −0.1 % |
| 256 threads | 80 | 3 | 50 % | 9.0 | 801.6 | −0.0 % |
| 512 threads | 80 | 1 | 33 % | 13.5 | 830.7 | +3.6 % |
| `--maxrregcount=72` | 72 (8 B local) | 7 | 58 % | 7.7 | 806.1 | +0.5 % |
| `--maxrregcount=64` | 64 (104 B local) | 8 | 67 % | 6.75 | 818.4 | +2.1 % |

- **Block size cannot be the lever.** With 80 registers, a thread costs 2,560 registers per warp. The SM's
  65,536 registers therefore cap it at 25 warps whatever the block size, and every size from 32 to 256 gets
  the same 24 warps. 512 loses a third of them. All four sizes from 32 to 256 time within 0.2 %, which is
  noise.
- **Edges per block is fixed by the kernel's design.** One block covers whole receivers: 128 threads are
  one receiver's 128 channels, and they scan its CSR row. Changing it would mean another work partition in
  the generated kernel, which is not a launch shape. The research engine's receiver-owned and edge-owned
  rows already cover that space.
- **More occupancy does not help.** A register cap raises occupancy to 58 or 67 % and is slower, as the
  research engine found for the backward. DRAM is already saturated, so more warps in flight only queue.
- **Waves:** 9.0 on NbCl5, with no partial wave. The shared waves rule has nothing to correct.

## Why the plan saw 47 % of DRAM

OPT-T10's ncu run timed this kernel at 1,681 µs, while the profiler gave 0.87 ms. OPT-T09's ncu gave 822 µs.
ncu's DRAM percentage is bytes per duration, so a doubled duration halves it.

This run shows the same effect in one process. The default launch, first in the profiled range, took
813 µs under ncu, in line with the events. The six variants after it took 727–1,745 µs under ncu, while the
events gave 801–831 µs for all of them. With `--clock-control none`, ncu's replay passes run at whatever
clock the card drops to between passes. A single ncu duration on this card is not a rate. Only the first
kernel of a range, checked against events, should be read as one.

## Levers left, none a shape rule

- **L2 eviction-priority hints** (`evict_first` on the `w` loads, `evict_last` on `x`): these could recover
  part of the 41 MB re-read. That is a code change in the vendored template, not a launch shape, and it is
  bounded by the 56 µs (0.26 %) ceiling. Not built.
- The FP32 fusing note (ncu's "3.3 %" estimate) concerns compute, which is at 16 %, so it does not apply.

## Decision

Candidate 6 is **cut by its falsifying measurement** (`plans/opt-4.md` §3: "Stop if there is no lever a
shape rule can set"). Nothing was added to the library. The record is the row in `docs/rewrites.md`.
