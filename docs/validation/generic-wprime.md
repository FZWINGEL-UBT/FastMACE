# Generic fused forward and wprime_backward for OMAT-0 (3 November 2026)

Owner request: "Yes, build it", after the opt-9 results (MH-1's `wprime_backward` 11–19 % on the A100) raised the
question whether the generic folded path has it. Owner decision the same day, after these results: **default on for
OMAT-0 medium on the RTX 5060 Ti, the RTX 4000 Ada and the A100-SXM4-80GB** (decision 66,
`planner.fixed.DECISION_66_GENERIC_WPRIME_GPUS`, `DECISION_66_GENERIC_WPRIME_LABELS`).

**Verdict:**
- Together the two rewrites cut OMAT-0 medium's MD step (SrTiO3, 5,000 atoms) by **16–17 % on the RTX 5060 Ti and
  the RTX 4000 Ada and 10.5 % on the A100-SXM4-80GB**, in both build orders.
- On the A100 the fused forward alone is 29 % slower: it needs the chunked backward, whose chunking costs 31 % there
  (`results/a100-chunk-size-20261103`). `wprime_backward` writes no chunked weight gradient, so with it the chunking
  never runs. The default is therefore the pair, never the fused forward alone.
- Force gates pass on both measured GPUs (IEEE float32 rounding only; no precision change). With tf32x3 on (the
  A100 default) the pair still saves 11.5 % (§4).

## 1. What was built

- **`generic_fused_folded_forward`** (`fastmace/rewrites/generic`, config field `fused_folded_forward`): MH-1's
  `FusedFoldedForward` (the folded radial layer's product inside the streaming convolution's forward, OPT8-T02)
  attached to a generic interaction's folded path (`generic_folded`, `chunked_backward_edges`, no
  `generic_recompute`). Exact; requires `generic_chunked_conv_backward`.
- **`generic_wprime_backward`** (`fastmace/rewrites/radial`, config field `wprime_backward`): MH-1's `wprime_backward`
  (OPT9-T03: each edge's weight gradient dotted with `w'(r)` from per-piece derivative tables inside the streaming
  backward; no `dw` chunk stores, no folded input gradient, no table backward) on the generic folded radial.
  Rounding; requires `generic_radial_tables` and `generic_chunked_conv_backward`; float32, not reproducible.
- `GenericInteraction` routes through `streaming_message(..., wprime_lengths=...)` when both tables and the chunked
  path are on and the record is attached; `_wprime_case` and `streaming_message` read `generic_folded` where it
  exists.
- **Planner (decision 66):** `fused_folded_forward` and `wprime_backward` are on together for `omat0-medium` on the
  three GPUs, in float32 plans that are not reproducible, with `radial_tables` on. On the A100, where decision 66
  leaves the generic chunked path off (`DECISION_66_GENERIC_CHUNKED_GPUS`), `chunked_conv_backward` stays on as their
  carrier and goes off with them (float64, reproducible, or `wprime_backward="off"`). Other GPUs and models keep both
  off; explicit `"on"` enables them anywhere the requirements hold.
- **Not covered:** tiled calls. The fused forward and `wprime_backward` paths pass `rows=None`, so a tile's sweep 2
  does not trim to the tile's own receivers there. A 1M-atom tiled A/B is open.

## 2. Force gate (`tools/force_stress_gate.py`, OMAT-0 medium, NaCl and NbCl5, both rewrites on)

Ratio of the production error to the float32 calculator's (both against float64):

| GPU | NaCl energy | NaCl forces | NaCl stress | NbCl5 energy | NbCl5 forces | NbCl5 stress |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 4000 Ada (leto) | 1.30 | 0.70 | 0.87 | 0.85 | 0.97 | 0.65 |
| A100-SXM4-80GB (Modal), chunked on | 1.31 | 0.75 | 0.89 | 1.05 | 0.99 | 0.51 |

All pass (energies far under the 10 meV floor: at most 0.2 meV).

## 3. MD step A/Bs (`tools/ab_rewrites.py`)

OMAT-0 medium, Benchmark_loup SrTiO3 10×10×10 (5,000 atoms), 1 fs, skin 0.5 Å, 200 warm-up steps, 8 blocks of 200
steps, same process, b/a in both build orders (abab / abba). a = the GPU's default plan before this decision.

| GPU (provider) | a (ms) | fused forward | wprime_backward | both |
| --- | --- | --- | --- | --- |
| RTX 5060 Ti (local, WSL) | 45.3–46.8 | 0.900 / 0.894 | 0.908 / 0.928 | **0.839 / 0.831** |
| RTX 4000 Ada (leto) | 49.6–49.8 | 0.924 / 0.923 | 0.909 / 0.910 | **0.832 / 0.833** |
| A100-SXM4-80GB (Modal), b with `chunked_conv_backward=on` | 22.7–22.8 | 1.289 / 1.287 | 0.912 / 0.913 | **0.895 / 0.894** |

The 5060 Ti is a desktop GPU (noisier: wprime 0.908 / 0.928). On the A100, wprime alone with the chunked carrier
(0.912) already contains no chunking; the fused forward adds 2 % on top of it.

## 4. A100 with tf32x3_gemms on in both a and b

tf32x3_gemms is the A100's default for OMAT-0 since the same day (`cutlass-tf32x3.md`), so the A/Bs were repeated
with it on in a and b (Modal A100-SXM4-80GB, source `14fe2e47`, the merged branches; b with
`chunked_conv_backward=on`):

| a (ms) | fused forward | wprime_backward | both |
| --- | --- | --- | --- |
| 21.49–21.51 | 1.304 / 1.311 | 0.907 / 0.907 | **0.885 / 0.885** |

The gains add up: the pair saves 11.5 % on top of tf32x3 (22.8 → 21.5 → 19.0 ms). The gate with both and tf32x3
passes; its ratios are tf32x3's (NaCl energy 4.71, forces 0.83, stress 1.59; NbCl5 11.7, 1.07, 1.69; energies at
most 1.2 meV, under the 10 meV floor), not the pair's.

## Runs

- `results/generic-wprime-20261103/rtx5060ti-ffwd` (source `b301ee58`), `rtx5060ti` (source `5b07cef3`): local
  RTX 5060 Ti, WSL, torch 2.14.1+cu130.
- `results/generic-wprime-20261103/rtx4000ada`: leto, `mace_cueq_oeq` (NAS), source `5b07cef3`, idle check in
  `idle_check.txt`.
- `results/generic-wprime-20261103/a100-sxm4-80gb`: Modal A100-SXM4-80GB, driver 610.57.04, source `5b07cef3`
  (`tools/modal/generic_wprime_ab.sh`, `CHUNKED=on`); `a100-sxm4-80gb-tf32x3`: the same with tf32x3_gemms on, source
  `14fe2e47`.
