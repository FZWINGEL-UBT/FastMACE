# Review of five rejections (11 November 2026, from the user)

Code and records only; nothing was run.

| Idea | Rejection | Main reason |
| --- | --- | --- |
| Half-precision far edges (`HALF_FAR`, 1.05–1.11× slower) | Right for this build, not for the idea | The build dropped what made the lab kernel fast |
| Two half-systems out of phase (1.008) | Test invalid, verdict probably right | Both halves were started in phase |
| TF32 big·big plus FP16 cross GEMM (accuracy) | Invalid | The error comes from a cuBLAS GEMM the idea does not use |
| Formulation search | Holds | – |
| Wavefront row pool on the device | Holds | – |

## Half-precision far edges

`fastmace/kernels/oeq_streaming/half_far.py` (branch `fastmace/half-far-wprime`) differs from the lab kernel that passed:

- **dout is not staged.** Each pair loads all 5,120 dout values of its receiver from global memory, each with a multiply and a conversion; the lab read a staged half row.
- **Pairs are scattered.** An atomic counter fills the pair list, so a receiver's pairs land on unrelated warps, each starting with a cold row. That is about 0.46 GB of extra dout reads per force call on SrTiO3, against about 0.34 GB for the whole float32 kernel.
- **The kernels run one after the other.** The estimate rested on the two FMA pipes working at once (167.7 against 63.8 FMA results per clock), which never happens here.
- **The float32 kernel keeps its blocks.** A skipped edge only idles one warp for one turn.

The A/B fits the dout explanation, though it was not profiled. SrTiO3 (dout 102 MB, above the A100's 40 MB L2): 1.052 → 1.106 as the far share doubles. NbCl5 (40 MB): 1.061 → 1.079.

Two things weaken the idea itself:

- **The lab overstated the gain.** Its float32 side read a random table slab per edge (8.7 ns per edge against 6.06 ns in production). Against production the half kernel is 0.69–0.76, not 0.48.
- **The 5.8 Å gate says nothing for NaCl.** NaCl has no shell between 5.8 and 6.0 Å (a = 5.59 Å; the next shell is at 6.25 Å). The one run with a NaCl shell in half precision (5.5 Å) failed at 8.9×.

**Fair retest:** one kernel, where far pairs take warps in the same block as near edges and read a half row staged once per receiver. Estimate: 1–4 % of the A100 step. Explain the NaCl failure on the CPU oracle first.

## Two half-systems

`run_concurrent` (`tools/bench/half_system_streams.py`) makes both streams wait on one start event every round. The two identical graphs therefore run the same kernels side by side, which is one full-size kernel cut in two. The out-of-phase case never ran.

It is still expected to fail. The edge kernels fill 61,440 of 65,536 registers per SM (8 November overlap bound). Earlier GEMM-beside-edge two-stream tests gave 0.985–1.151 on the 5060 Ti and 0.77–0.98 on the Ada.

## Hybrid GEMM

- **The error is not the idea's.**
  - The cross GEMM enters divided by 2,048. The same cuBLAS FP16 GEMM is accurate to 2e-6 in the bench's 3×FP16 column, so it adds at most about 2e-9.
  - The 6.3e-7 to 1.4e-6 comes from the single-pass cuBLAS TF32 big·big. A fused kernel would keep CUTLASS's big·big pass (1.1–2.7e-7 today).
- **The 2.7e-7 stop rule was too strict.** Plain cuBLAS float32 misses it on five of seven shapes (3.3–4.2e-7).
- **The scheme has been accurate before.** The 26 October Triton 3×FP16 kernel, accumulating in float32, measured 0.63–0.78× of the IEEE error. The 5 November all-FP16 rejection rests on the same cuBLAS GEMM, which ran at 0.49–0.66 of 3×TF32.
- **Speed is open.**
  - The two cuBLAS calls sum to 0.92 over the profile's shapes.
  - The only in-kernel precedent, in the fused forward, was 1.017 on the A100 and 0.851 on the 5060 Ti.
  - A fair test needs a CUTLASS kernel with an FP16 pass. The ceiling is about 5 % of the A100 step.

## The two that hold

- **Formulation search:** today's form ranks 2nd of 27; the best mix saves 5.1 % of the modelled arithmetic. One form outside its list (products accumulated per channel, contracted after the reduction) scores 250 against 251, with a wider reduction.
- **Device pool:** a tie with the host wavefront (1.012 / 0.994). The wavefront's cost at that size is its plan, 100 tiles against 68.

One status is broader than its test: receiver caps above 16,384 were rejected on the 5060 Ti only. The A100's host-state and wavefront calls, which are not captured in CUDA graphs, were not screened.

Suggested order: the NaCl oracle check (CPU, free), then the CUTLASS FP16 pass.
