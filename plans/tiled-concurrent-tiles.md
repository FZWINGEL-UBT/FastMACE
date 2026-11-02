# Two tiles at once: tensor cores and CUDA cores busy in different tiles (plan, 2 November 2026)

Owner idea (2 November 2026): use the tensor cores beside the CUDA cores, not inside one tile but at the same time
in different tiles. Status: plan only, nothing built. It builds on the wavefront schedule
(`fastmace/omat0-wavefront`, `docs/validation/tiled-wavefront.md`), which already knows which tile sweeps are
independent.

## 1. The idea in one paragraph

Today the tiled call runs one tile-sweep after another on one stream. Each tile-sweep alternates between phases
that load different units: FP32 GEMMs and the convolution on the CUDA cores (SIMT), and gathers, scatters,
`index_add_` and elementwise kernels that are bound by memory bandwidth. The tensor cores are idle throughout,
because FP32 with TF32 off (owner decision, 9 October 2026) never uses them. Two lanes on two CUDA streams would
run two independent tile-sweeps at once, so one lane's memory-bound phase overlaps the other lane's compute. In a
second variant, one lane's GEMMs run as 3×TF32 on the tensor cores while the other lane's run on the CUDA cores.

## 2. What we already know (and why it might pay off now)

| Evidence | What it says for this idea |
| --- | --- |
| `tiled-call-memory.md`, `tiled-omat.md`: a compiled tiled call keeps the device busy 95-96 % | Filling idle gaps between kernels is worth at most about 4 %. Any gain has to come from kernels running side by side. |
| `fp32-profile-20261028.md`: cuBLAS GEMMs reach 10-15 TFLOPS at 21-25 % occupancy; elementwise kernels 160-230 GB/s | Neither the SIMT units nor the memory bus is saturated by a single kernel, so there is room for co-resident kernels. |
| `tf32-screen.md` (Ada): 3×TF32 alone is 12-22 % slower per step than FP32; 1×TF32 fails the force gate by 3,500× | The tensor cores only pay off if their work runs *beside* SIMT work, never instead of it. Only 3×TF32 (or rz3) is accurate enough. |
| `tc-gemm-screen.md` (A100): CUTLASS rz3 3×TF32 passes the force gate and NVE; in one kernel, SIMT and tensor-core work co-execute badly (keep 0.57-0.75) | Accuracy is settled for rz3 on the A100, not yet measured on the Ada or the 5060 Ti. Mixing both in one kernel is closed. |
| `opt5-T01-scheduler-probe.md` (sm_120): SIMT and MMA warps on one SM give at most 1.07× | Each SM sub-partition issues one instruction per clock to either unit. Two kernels sharing an SM hit the same limit, so the tensor-core variant is bounded by this. |
| Whole-system MD (`tc-gemm-screen.md`): the step is one dependency chain with only 7-69 µs of concurrent kernel time | That's why concurrency didn't pay inside one step. Tiles differ: thousands of independent tile-sweeps per call. |

**Why now:** the wavefront makes the dependencies explicit (`needs`, `ready`). Sweep 1 of a tile ahead and sweep 3
of a tile behind are independent of the current sweep-2 tile, so a second lane always has work without changing
results beyond the existing rounding class. The tiles are also large (R = 4,096-16,384), so each lane's kernels
are long enough to overlap.

**Honest prior:**
- **Plain two-lane FP32:** 0-10 % shorter calls.
- **The tensor-core lane on the Ada and the 5060 Ti:** a few per cent beyond that at best (sub-partition issue
  limit). It might be more on the A100, where tensor-core TF32 is 8× its FP32 rate.

## 3. Stages with stop criteria

### Stage 0: where the time goes (one GPU hour, measurement only)

- **What:** a kernel-class profile of one compiled tiled call. MACE-OMAT-0 medium, SrTiO3 975,560 atoms,
  R = 16,384, on murphy or leto (Ada), and the same locally on the 5060 Ti.
- **Classes:**
  1. SIMT FP32 GEMMs (cuBLAS, `segment_gemm`, Inductor `mm`);
  2. the convolution and symmetric contraction (OEq, cuEquivariance);
  3. memory-bound kernels (gathers, scatters, `index_add_`, elementwise, the host-state copies);
  4. the radial tables.
- **Per class:** time share, achieved occupancy, SM throughput and DRAM throughput. Use `ncu` on the top 10
  kernels; Nsight's "SOL" sections give the compute or memory bound.
- **Bounds:**
  - Two-lane FP32 overlap is at most about min(compute-bound share, memory-bound share), discounted by co-residency.
  - The tensor-core lane is at most about 0.45 × the GEMM share. That figure comes from rz3's measured GEMM rate on
    the Ada against SIMT FP32 (0.8×) and assumes perfect co-issue; opt5-T01 says to expect much less.
- **Stop** if the two-lane bound is below 5 % on both GPUs.

### Stage 1: the co-run probe (no library change, about two GPU hours)

- **What:** a script in `tools/bench/` that builds two padded slots of the same bucket and runs one sweep's
  compiled frame on each on two streams:
  - **(a) Back to back** on one stream: the baseline.
  - **(b) Concurrently** on two streams.
  - **(c) As (b), with an SM split** between the lanes (CUDA green contexts, or a `cudaStreamAttr` SM limit where
    torch 2.14 exposes it). Full-grid GEMMs otherwise block co-residency.
  - **(d) As (b) or (c), with one lane's GEMMs as 3×TF32.** Triton `tf32x3` through the existing opt-in
    `tf32x3_gemms` record, and CUTLASS rz3 from `tools/bench/tc_gemm/` where it builds for sm_89/sm_120.
- **Combinations:** the same sweep twice, and the realistic mixes (sweep 2 with sweep 1, sweep 2 with sweep 3).
- **Measure:** throughput, i.e. tile-sweeps per second of (b), (c) and (d) against (a), in alternating blocks in
  one process, in both orders. Also each lane's latency.
- **GPUs:** RTX 4000 Ada (leto or murphy) and the RTX 5060 Ti (local); the A100 on Modal only if (b) or (d)
  passes on one of them (budget guard; the October credit is used up).
- **Continue** if (b) or (c) gives at least 1.05× on at least one GPU, or (d) at least 1.05× over the better of
  (b) and (c).
- **Stop** otherwise, and record the numbers as a negative result.

### Stage 2: two lanes in the tiled call (library, behind an option)

`TileConfig.lanes` (1 or 2; default 1 until stage 3 decides).
- **Schedule:** a second pass over the wavefront steps assigns each step to a lane.
  - Lane A runs sweep 2 in tile order.
  - Lane B runs the sweep-1 steps of upcoming `needs` and the sweep-3 steps that became ready.
  - Cross-lane order comes from CUDA events: a sweep-2 tile waits for its needed sweep-1 tiles' events, and a
    sweep-3 tile waits for its ready tile's sweep-2 event.
- **Double-buffered slots:** each lane has its own padded slot per bucket.
  - This doubles the workspace: about 0.5 MB per receiver per lane.
  - The budget measures it. Two lanes are only allowed where the second workspace fits; at the capacity limit the
    call falls back to one lane.
  - So this is a speed feature for systems below the limit (about 1-60M atoms), not for the record size.
- **Shared state that must become per lane:**
  - `strain_grad` (a non-atomic `add_`): one accumulator per lane, summed at the end.
  - The forces' and node energies' `index_add_` are atomic and can stay shared (rounding class unchanged).
  - The host state's pending writes and flushes: one queue per lane, plus the cross-lane rule that a read waits
    for the other lane's writes to the same rows. The wavefront dependencies already guarantee the order; the
    flush must wait on the producing lane's event, not on "the previous step".
  - The streaming kernels' `streaming_count` buffer and the species tables are bound into the model per slot
    (`register_buffer`).
    - A compiled frame reads the buffer's address at launch, so rebinding before each lane's launch works on one
      Python thread.
    - Eager OEq kernels must be checked the same way.
  - Inductor's workspace and cuBLAS handles are per stream in torch, so nothing to share there.
- **Python:** one thread launches both lanes alternately. A tile's host syncs (`unique`, the list search's
  `int(counts.sum())`, the host-state `.cpu()`) stall the launching thread, not the queued kernels. If the
  launch thread becomes the bottleneck (gaps in the profile), move the lists' search for the next tile into a
  host worker thread.
- **Tensor-core lane (only if stage 1 (d) passed):** lane B's frames are compiled with the 3×TF32 record.
  - This changes the 9 October precision decision for those GEMMs. It needs an owner decision (§18), the force
    gate on each GPU and NVE before it can be a default.
  - Lane A stays IEEE FP32.
- **CPU tests:**
  - A CPU stream stand-in (the lanes in sequence) must give the wavefront's results.
  - The energy must stay bitwise against one lane: x, sc and the adjoints are written by the same steps in the
    same per-atom order.
  - Forces and stress within the rounding class.
  - The per-lane strain sum, the per-lane flush order, and refusals (graphs mode, state on the device).

### Stage 3: adoption

- **A/Bs:** same-process A/Bs (decision 66), lanes 1 against 2, both build orders, on the Ada and the 5060 Ti at
  5M and 20M SrTiO3 (OMAT-0) and 9.26M NbCl5 (MH-1).
- **Gate:** the force/stress gate and 20 ps NVE drift for the 2-lane path; with the tensor-core lane, also
  against float64.
- **Default:** two lanes become the default only on the GPUs where they measured faster, and only below the
  memory limit.

## 4. Risks

- **Co-residency:** full-grid GEMMs and Triton kernels fill every SM, so the two lanes may just time-slice. Stage 1
  (c) tests the SM split. If green contexts are not reachable from torch 2.14.1 + cu130, the split needs a small
  extension; that's no environment change, the same route as the OEq kernels.
- **Memory:** a second workspace costs as much as a whole tile at the limit. That's why the feature is limited to
  below the limit.
- **Determinism:** unchanged class. The hybrid is already not bitwise reproducible (OEq atomic sums).
- **Complexity:** the host-state queue becomes per lane. That is the part most likely to hide an ordering bug, so
  its CPU tests come first.

## 5. Cost

- **Stage 0:** about 1 GPU hour on one Ada plus the local 5060 Ti.
- **Stage 1:** about 2 GPU hours on each. The A100 only if a local result passes: about $3-5 on Modal, and only
  after the monthly credit resets.
- **Stage 2:** about two working days for the code and its CPU tests.
- **Stage 3:** about 4 GPU hours.

Stages 0 and 1 can run while the capacity scans use leto and murphy: the 5060 Ti locally, plus one more
workstation GPU (four at most).
