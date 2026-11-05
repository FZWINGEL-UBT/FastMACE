# Capacity reports

Memory capacity of the whole-system path (29 October – 1 November 2026; plan `plans/capacity.md`): capacity tasks CAP-T01–T12, the memory-rescue ladder (rule 12, decision 67), setup and neighbour-list fixes.

Columns: file, date added, title, and the report's own verdict line or first sentence.

| File | Added | Title | Verdict / first sentence |
| --- | --- | --- | --- |
| [capacity-16gb](capacity-16gb.md) | 2026-10-29 | Capacity on the RTX 5060 Ti 16 GB: atoms and edges in NVT MD | decision 51). It is task CAP-T02 and waits for the conductor. |
| [capacity-T01](capacity-T01.md) | 2026-10-29 | CAP-T01: one force graph near the memory limit | Plan task CAP-T01 (plans/capacity.md), owner request of 29 October 2026. |
| [capacity-T02](capacity-T02.md) | 2026-10-29 | CAP-T02: rule 12 keeps MH-1's fused path, and the estimate predicts the measured peak | Plan task CAP-T02 (plans/capacity.md), decided by the owner on 29 October 2026: "It should only do this if memory would not be enough" (SPEC §18, decision 67). |
| [capacity-T03](capacity-T03.md) | 2026-10-29 | CAP-T03: device memory outside PyTorch's allocator | Plan task CAP-T03 (plans/capacity.md). |
| [capacity-T04](capacity-T04.md) | 2026-10-29 | CAP-T04: fragmentation at the peak | Plan task CAP-T04 (plans/capacity.md), owner go-ahead on 29 October 2026, measured on one structure (NbCl5) as the owner asked. |
| [capacity-T05](capacity-T05.md) | 2026-10-30 | CAP-T05: capture snapshots in pinned host memory | Plan task CAP-T05 (plans/capacity.md). |
| [capacity-T06](capacity-T06.md) | 2026-10-30 | CAP-T06: a smaller noise batch | Plan task CAP-T06 (plans/capacity.md). |
| [capacity-T07](capacity-T07.md) | 2026-10-29 | CAP-T07: recompute the per-edge tensor-product weights when memory requires it | Plan task CAP-T07 (plans/capacity.md). |
| [capacity-T07b](capacity-T07b.md) | 2026-10-30 | CAP-T07b: recompute_conv_weights with the w-rebuilding kernels | Plan task CAP-T07b (plans/capacity.md), the follow-up recommended in capacity-T08-inventory.md section 2. |
| [capacity-T08-chunks](capacity-T08-chunks.md) | 2026-10-30 | CAP-T08: recomputing the node-local segments (`recompute_node_segments`) | Plan task CAP-T08 (plans/capacity.md), idea 2 of the owner's capacity list of 30 October 2026, through the conductor: cut the per-atom state the force call keeps, which … |
| [capacity-T08-inventory](capacity-T08-inventory.md) | 2026-10-30 | CAP-T08 checks: what the force call keeps, and the cheap float32 levers | The conductor's float32 checks of 30 October 2026, asked for by the owner, ahead of CAP-T08 (plans/capacity.md). |
| [capacity-ledger](capacity-ledger.md) | 2026-10-30 | Memory at the limit, and the first padded edge size (ideas 4 and 5) | The owner's ideas 4 and 5 of 30 October 2026, through the conductor, after CAP-T09 (capacity-nlist.md): |
| [capacity-memplan](capacity-memplan.md) | 2026-10-30 | Capacity screen B: Inductor's memory planning: no memory change | Conductor, 30 October 2026: compile the force model with Inductor's `{"memory_planning": True, "memory_pool": "intermediates"}` and measure memory, setup and step time. |
| [capacity-nlist](capacity-nlist.md) | 2026-10-30 | CAP-T09: a leaner neighbour list | Plan task CAP-T09 (plans/capacity.md). |
| [capacity-outputs](capacity-outputs.md) | 2026-10-30 | Capacity screen A: the compiled model leaves out `node_feats` | Conductor, 30 October 2026: check whether model outputs that no evaluation reads, such as MACE's concatenated `node_feats`, stay alive through the force backward; if so, … |
| [capacity-pool](capacity-pool.md) | 2026-10-30 | The graph pool's packing slack: where it comes from, and what recovers it | The owner's go-ahead of 30 October 2026, through the conductor, after capacity-ledger.md: |
| [capacity-tiled-t1t2](capacity-tiled-t1t2.md) | 2026-10-30 | Tiled capacity path: T2 (smaller global state) and T1 (one padded CUDA graph per sweep) | The owner's approved next steps T1 and T2 of plans/capacity-ideas-20261030.md, through the conductor, after capacity-tiled.md. |
| [capacity-tiled](capacity-tiled.md) | 2026-10-30 | Tiled capacity path: receiver tiles, three sweeps (prototype) | The owner's request of 30 October 2026, through the conductor, after an outside review of Symmetrix-XL: a capacity execution path that keeps only the inter-layer … |
| [capacity-workspace](capacity-workspace.md) | 2026-10-30 | A planned workspace for `segment_gemm` (option 1): falsified in the tracing spike | Results (`spike-into.log` and `spike-reuse.log` hold Inductor's generated code): |
| [chunked-region](chunked-region.md) | 2026-10-31 | Atom-chunked combined node region: `recompute_node_region` (31 October 2026) | Verdict: GO, added as the last memory-rescue record. |
| [compact-chunked](compact-chunked.md) | 2026-10-31 | Chunked neighbour-list compaction (exact) | result: Triton contracts `a * b + c` into FMA, while PyTorch's eager kernels round each product. |
| [estimate-band](estimate-band.md) | 2026-11-01 | Rule 12's estimate per side and per stage (1 November 2026) | The owner's item, through the conductor: make rule 12's memory estimate more accurate. |
| [estimate-probe](estimate-probe.md) | 2026-11-01 | The near-limit memory probe (1 November 2026) | The conductor's decision (ii) after estimate-band.md: |
| [estimate-refit](estimate-refit.md) | 2026-10-31 | Rule 12's estimate refitted for the memory rescue (31 October 2026) | Verdict: the estimate stays within 1.08–1.39 times the measured peak at every stage of the memory rescue, where it |
| [l2-gap-audit](l2-gap-audit.md) | 2026-10-31 | M1 (L2 audit) and M2 (gap trace), 31 October 2026 | Eighth list, measurements M1 and M2 (plans/capacity-ideas-20261030.md), plus item 4 of the ninth list. |
| [m3-ledger](m3-ledger.md) | 2026-10-31 | M3: where the memory sits at the new limit (31 October 2026) | Measured only. |
| [pair-search-grid](pair-search-grid.md) | 2026-10-30 | The GPU pair screen beyond 2.1M atoms (pair-search grid fix) | The conductor's go of 31 October 2026, after the tiled capacity path's T2 (capacity-tiled-t1t2.md). |
| [rebuild-a](rebuild-a.md) | 2026-10-31 | Rebuild-`a`: the first interaction's radial table rebuilt in the backward (`recompute_first_table`) | Capacity idea rebuild-`a` (plans/capacity-ideas-20261030.md), through the conductor. |
| [region-norm](region-norm.md) | 2026-10-31 | The message divided inside the chunked region (M3 target 1, 31 October 2026) | Verdict: GO. |
| [region-recompute](region-recompute.md) | 2026-10-31 | The combined region's recompute in the backward (31 October 2026) | Verdict: STOP. The region's recompute is necessary work done efficiently. Less than 2 % of the step is |
| [region-tuple](region-tuple.md) | 2026-10-31 | The combined region's inputs as a tuple, and the contraction inside the region (31 October 2026) | Verdict: STOP for both. The tuple pass-through leaves the force call's peak unchanged, and the contraction output |
| [setup-across-compositions](setup-across-compositions.md) | 2026-10-30 | Engine setup across compositions (float32, 30 October 2026) | Owner request via the conductor, 30 October 2026. |
| [setup-fixes](setup-fixes.md) | 2026-10-30 | Setup fixes F1, F2, F3 and F5 | Fixes from setup-across-compositions.md, "Fixes, ranked by expected seconds saved". |
