# A planned workspace for `segment_gemm` (option 1): falsified in the tracing spike

The owner chose option 1 of [capacity-pool.md](capacity-pool.md) on 30 October 2026, through the conductor:

- `segment_gemm`'s two custom ops write their products and input gradients into views of one planned buffer;
- the spike comes first, with a stop if capture or tracing fights it.

## Verdict

**Stopped at the spike. Tracing fights the layout that would save memory. No library change.**

The simulation in capacity-pool.md credits the workspace with 0.76 GiB at 51,840 NbCl5 atoms only for one layout.
There, each interaction's input gradients reuse the storage of that interaction's product, which is dead by then:
the products and gradients pack into 3.16 GiB, and the pool beside them drops to 9.42 GiB.

| Layout | Under `torch.compile` (Inductor, torch 2.14.1) | Memory, simulated at 51,840 atoms |
| --- | --- | --- |
| **A.** Product into a workspace tensor; gradients into a separate workspace tensor | **Works.** The forward op runs on the workspace directly: `auto_functionalized_v2` is reinplaced on the graph input, with no copy. The backward op writes the gradients into their region. The results are bitwise equal to the compiled original. | products 3.16 + gradients 1.19 + pool 9.42 = **13.77 GiB**, against 13.34 today: **worse** |
| **B.** Product into a workspace tensor; gradients into the product's own storage, passed to the backward op as its mutated argument | **Not reinplaced.** In the backward the saved product is not a kept-mutation graph input. Inductor therefore allocates a fresh buffer of the product's size, writes the gradients there and copies them back. That is the allocation the layout was meant to avoid, and the gradients do not land in the workspace. | the 0.76 GiB gain is lost to that extra buffer |
| **B′.** As B, but the gradient region is a separate tensor sharing the product's storage (aliasing hidden from the compiler) | **Not safe.** In layout A's backward, Inductor reused the saved product's buffer for its own temporaries once the product was dead (`buf0 = getitem_1  # reuse`, the product's gradient written in place). With hidden aliasing, our op's writes could overlap live Inductor buffers. Whether they do depends on Inductor's scheduling, which no test could pin down across torch versions. | not considered |

So only layout A works mechanically, and it does not save memory. The packing gain needs the gradients to reuse the
product's storage, and Inductor cannot do that in the backward without the extra buffer or hidden aliasing.

## The spike

**Script:** `results/cap-workspace-20261030/spike_workspace.py`, run on the CPU in WSL2 (`fastmace_cpu`, torch
2.14.1+cpu).

- **The ops.** Two out-argument custom ops have `segment_gemm`'s cuBLAS-branch arithmetic:
  `joined_products_into` (`mutates_args=("out",)`), and `input_grads_into` or `input_grads_into_space`.
- **The wrapper.** They sit inside an `autograd.Function` that marks the workspace dirty and returns it.
- **The comparison.** Forward plus backward through `sin(p) * p`, eager and with `torch.compile(backend="inductor",
  fullgraph=True)`, each against the existing `joined_products` compiled the same way.

**Results** (`spike-into.log` and `spike-reuse.log` hold Inductor's generated code):

| Variant | Energy and input gradients | Product in the workspace | Gradients in the workspace |
| --- | --- | --- | --- |
| A, eager | bitwise equal | yes | yes |
| A, compiled | bitwise equal (max difference 0) | yes | yes |
| B, eager | bitwise equal | yes | yes |
| B, compiled | bitwise equal | yes | **no**: a fresh buffer, copied back |

The script's `bitwise` column also compares the returned product after the backward. That is False wherever the
workspace was overwritten after the product's last use: with the gradients (B), or with the product's own gradient
(A compiled, by Inductor's reuse). This is expected, not an error.

**Finding along the way:**

- `ctx.mark_dirty` on a *view* of the workspace makes AOT autograd's joint tracing fail with an internal assert
  (`fn_edge.is_valid() == this_edge.is_valid()`, `torch/csrc/autograd/functions/tensor.cpp:136`).
- A whole base tensor per region works.
- So a single buffer could only be carved into regions through tensors that share its storage without being views,
  which brings back B′'s hidden aliasing.

**Not reached:** the CUDA-graph capture at 9,216 and 51,840 atoms was not run. The stop came from tracing, before
the layout that would have needed the capture test.

## What remains of the pool slack

capacity-pool.md stands. In principle 1.47 GiB is recoverable by placement. What could still reach it:

- **A capture-time allocator** following an offline plan (option 2 there). This needs a SPEC decision.
- **An Inductor-level change:** a custom post-grad pass that reinplaces the backward mutation of a saved activation,
  or a supported way to declare the alias. That depends on torch internals.

Both were tried afterwards; see the follow-up below.

## Follow-up, 30 October: both remaining routes tried

The owner asked for both routes above. **Neither gives memory. No library change.**

### Route 2: an Inductor post-grad pass that writes the backward in place

**Prototype:** `tools/bench/workspace_prototype.py`, measured with `tools/bench/memory_ledger.py
--workspace-prototype` on branch `fastmace/cap-reinplace`. Records: `results/cap-reinplace-20261030/` (runs w1 to
w4, `jobs.log`).

- **The ops.** `segment_gemm`'s products and input gradients go through two out-argument custom ops into one
  workspace per gated segment, of shape (atoms, width), outside the graph pool. The backward op writes the
  gradients into the saved product's storage, which is layout B above.
- **The pass.** `InPlaceIntoSaved`, a `CustomGraphPass` on `torch._inductor.config.post_grad_custom_post_pass`.
  It rewrites the backward's `auto_functionalized_v2` of that op into a direct call on the saved product and drops
  the copy back. That removes the fresh buffer layout B had.
- **On the CPU** the rewritten backward runs in place and is bitwise equal to the original.
- **In the real model** the pass also applies: the generated code writes the gradients into the workspace. But
  Inductor keeps the incoming gradient buffer alive across the op, until the end of its scope. We tried passing
  the arguments positionally, and an op that returns an empty tensor. Neither changed it.

| NbCl5 float32 | `main` peak | prototype peak | `main` graph pool | prototype graph pool |
| --- | ---: | ---: | ---: | ---: |
| 4×4×8, 9,216 atoms | 4.100 GiB | 4.119 GiB | 3.631 GiB | 3.104 GiB |
| 6×6×20, 51,840 atoms | 14.285 GiB | 14.793 GiB | 13.365 GiB | 10.436 GiB |

The pool shrinks, because the products moved out of it. But the workspaces outside the pool cost more than the
pool saves: +0.51 GiB at 51,840 atoms. capacity-pool.md's packing simulation, with the incoming gradient freed at
the op, gives −0.75 GiB. Freeing it needs a change to Inductor's buffer lifetimes, not just a graph pass, so this
route stops here.

### Route 1: a capture-time allocator following the offline plan

**Not reachable without a SPEC decision; not built.**

- `torch.cuda.MemPool` with a pluggable allocator only supplies the pool's segments. Which block a tensor gets
  inside those segments is still the caching allocator's choice. So the offline plan cannot be imposed that way.
- A second `MemPool` for `segment_gemm`'s kernels (`use_mem_pool`, allowed during capture) was measured in
  [capacity-pool.md](capacity-pool.md): 4.53–4.67 GiB peak at 9,216 atoms against 4.10, and the build fails at 51,840
  atoms. Two pools cannot reuse each other's free blocks.
- Placing every block by plan needs a native allocator for the whole process, installed with
  `torch.cuda.memory.change_current_allocator` before the first CUDA allocation. That replaces the caching
  allocator, and with it the graph pools and expandable segments FastMACE relies on (SPEC §14). That is a SPEC
  decision for the owner, not a capacity task.

### Caveat: a custom pass's `uuid()` is Inductor's cache key

Inductor keys its FX graph cache on the custom pass's `uuid()`, not on the pass's code. The prototype's pass changed
twice while keeping its uuid, so ledger runs w2 and w3 (`results/cap-reinplace-20261030`) may have been served
code compiled with the earlier pass. The uuid was bumped (`fastmace-proto-in-place-into-saved-3`), and run w4 used a
fresh compiler cache. w1 to w4 measure the same, so the verdict stands.

For any future custom pass or partitioner, change its uuid with every change to its code, as
`save_transcendentals` does with `SAVE_TRANSCENDENTALS_UUID`, and measure from a fresh compiler cache. The
spike above did not use a custom pass, so its own results are not affected.

## What changed

- `results/cap-workspace-20261030/`: the spike script and its two logs.
- This report.
- Follow-up: `tools/bench/workspace_prototype.py`, `tools/bench/memory_ledger.py --workspace-prototype`, and
  `results/cap-reinplace-20261030/`.
- No library code changed.
