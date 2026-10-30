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

Neither is started.

## What changed

- `results/cap-workspace-20261030/`: the spike script and its two logs.
- This report.
- No library code changed.
