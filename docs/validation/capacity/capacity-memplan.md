# Capacity screen B: Inductor's memory planning: no memory change

Conductor, 30 October 2026: compile the force model with Inductor's
`{"memory_planning": True, "memory_pool": "intermediates"}` and measure memory, setup and step time. It was a
screen: adopt nothing unless it clearly wins. MH-1 hybrid float32 on the local RTX 5060 Ti, on `main` at `f3c635b9`.

## Verdict

**Falsified for memory: the peak and the graph pool are unchanged to the byte.** Nothing is adopted.

| NbCl5 | `main` peak | memory planning, peak | `main` pool | memory planning, pool |
| --- | ---: | ---: | ---: | ---: |
| 4×4×8, 9,216 atoms | 4.100 GiB | 4.100 GiB | 3.631 GiB | 3.631 GiB |
| 6×6×20, 51,840 atoms | 14.285 GiB | 14.285 GiB | 13.365 GiB | 13.365 GiB |

CUDA-graph capture, replay, release and re-capture all worked with the options on (the ledger's full sequence ran
at both sizes). The peak allocated memory is the same as on `main` too: 12.014 GiB at 51,840 atoms and 3.628 GiB
at 9,216.

**Why nothing changes (an explanation, not measured).** Inductor's memory planning packs the buffers of one compiled graph's generated code into
pools it allocates itself. The force call's peak is set by the CUDA-graph pool that holds every captured stage. The
caching allocator already reuses freed blocks inside that pool, and the planner's pool is one more allocation in it.
In the force backward, what is alive at the peak is the saved activations and the gradients. Those are inputs and
outputs of the backward graph, and Inductor's planner does not place them.

**Step time and setup.** Not A/B'ed, since the screen saves no memory and decision 67 plans on a change for memory
only. The ledger's wall time from a cold compiler cache was 71.4 s (9,216 atoms) and 79.8 s (51,840) with the options,
against 68.3 s and 77.9 s without them. The run without them is screen A's first control (`--drop-node-feats`, which
leaves the memory as on `main`), also from a cold cache. That is one process each, so
treat the difference as noise until measured.

**Against recompute.** Not applicable: the screen removes nothing.

## How it was run

`tools/bench/memory_ledger.py --inductor-option memory_planning=True --inductor-option memory_pool=intermediates`
(branch `fastmace/cap-reinplace`, commit `626eb19c`). The tool merges the options into `torch.compile(options=...)`
inside `compile_model`, the route the model's Inductor-option records take. A fresh compiler cache was used
(`LEDGER_CACHE`). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, local RTX 5060 Ti (sm_120, WSL2, driver
610.88), torch 2.14.1+cu130, environment `mace_cueq_oeq`. `main` is the ledger at main's library code
(runs p2 and p8 of the ledger work; the 51,840-atom one is
`results/cap-ledger-20261030/ledger/ledger-NbCl5-float32-51840.json`, the 9,216-atom one is
`results/cap-pool-20261030/ledger/ledger-NbCl5-float32-9216.json`).

Records: `results/cap-memplan-20261030/` (`ledger/`, `jobs.log`).
