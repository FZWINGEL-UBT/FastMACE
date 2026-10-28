# Future work: an O(N) neighbour-list rebuild for very large cells

Status: **future work, not scheduled** (owner, 28 October 2026). Nothing here is built or measured; the
numbers below are estimates.

## Today

`SkinNeighbourList.build_gpu` screens candidate pairs with one Triton kernel
(`fastmace/neighbors/pair_search.py`): every 32-atom sender tile against every 32-atom receiver tile, over
all periodic images. A rebuild therefore costs O(N² × images). The skin keeps this cheap: the per-step
rebuild test is O(N), and a rebuild happens only every few tens of steps. The D3 pair list
(`fastmace/neighbors/d3_list.py`) uses the same kernel as a half list, with the larger D3 cutoff.

## When the N² matters

These are estimates, assuming a GPU screens roughly 10¹¹–10¹² pairs per second:

| Atoms | One rebuild | Effect |
| --- | --- | --- |
| ~10⁴ | well under 1 ms | negligible against the model step |
| ~10⁵ | tens of ms | small when spread over 20 or more steps |
| ~10⁶ | seconds | dominates the run |

So the rebuild matters from a few hundred thousand atoms. The D3 list reaches that point sooner, because
of its larger cutoff.

## Options (exact: the same edge set as today)

1. **Spatially sorted tiles with bounding-box pruning** (the cluster-pair idea of GROMACS's nbnxm). This is
   the preferred option.
   - Sort atoms along a space-filling curve (Morton or Hilbert, in fractional coordinates for triclinic
     cells) at each rebuild.
   - Give each 32-atom tile a bounding box. Launch the existing tile kernel only on tile pairs (and images)
     whose boxes lie within the cutoff plus skin of each other.
   - Each tile then meets a constant number of neighbour tiles, so the cost is O(N).
   - It reuses the kernel and everything after it: buckets, the float64 decision, receiver sorting.
   - The earlier Morton atom-order screen (no gain inside the model at our sizes;
     ResearchModule `studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`) was a different use of the sort.
     Here the sort only orders the search.
2. **GPU cell list.**
   - Bin atoms into cells at least the cutoff plus skin wide (fractional binning for triclinic cells).
   - Counting-sort the atoms by cell, then search the 27 neighbouring cells.
   - Simple and robust, but it adds a sort and scatter to every rebuild, and it is slower than option 1 in
     sparse cells.
   - Existing code to compare with: NVIDIA's `nvalchemi-toolkit-ops` (installed in
     `~/environments/fastmace_alchemi_compare`) and vesin's GPU backend, if present.
3. **Partial rebuilds.** Rebuild only the neighbours of atoms that moved more than half the skin. This
   saves work in dense, slow systems, but it is hard to combine with CUDA graphs and fixed-shape edge
   buckets. Last priority.

## Other limits large cells hit

- Edge counts above ~2·10⁹ overflow 32-bit edge indices; check every index buffer and kernel.
- The edge buffers and the model's per-edge work scale with N too, and they dominate memory long before
  the list does. The largest cell that fits on one GPU should be measured.
- The candidate-image count for a large cell falls to the 27 nearest images, so image handling needs no
  change.

## Steps when resumed

1. **Measure the crossover.** Time a rebuild and a model step for cells from 10⁴ to the largest that
   fits, on an Ada GPU and an A100 (NaCl and NbCl5 supercells; MH-1, float32 and float64).
2. **Build option 1** behind a `NeighbourConfig` field (`"auto"` switches it on above a measured atom count).
   - Gate: the edge set is bitwise equal to the current list's, after receiver sorting, on random and
     triclinic cells and across skin rebuilds.
   - Apply the same change to the D3 half list.
3. **Compare with option 2** on the same cells. Keep the faster one, or both if the crossover depends on
   density.
4. **Check the 32-bit limits** and the memory ceiling, and document the largest supported cell per GPU.

Optimisations must stay general (no per-GPU or per-crystal tuning). The switching atom count is the only
threshold, and it must hold on every measured GPU.
