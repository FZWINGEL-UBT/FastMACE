"""Receiver-tiled, three-sweep force call: a capacity prototype (docs/design/tiled-backend.md).

``TiledForceField`` wraps an eager ``MACEForceField`` of a two-interaction ``ScaleShiftMACE`` whose first
interaction reads species-only sender features (MH-1 with ``reassociate_first``, or stock: the first
interaction's input is the species embedding). One force call runs the rewritten modules on tiles: a sorted set
of receivers, the edges into them and the halo of their senders, laid out as MACE's LAMMPS ghosts so that the
interactions cut their outputs to the receivers themselves. Three sweeps:

1. interaction 0 forward without autograd. Of the inter-layer features h1 only interaction 1's two reads of
   them are kept globally: x = linear_up(h1) and sc = skip_tp(h1), both linear (MH-1: 512 columns each, against
   about 2,100 for h1);
2. interaction 1 forward and backward on x and sc (its linear_up and skip_tp stand in for the stored values): the
   adjoints dE/dx, scattered over the tile and its halo, and dE/dsc of the receivers, and the edge gradients of
   the second interaction;
3. interaction 0 again, with autograd, backward of its own energy terms plus <dE/dx, x> + <dE/dsc, sc> of the
   recomputed h1: the remaining edge gradients.

Only x, sc, their adjoints and per-atom outputs grow with the atom count (MH-1: 8 kB per atom); every other
intermediate is bounded by the tile caps. Forces come from the edge-vector gradients (MACE's edge forces), the
stress from the edge virial on a symmetric strain of the edge vectors (stress = dE/d(strain) / V, MACE's sign).
The result equals the
whole-system call up to float32 rounding (the tiles' GEMMs see fewer rows; the force scatter is atomic).

Prototype scope: eager, no CUDA graphs, fixed cell per call, no PolarMACE, D3, joint embedding or embedding
readout. Tiles are planned on the host from a spatial binning of the fractional coordinates.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import math
import time
from typing import Any

from fastmace.kernels.gpu_models import GPUModels
from fastmace.model.forces import ForceOutput

# Compiled tiles' device memory held between calls beyond eager tiles' (their padded slots and outputs; 0.065 GiB
# measured at 9.26M NbCl5 atoms), and the margin the "auto" mode keeps for the compiled workspaces' first allocation
# (tiled-compiled-default.md).
COMPILED_RESIDENCY = int(0.07 * 2**30)
COMPILED_MARGIN = 2**30
# TileConfig.wavefront "auto": considered from this many layers of the tile grid along its slowest axis (the band of
# live rows is about five layers: three around sweep 2, and the first and last layers, alive for the whole call).
WAVEFRONT_LAYERS = 12
# ... and from this many when the plain state would not fit the host at all: a pool of about three layers and the
# boundary skins is still smaller than the plain state and its alias window, and without it the call fails.
WAVEFRONT_MIN_LAYERS = 4
# TileConfig.lanes "auto": the GPUs where two sweep-2 tiles on disjoint halves of the SMs ran faster than one after
# the other (docs/validation/tiled/tiled-concurrent-tiles.md: 1.09x per pair on the RTX 4000 Ada, 1.05x on the RTX 5060 Ti,
# under cudaMallocAsync); by exact device name (decision 66).
LANE_GPUS = ("NVIDIA RTX 4000 Ada Generation",)
# TileConfig.backward_chunk_scale "auto" (decision 66, by exact device name): the chunked streaming backward's edges
# per chunk times this in the tiles, where it measured faster (results/tiled-ideas-20261102, 1.08M SrTiO3,
# wavefront, R 8,192: x4 -10.5 % on the RTX 5060 Ti, -3.2 % on the RTX 4000 Ada; the RTX PRO 6000's tiles do not
# chunk). A tile's backward runs eagerly, one launch pair per chunk: the L2 rule's chunks were launch-bound there.
TILE_CHUNK_SCALE = {"NVIDIA GeForce RTX 5060 Ti": 4, "NVIDIA RTX 4000 Ada Generation": 4}
# Tiled speed (results/tiled-speed-20261108). None: on for TILED_SPEED_GPUS (decision 66: each keeps more memory
# alive or replaces a kernel, so it is on where same-process A/Bs at 975,560 SrTiO3 atoms measured it faster in both
# orders); True / False (TILE_GRAPH_CACHE: a fraction) are the A/B sides' choice on every device.
# 10 November 2026 (results/tiled-speed-20261110): all of them off against all on at 975,560 SrTiO3 atoms, both orders:
# RTX 4000 Ada device state 0.868 / 0.868, wavefront 0.772 / 0.769; A100-SXM4-80GB 0.792 / 0.789 and 0.782 / 0.785
# (every A100 model: GPUModels matches the family).
TILED_SPEED_GPUS = GPUModels(("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB"))
# TileConfig.compiled_graphs None: on for these GPU families (decision 66: where a same-process A/B measured the
# captured compiled bodies faster in both orders), device state only, and only where the memory guard below fits.
# A100-SXM4-80GB 0.569 / 0.571 at 975,560 SrTiO3 atoms (Alex 4981483; the host-bound call: GPU idle 52 % in nsys), gate
# 3x passed with it on (Alex 4982466); every A100 model (GPUModels matches the family). RTX 5060 Ti: no gain (30^3
# 1.007 / 0.990, 40^3 0.998; local job 49), not listed.
COMPILED_GRAPH_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB",))
# The memory guard of TileConfig.compiled_graphs (on or None): the first compiled call runs uncaptured and measures its
# tile workspace; the bodies are captured from the next call on when ``available`` holds one capture pool of that
# workspace per slot, plus the uncaptured warm-ups' own workspace, each with this margin (``compiled_graphs_fit``).
# A captured call that still runs out of device memory drops its graphs and runs again without them.
COMPILED_GRAPH_MARGIN = 1.1


def _tile_list():
    """``fastmace.neighbors.tile_list``, imported when a call needs it (it imports torch; this module stays importable
    without it, as the gate driver's subprocess needs)."""
    from fastmace.neighbors import tile_list

    return tile_list


def compiled_graphs_fit(workspace, available, slots):
    """Whether ``slots`` capture pools of a measured tile ``workspace`` (bytes), and the warm-ups' workspace beside
    them, fit ``available`` bytes (the guard of ``TileConfig.compiled_graphs``)."""
    return (int(slots) + 1) * COMPILED_GRAPH_MARGIN * float(workspace) <= float(available)
# TILE_PREFETCH_DEVICE (exact): TileConfig.prefetch also with the state on the device, where it builds the next tile's
# graph on the side stream while the current tile runs (its rows are read in the body, as before). Without it a
# device-state call built each graph after the previous body had drained the GPU (torch.unique and the edge counts
# wait for it): 1.2 s of 8.6 s idle at 975,560 SrTiO3 atoms on the RTX 5060 Ti.
TILE_PREFETCH_DEVICE = None
# TILE_VIRIAL (rounding): the strain gradient of sweeps 2 and 3 from the edge vectors and their gradient,
# sym(V^T dE/dV), summed elementwise, instead of a strain leaf in the autograd graph. The leaf's backward is a
# [3, E] x [E, 3] product that cuBLAS runs as a split-K GEMM (sgemm_largek, about 0.3 us per atom per call).
TILE_VIRIAL = None
# TILE_GRAPH_CACHE (exact): a call without the wavefront keeps each tile's graph on the device from its first build
# (the slot-key pass) to the end of the call, as its local edge index (int32, halo rows after the receivers) and edge
# shifts (20 bytes per edge), so sweeps 1-3 take it back with one gather instead of building it again from the list
# cache (copy, unpack, torch.unique; four builds per tile and call). The pieces take at most this fraction of the
# device memory free when the call starts (the allocator's unused reserve included; None: TILE_GRAPH_CACHE_FRACTION on
# TILED_SPEED_GPUS, else 0); tiles past it are built again as before. The same edges in the same order: bitwise the
# call without.
TILE_GRAPH_CACHE = None
TILE_GRAPH_CACHE_FRACTION = 0.25
# TILE_GRAPH_CACHE_WAVE (exact): the graph cache under the wavefront too, filled by its search pass (which builds every
# tile's graph), each tile's graph dropped after its sweep 3. The wavefront built every graph four times per call
# (the search pass and three sweeps), with a sync per build: 4.3 s of host time in a 13 s call at 975,560 SrTiO3 atoms
# on the RTX 5060 Ti, the GPU idle for 4.9 s. None: on for TILED_SPEED_GPUS.
TILE_GRAPH_CACHE_WAVE = None
# TILE_HOST_WORKER (exact): host state with overlap applies a tile's deferred writes (the event wait for the tile's
# device work, the put, index_add and close on the host) on one worker thread, in order, instead of on the calling
# thread at the next tile's first write; a read of the state (rows, fetch, flush) waits for every batch handed over
# before it, the same writes the calling thread had applied by then. In the wavefront's 13 s call at 975,560 SrTiO3
# atoms on the RTX 5060 Ti the calling thread spent about 3 s in these writes and their event waits, the GPU idle.
# None: on for TILED_SPEED_GPUS.
TILE_HOST_WORKER = None
# TILE_FILTER32 (exact): the kept lists' per-call cutoff test (list_reuse) in float32, with the float64 test again
# only for the edges whose float32 squared length lies within a bound of the cutoff's (``_filter32_band``: four times
# the float32 error for coordinates up to the call's largest), and the float64 shifts of the one-byte unit shifts
# from a 27-row table (``times_3x3`` per row: bitwise the per-edge product). The same edges, shifts and order as the
# float64 test. The RTX 5060 Ti runs float64 at 1/64 of float32: the test took about 0.2 us per atom per call.
TILE_FILTER32 = None
# TILE_EPILOGUE (rounding): the sweeps' eager work around the compiled forwards moves into compiled code (mode
# "compiled"; the same functions run eagerly in the other modes): sweep 1's padding masks and sweep 3's objective
# (scale and shift, the masks, <dE/dx, x> + <dE/dsc, sc>) join the sweeps' forwards; the edge-gradient masks and the
# virial (TILE_VIRIAL) of sweeps 2 and 3 are one compiled epilogue each, and a padded slot's edge vectors one kernel.
# None: on for TILED_SPEED_GPUS (wavefront: 1.012 / 1.009 host-bound, 0.985 / 0.983 since the host worker).
# Before, these were about 20 eager elementwise kernels and reductions per tile and sweep (about 0.25 us per atom per
# call at 975,560 SrTiO3 atoms). The gradients are the same products; the virial's and the objective's sums change
# their order.
TILE_EPILOGUE = None
# TILE_LOCAL_FRAME (accuracy; idea 11, results/tiled-speed-20261111 "Idea 11"): the tiles' edge vectors from positions
# held as an int16 bin of TILE_LOCAL_FRAME_WIDTH Angstrom plus a float32 offset within it, and each edge's shift
# rebuilt in float64 from its unit shift (recovered exactly from the float32 shift and the cell) and split the same
# way: v = (bin_r - bin_s + k) W + (off_r - off_s + rem). Plain float32 positions round each coordinate to its
# float32 spacing, so a perfect crystal's edge vectors carry 1.5e-5 A at 226 A and 1.1e-4 A from 1,024 A (the
# residual force 1.64e-4 -> 1.32e-3 eV/A from 1M to 149M atoms); in the frame they carry a few 1e-6 A in any
# box (6e-6 A worst case). Where the call's float64 positions are on the device, each tile builds its float32 rows
# relative to its first receiver instead (``_tile_local``; no stored bytes); the bins and offsets (18 bytes per atom
# beside the field's float32 positions) only for positions on the host. None (auto, the conductor's size rule of 11
# November): on where the structure's extent (the largest |coordinate| of the positions and of the cell's corners) is
# at least ``forces.FLOAT64_EDGES_EXTENT`` (64 A), decided once per atom count and cell; True / False: every size.
TILE_LOCAL_FRAME = None
TILE_LOCAL_FRAME_WIDTH = 32.0  # a power of two: bin multiples are exact in float32
# TILE_HOST_ROWS (exact; billion-atom plan step 1, 11 November 2026): where the call's float64 positions are on the
# host (``ForceField._prepare_one_off(host=True)``: 24 bytes per atom off the device), the tile-local frame stays the
# tiles' form: each padded slot gets its float64 rows gathered on the host and copied in (``_fill``; an eager tile
# gathers its own), the values ``positions64[nodes]`` reads on the device. False: the bins and offsets as before.
TILE_HOST_ROWS = True
# TILE_WHOLE_BLOCK_ORDER (A/B only; billion-atom step 1): True binds the whole-system BLOCK_ORDER receiver order in the
# tiled call as before (int32 [N], 4 B/atom; its per-tile kernels never read it). False (default): not built.
TILE_WHOLE_BLOCK_ORDER = False
# TILE_LARGE_CAPS (B3, 11 November 2026, results/tiled-caps-20261111): a host-state call whose budget_receivers were
# left at their default may also take receiver caps of TILE_LARGE_CAP_RECEIVERS where the memory budget fits them
# (``_choose_receivers``: the first call's measured workspace per receiver, times the cap, plus 10 %, within the
# memory left; at the capacity limit it keeps the smaller cap). A100 against R 16,384 (Alex 4984555), b/a both
# orders: 58^3 host state 0.886 / 0.893 at 32,768 and 0.549 / 0.549 at 65,536, wavefront 0.675 / 0.671 and 0.410 /
# 0.409; 160^3 host state 0.732 / 0.724 and 0.597 / 0.599, wavefront 0.722 / 0.723 and 0.573 / 0.583. None: on for
# TILE_LARGE_CAP_GPUS (decision 66, per family); True / False force it.
TILE_LARGE_CAPS = None
TILE_LARGE_CAP_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB",))
TILE_LARGE_CAP_RECEIVERS = (32768, 65536)
# The large caps' host cost (the size rule): host-state tiles hold per-tile host buffers (kept lists, staging) that grow
# with the cap. SrTiO3 160^3 on the A100 (Alex 4985398): host peak 24.3 / 28.0 / 31.9 / 43.3 GB at R 8,192 / 16,384 /
# 32,768 / 65,536, about 0.33 MB per receiver, while the device workspace fits at every size. ``_choose_receivers``
# takes a large cap only where the host keeps this much per added receiver (plus 10 %) above ``list_reuse_floor``:
# at the 397.5M record (229 of the cgroup's 231.5 GB) the cap stays where it was.
TILE_LARGE_CAP_HOST_BYTES = 350_000
# TILE_FIRST_CALL_PROBE (billion-atom plan, "Before step 4" 2; 11 November 2026): mode "auto"'s measuring first call
# runs its eager sweeps on the TILE_PROBE_TILES largest tiles only (the wavefront's order, the others left out), takes
# the mode and the receiver cap from that measurement as from a whole eager call, drops that call's buffers and kept
# lists, and runs the call again in the chosen mode. The eager tiles' workspace is set by the largest tile, not by the
# atom count: SrTiO3 at R 8,192 on the A100, 526,080,512 bytes above the tiles' start at 149M, 320M and 397.5M atoms
# (Alex 4973640), 503,742,464 at 20.48M with a smaller largest tile. The 397.5M record's eager first call took
# 11,468 s of 15,041. Wavefront calls only (else the whole eager call, as before). None: on for no GPU yet (A/B:
# the probe's decision against the whole call's); True / False force it (True also off a CUDA device, for tests).
TILE_FIRST_CALL_PROBE = None
TILE_PROBE_GPUS = GPUModels(())
TILE_PROBE_TILES = 8
# TILE_FILL_SKIP (exact): a tile whose graph the call's graph cache (TILE_GRAPH_CACHE) takes is not also packed into
# the host list cache (the copy back to the host and its sync), which only sweeps that build the graph again read. A
# tile built while the cache had room but too large for what was left is built again from the kept lists (the same
# edges). It only removes work: on everywhere (decision 66).
TILE_FILL_SKIP = True
# TILE_PINNED_SUPERSET (exact): the kept lists (list_reuse's superset within cutoff + skin) in pinned host memory while
# they total at most TILE_PINNED_SUPERSET_BYTES, so each call's copy to the device runs at the link's pinned rate and
# without a sync (pageable copies: about 70 ns per atom per call at 975,560 SrTiO3 atoms on the RTX 5060 Ti); past
# the cap, pageable as before. None: on for TILED_SPEED_GPUS (wavefront: 1.012 / 1.028 host-bound, 0.989 / 0.990
# since the host worker). The cap counts the pinned allocator's power-of-two blocks, and lists dropped or replaced
# return them to the system (``_release_kept``; the allocator would keep them resident, uncounted by the host budget).
TILE_PINNED_SUPERSET = None
TILE_PINNED_SUPERSET_BYTES = 4 * 2**30
# TILE_COPY_STREAM (exact): host state with overlap copies each tile's deferred writes (values and device indices) to
# pinned host memory on a side stream instead of the compute stream: the compute stream only clones them on the
# device (at the device's bandwidth; the slot's outputs and index buffers are overwritten by the next tile), and the
# next tile's kernels no longer queue behind the link. The host applies the same values in the same order after the
# copy's event. The wavefront at 975,560 SrTiO3 atoms on the RTX 5060 Ti had 0.44 s of device-to-host copies on the
# compute stream in a 7.7 s call. None: on for TILED_SPEED_GPUS.
TILE_COPY_STREAM = None
# TILE_HEAP_TRIM (memory only): kept lists or the list cache dropped hand glibc's free heap back to the system
# (``malloc_trim(0)``, ``_trim_heap``). The lists are many tensors of a few MB, which glibc serves from its heap once
# its mmap threshold has risen, and keeps resident when they are freed; the host budget (MemAvailable, the cgroup)
# counted them as used. At 20.1M SrTiO3 atoms on the RTX 5060 Ti a forced drop of 7.0 GiB of lists left 2.77 GiB in
# glibc's free heap, all returned by a trim; the A100's 397.5M record (lists all pageable) kept about 110 GiB and
# fell to float16 state (plans/ideas-20261111-tiled.md, idea 1).
TILE_HEAP_TRIM = True


@dataclasses.dataclass(frozen=True)
class TileConfig:
    """Tile caps of the prototype: at most ``receivers`` receiver atoms and ``edges`` edges per tile.

    ``edges=None`` caps the edges at ``receivers`` times twice the structure's mean degree, so a tile is split
    where its receivers are unusually crowded. ``mode``: ``"eager"`` (each tile at its own shape), ``"padded"``
    (every tile in one padded slot, eagerly), ``"graphs"`` (the padded slot, one CUDA graph per sweep) or
    ``"compiled"`` (the padded slot, each sweep's forward compiled by Inductor at the slot's shapes; the force
    derivative outside the compiled frame; needs a compiled force field, whose model is traceable).
    ``headroom``: a padded slot's halo count above the largest of its tiles', as a fraction. ``bucket``: the padded
    modes keep one slot per edge bucket, ``bucket`` times the largest tile's edges (rounded up to 256) wide, so a
    small tile runs in a small slot; 1 or more gives one slot for every tile.
    ``budget_receivers``: larger receiver caps the call may take after its first call (at ``receivers``; default
    (8192, 16384) with ``mode="compiled"``, none otherwise), from
    the memory model: the first call measures its workspace per receiver (the peak above what stays allocated
    after it: the tiled state, the field and the outputs), and the call then takes the largest candidate whose
    predicted workspace, with 10 % headroom, fits the memory left on the device. Larger tiles run faster (fewer
    launches, a smaller halo fraction; docs/validation/tiled/tiled-screen.md), and at the capacity limit the
    workspace does not fit, so the cap stays at ``receivers``. Chosen once; needs a CUDA device (elsewhere the
    cap stays).
    ``state``: where x, sc and their adjoints live, ``"device"`` or ``"host"`` (pageable host memory; each
    tile copies its rows to the device and its results back, so the device holds no per-atom state of the
    sweeps; not with ``mode="graphs"``, whose captured bodies read the state themselves).
    ``alias_state``: dE/dsc takes sc's rows (a receiver's sc is read and its dE/dsc
    written by the same tile) and dE/dx takes x's rows atom by atom (an atom's last tile in sweep 2 is the last to
    read its x and the last to add to its dE/dx; until then the sum lives in a window of the open atoms, added in
    the same order): bitwise the same call with two of the four state tensors. With ``state="device"`` (and the
    wavefront: its row pool on the device) the window's rows are planned on the host after sweep 1, so the tiles
    index the state without syncs; not in CUDA graphs, not with lanes.
    ``species``: the tiles' one-hot species rows from the field's ``node_attrs`` (``"one_hot"``) or built from
    its species index (``"index"``, bitwise the same rows): the field's device one-hot is then released to the
    host (``MACEForceField.release_node_attrs``; its own whole-system call refuses afterwards).
    """

    receivers: int = 8192
    edges: int | None = None
    mode: str = "auto"
    headroom: float = 0.1
    bucket: float = 0.15
    # "global": the force field's neighbour list; "tiles": each tile's edges from a cell binning of the current
    # positions (``fastmace.neighbors.tile_list``), bitwise the global list's, without it (T3).
    neighbours: str = "global"
    # neighbours "tiles": each tile's edges found once per call and kept on the host (int16 ranks into the tile's
    # int32 senders, receiver counts and one-byte unit shifts, about 3 bytes per edge), the later sweeps rebuilding
    # them exactly from it. At the host limit (below ``list_reuse_floor`` available while it fills) it is dropped,
    # for this call and the later ones: the later sweeps search again (bitwise), the reason in the call's report
    # (``list_cache``).
    list_cache: bool = True
    # neighbours "tiles": the tiles' edges searched within cutoff + skin (the force field's list's skin) and kept on
    # the host across calls with the tile plan; a call with the same cell while no atom has moved more than skin / 2
    # since that search filters them at the cutoff (bitwise a fresh search on the same tiles) instead of binning,
    # planning and searching again (MD). Kept only while the host keeps ``list_reuse_floor`` bytes available
    # (MemAvailable, Linux): below it the kept lists are dropped and reuse is off, the reason in the call's report.
    list_reuse: bool = True
    list_reuse_floor: int = 20 * 2**30
    budget_receivers: tuple[int, ...] | None = None
    state: str = "device"
    alias_state: bool = False
    # state "host" with alias_state: the dtype x, sc and their aliased adjoints are stored in on the host. None: the
    # call's dtype; "float16" or "bfloat16" halve the host state (capacity: with the kept lists, the largest host
    # item per atom under the wavefront). Each value is rounded once where it is written (on the device, before its
    # copy to the host, which halves that traffic too) and read back exactly into the call's dtype; the tiles compute
    # in the call's dtype and the alias window (dE/dx summed over an atom's tiles) stays in it. Neither format passes
    # the 3x float32 force/stress gate (float16 8-51x, bfloat16 50-750x; NVE drift with float16 as float32's), so
    # they are a capacity fallback (tiled-host-state-dtype.md). "auto" (the default; aliased host state only): the
    # call's dtype while the state fits above ``list_reuse_floor`` on the host; else the kept lists go first (exact,
    # about 14 % slower calls), and float16 only if the state still does not fit. The choice and its reason are in
    # the call's report (``state_dtype``).
    state_dtype: str | None = "auto"
    # state "host": a tile's writes to the host state are applied after the next tile's work is queued, so the
    # device runs it while the host indexes (on CUDA through pinned buffers and events); the same order.
    overlap: bool = True
    species: str = "one_hot"
    # ``"auto"``, ``"on"`` or ``"off"`` (with ``alias_state``): the wavefront schedule (docs/design/tiled-wavefront.md).
    # Sweep 2 visits the tiles in their order; sweep 1 runs on a tile just before the first sweep-2 tile that reads
    # its x, and sweep 3 just after the last sweep-2 tile that adds to its receivers' dE/dx. x, sc and their adjoints
    # then live in a pool of the rows of the atoms between their sweeps 1 and 3 instead of [N, width] tensors: per
    # atom bitwise today's state and node energies, the forces and stress summed in another order (float32 rounding;
    # the scatter is atomic on the GPU anyway). The search pass that plans it keeps no tile lists, so the tiles
    # search once more (about 3 % of a call). "auto": on from ``WAVEFRONT_LAYERS`` layers of the tile grid (from
    # ``WAVEFRONT_MIN_LAYERS`` when the plain state would not fit the host at all) when the
    # plain state would leave the host below ``list_reuse_floor``.
    wavefront: str = "auto"
    # ``"auto"``, ``"on"`` or ``"off"``: sweep 2's tiles in pairs on two lanes, CUDA streams of two green contexts
    # holding disjoint halves of the SMs (padded and compiled tiles; sweeps 1 and 3 keep the whole device). The second
    # lane has its own padded slots and strain gradient; the node energies, forces and dE/dx are the same sums (the
    # GPU's scatter is atomic, the host writes keep their tile order). Green-context streams need PyTorch's
    # ``cudaMallocAsync`` allocator backend (``PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`` before CUDA starts;
    # with expandable segments their first tile fails to map memory). "auto": on with that backend on the GPUs of
    # ``LANE_GPUS`` when the first call's memory model fits a second tile workspace; "on" refuses elsewhere (off a
    # CUDA device the lanes run one after the other: the CPU tests' stand-in).
    lanes: str = "auto"
    # The wavefront's sweeps by groups of this many sweep-2 tiles: sweep 1 on every tile the group reads, then the
    # group's sweep 2, then sweep 3 on every tile it closes. One tile at a time alternates the sweeps at every tile,
    # and every change of sweep applies the pending host writes, which waits for the device (two drains per tile);
    # a group pays them once and keeps the rows' prefetch within its runs. The pool grows by about two groups' tiles.
    wavefront_group: int = 16
    # The chunked streaming backward's edges per chunk in the tiles, times this (``None``: ``TILE_CHUNK_SCALE`` by
    # device name, else 1). The chunks only change which edges' weight gradients share an L2-sized buffer.
    backward_chunk_scale: int | None = None
    # The plan's grid from ``fit_grid`` (the fewest receiver-cap pieces among the rule's grid and finer ones), chosen
    # once per receiver cap, atom count and cell; False: the rule's grid alone (``_grid_shape``).
    grid_fit: bool = False
    # Padded tiles with the state on the host (CUDA): the next tile's graph is built, and the host rows its sweep
    # reads gathered and copied, on a side stream while the current tile runs, so the host's syncs and gathers no
    # longer drain the device's queue at every tile (a profiled 5M call left the GPU idle 17 % of the time).
    # Bitwise the same call: the same graphs and rows, read where the sweep would read them. Rows are fetched ahead
    # only within a run of one sweep (2 or 3), where no host write reaches a row the next tile reads.
    prefetch: bool = True
    # With the wavefront: the tile grid's first and last layer along its slowest axis each split into a skin of one
    # reach (cutoff + the kept lists' skin) at the periodic boundary and the rest. Only the skins are read across the
    # boundary, so only they stay in the row pool for the whole call, not two whole layers (a block model of 149M
    # SrTiO3 at R 8,192: the pool from 16 % to 9 % of the atoms; results/tiled-ideas-20261102/band_model.py). Another
    # tile plan: the forces and stress are the same sums in another order (float32 rounding).
    boundary_skin: bool = True
    # Compiled tiles (mode "compiled", or "auto" once it compiles): each slot's compiled sweep bodies captured in a
    # CUDA graph (one per slot and sweep, after two warm-ups) and replayed per tile, as mode "graphs" does with the
    # eager bodies. Device state only (the captured bodies read it), no lanes. The first compiled call runs
    # uncaptured and measures its workspace; the captures follow only where one capture pool per slot fits beside it
    # (``compiled_graphs_fit``; the 16 GB A4000 ran out of memory at 975,560 atoms, R 8,192, with graphs on from the
    # first call), and a captured call that runs out of memory runs again without graphs. None: on for
    # ``COMPILED_GRAPH_GPUS`` where the configuration allows it (mode compiled or auto, device state, no lanes="on",
    # no aliased device state), else off; True refuses those configurations.
    compiled_graphs: bool | None = None

    def __post_init__(self):
        if self.mode not in ("auto", "eager", "padded", "graphs", "compiled"):
            raise ValueError("TileConfig.mode must be 'auto', 'eager', 'padded', 'graphs' or 'compiled', "
                             f"got {self.mode!r}")
        if not 0 <= float(self.headroom) <= 10:
            raise ValueError(f"TileConfig.headroom must be in [0, 10], got {self.headroom!r}")
        if self.neighbours not in ("global", "tiles"):
            raise ValueError(f"TileConfig.neighbours must be 'global' or 'tiles', got {self.neighbours!r}")
        if not 0 < float(self.bucket) <= 10:
            raise ValueError(f"TileConfig.bucket must be in (0, 10], got {self.bucket!r}")
        if int(self.receivers) < 1:
            raise ValueError(f"TileConfig.receivers must be positive, got {self.receivers!r}")
        if self.species not in ("one_hot", "index"):
            raise ValueError(f"TileConfig.species must be 'one_hot' or 'index', got {self.species!r}")
        if self.alias_state and self.state == "device" and (self.mode == "graphs" or self.compiled_graphs is True):
            raise ValueError("TileConfig.alias_state with state='device' cannot run in CUDA graphs: the window and the "
                             "pooled rows are indexed outside the captured bodies")
        if self.alias_state and self.state == "device" and self.lanes == "on":
            raise ValueError("TileConfig.alias_state with state='device' cannot run lanes='on': a tile's closing "
                             "atoms read the window on the call's stream")
        if self.state_dtype not in (None, "auto", "bfloat16", "float16"):
            raise ValueError(f"TileConfig.state_dtype must be None, 'auto', 'bfloat16' or 'float16', got "
                             f"{self.state_dtype!r}")
        if self.state_dtype not in (None, "auto") and not (self.alias_state and self.state == "host"):
            raise ValueError("TileConfig.state_dtype needs alias_state=True (state='host'): every stored value is then "
                             "written once, never summed in the stored dtype")
        if self.state not in ("device", "host"):
            raise ValueError(f"TileConfig.state must be 'device' or 'host', got {self.state!r}")
        if self.state == "host" and self.mode == "graphs":
            raise ValueError("TileConfig.state='host' cannot run in mode 'graphs': the captured bodies read the "
                             "state from device memory")
        if self.budget_receivers is None:  # compiled tiles take the larger caps when they fit (tiled-speed.md)
            compiled = self.mode in ("compiled", "auto")  # "auto": only once its tiles compile
            object.__setattr__(self, "budget_receivers", (8192, 16384) if compiled else ())
            # (not a field: TiledForceField adds TILE_LARGE_CAP_RECEIVERS to a default budget only)
            object.__setattr__(self, "_budget_default", compiled)
        if any(int(r) < 1 for r in self.budget_receivers):
            raise ValueError(f"TileConfig.budget_receivers must be positive, got {self.budget_receivers!r}")
        if self.edges is not None and int(self.edges) < 1:
            raise ValueError(f"TileConfig.edges must be positive or None, got {self.edges!r}")
        if self.wavefront not in ("auto", "on", "off"):
            raise ValueError(f"TileConfig.wavefront must be 'auto', 'on' or 'off', got {self.wavefront!r}")
        if self.wavefront == "on" and not self.alias_state:
            raise ValueError("TileConfig.wavefront='on' needs alias_state=True")
        if self.backward_chunk_scale is not None and int(self.backward_chunk_scale) < 1:
            raise ValueError(f"TileConfig.backward_chunk_scale must be positive, got {self.backward_chunk_scale!r}")
        if int(self.wavefront_group) < 1:
            raise ValueError(f"TileConfig.wavefront_group must be positive, got {self.wavefront_group!r}")
        if self.lanes not in ("auto", "on", "off"):
            raise ValueError(f"TileConfig.lanes must be 'auto', 'on' or 'off', got {self.lanes!r}")
        if self.lanes == "on" and self.mode in ("eager", "graphs"):
            raise ValueError(f"TileConfig.lanes='on' needs padded or compiled tiles, not mode {self.mode!r}")
        if self.compiled_graphs not in (None, True, False):
            raise ValueError(f"TileConfig.compiled_graphs must be None, True or False, got {self.compiled_graphs!r}")
        if self.compiled_graphs is True:
            if self.mode not in ("compiled", "auto"):
                raise ValueError(f"TileConfig.compiled_graphs needs mode 'compiled' or 'auto', not {self.mode!r}")
            if self.state == "host":
                raise ValueError("TileConfig.compiled_graphs cannot run with state='host': the captured bodies read "
                                 "the state from device memory")
            if self.lanes == "on":
                raise ValueError("TileConfig.compiled_graphs cannot run with lanes='on'")


class _State:
    """x, sc and their adjoints [N, width] (sweep 1 allocates them from its first outputs), on the device or in
    host memory (``TileConfig.state``). Tiles read rows to the device and put or add rows; on the device these
    are the plain indexing and ``index_add_``, in host memory the same on the host after one copy."""

    def __init__(self, device, dtype, host, alias=False, overlap=False, store=None):
        self.device, self.dtype, self.host, self.alias = device, dtype, host, alias
        # TileConfig.state_dtype: the host tensors' dtype (the alias window stays in ``dtype``)
        self.store = dtype if store is None else store
        self.overlap = host and overlap
        self._pending, self._fresh = [], False  # overlap: the host writes not applied yet
        # TILE_HOST_WORKER: the batches of writes handed to the worker thread (set per call by the tiled call)
        self.worker, self._executor, self._handed = False, None, []
        # TILE_COPY_STREAM: the deferred writes' device-to-host copies on this side stream (set per call)
        self.copy_stream = None
        self.tensors = {}
        # alias: dE/dsc in sc's rows, dE/dx in x's rows once an atom is closed (``touch``, ``plan``, ``close``)
        self._storage = {"gx": "x", "gsc": "sc"} if alias else {}
        # wavefront (``pool``): x and sc (with their aliased adjoints) in pooled rows, ``_row_of`` each atom's
        self.pooled, self._row_of = False, None
        self.prefetched = {}  # TileConfig.prefetch: the running tile's rows, fetched ahead (name -> device rows)

    def pool(self, rows, n):
        """The wavefront schedule: x, sc and their aliased adjoints in ``rows`` pooled rows instead of ``n``; an atom
        takes a row before its sweep 1 (``assign``) and returns it after its sweep 3 (``release``). The pool grows,
        never shrinks, between calls; ``rows`` 0 switches back to [N, width]."""
        import torch

        self.settle()
        pooled = rows > 0
        size = next(iter(self.tensors.values())).shape[0] if self.tensors else None
        if size is not None and (pooled != self.pooled or (pooled and size < rows) or (not pooled and size != n)):
            self.tensors = {}  # allocated again at the right size from sweep 1's first outputs
        self.pooled = pooled
        if pooled:  # the row map and free rows beside the state (the device's: no host round trip per tile)
            where = "cpu" if self.host else self.device
            self._pool_rows = max(rows, size or 0) if self.tensors else rows
            self._row_of = torch.full((n,), -1, dtype=torch.int32, device=where)
            self._free_rows = torch.arange(self._pool_rows, dtype=torch.int32, device=where)
            self._rows_top = self._pool_rows
        else:
            self._row_of = None

    def assign(self, atoms):
        """Pooled rows for ``atoms`` (none of which has one)."""
        atoms = self._here(atoms)
        top = self._rows_top - atoms.shape[0]
        if top < 0:
            raise RuntimeError("the wavefront's row pool is exhausted (its schedule's band was underestimated)")
        self._row_of[atoms] = self._free_rows[top:self._rows_top]
        self._rows_top = top

    def release(self, atoms):
        """Return ``atoms``' pooled rows (after their sweep 3 has read them)."""
        atoms = self._here(atoms)
        rows = self._row_of[atoms]
        self._free_rows[self._rows_top:self._rows_top + rows.shape[0]] = rows
        self._rows_top += rows.shape[0]
        self._row_of[atoms] = -1

    def _rows_of(self, name, index, read=False):
        """The rows of ``index``'s atoms in the storage of ``name``: the atoms themselves, or pooled rows (x and sc's
        storage). A read maps an atom without a row (a padded slot's padding rows, atom 0) to row 0: padding rows
        reach no real output (tiled tests: padded equals eager bitwise), and a pooled row holds finite values."""
        if not self.pooled or self._storage.get(name, name) not in ("x", "sc"):
            return index
        rows = self._row_of[self._here(index)].long()
        return rows.clamp_(min=0) if read else rows

    def _here(self, index):
        """``index`` where the state is (an atom index from the host or the device)."""
        return index.cpu() if self.host else index.to(self.device, non_blocking=True)

    def touched(self, first, last):
        """The wavefront's search pass recorded each atom's first and last tile (``touch`` of sweep 1 without it)."""
        import torch

        self._first, self._last = first, last
        self._slot_of = torch.full_like(first, -1)
        self._window = None

    def allocate(self, n, widths):
        import torch

        if self.tensors:
            return
        # Host memory unpinned: the tiles copy through gathered (pageable) rows, never from these tensors, and
        # PyTorch's pinned allocator rounds every block up to a power of two, all of it resident (72 GiB for
        # 42 GiB of state at 9.26M NbCl5 atoms).
        where = "cpu" if self.host else self.device
        size = self._pool_rows if self.pooled else n
        for name, width in widths.items():
            if name not in self._storage:
                self.tensors[name] = torch.zeros((size, width), dtype=self.store if self.host else self.dtype,
                                                 device=where)
        if self.alias:
            self._width = widths["gx"]
            self._index(n)

    def _index(self, n):
        """alias: each atom's first and last tile and its row in the window while open (sweep 1 records the
        tiles before its first outputs allocate the state)."""
        import torch

        if getattr(self, "_first", None) is None:
            self._first = torch.full((n,), -1, dtype=torch.int32)
            self._last = torch.full((n,), -1, dtype=torch.int32)
            if getattr(self, "_slot_of", None) is None or self._slot_of.shape[0] != n:
                self._slot_of = torch.full((n,), -1, dtype=torch.int32)  # int32: 4 bytes per atom on the host

    def zero(self):
        self.settle()
        for tensor in self.tensors.values():
            tensor.zero_()
        if self.alias and getattr(self, "_slot_of", None) is not None:
            if self._first is not None:
                self._first.fill_(-1)
                self._last.fill_(-1)
            self._slot_of.fill_(-1)
            self._window = None

    def bytes(self):
        tensors = list(self.tensors.values())
        if self.pooled:
            tensors += [self._row_of, self._free_rows]
        if self.alias and getattr(self, "_slot_of", None) is not None:
            tensors += [t for t in (self._first, self._last, self._slot_of) if t is not None]
            tensors += [self._window] if getattr(self, "_window", None) is not None else []
        return int(sum(t.numel() * t.element_size() for t in tensors))

    def rows(self, name, index):
        """``state[name][index]`` on the device. A tile reads no row that a pending write of its sweep touches
        (sweep 2 reads sc and x of sweep 1, and its writes go to its own receivers' sc rows and to closed atoms'
        x rows), and every sweep ends with the writes applied."""
        import torch

        fetched = self.prefetched.pop(name, None)
        if fetched is not None and fetched.shape[0] == index.shape[0]:
            if fetched.is_cuda:  # made on the prefetch stream
                fetched.record_stream(torch.cuda.current_stream(fetched.device))
            return fetched
        tensor = self.tensors[self._storage.get(name, name)]
        if not self.host:
            return tensor[self._rows_of(name, index, read=True)]
        self.settle()
        index = self._rows_of(name, index.cpu(), read=True)
        if not (self.overlap and self.device.type == "cuda"):
            return tensor.index_select(0, index).to(self.device).to(self.dtype)
        staging = torch.empty((index.shape[0], tensor.shape[1]), dtype=tensor.dtype, pin_memory=True)
        torch.index_select(tensor, 0, index, out=staging)
        return staging.to(self.device, non_blocking=True).to(self.dtype)

    def fetch(self, name, index):
        """``rows(name, index)`` for a host ``index``, copied on the current stream without waiting (the prefetch
        stream's: its event orders the read)."""
        import torch

        self.settle()
        tensor = self.tensors[self._storage.get(name, name)]
        index = self._rows_of(name, index, read=True)
        if self.device.type != "cuda":
            return tensor.index_select(0, index).to(self.device).to(self.dtype)
        staging = torch.empty((index.shape[0], tensor.shape[1]), dtype=tensor.dtype, pin_memory=True)
        torch.index_select(tensor, 0, index, out=staging)
        return staging.to(self.device, non_blocking=True).to(self.dtype)

    def put(self, name, index, values):
        """``state[name][index] = values``."""
        if not self.host:
            self.tensors[self._storage.get(name, name)][self._rows_of(name, index)] = values
        else:
            self._defer("put", name, self._rows_of(name, index), values.detach().to(self.store))

    def add(self, name, index, values):
        """``state[name].index_add_(0, index, values)``; aliased dE/dx: into the open atoms' window rows."""
        if not self.host and self.alias and name == "gx":  # the window rows planned for the call (``plan``)
            self._window.index_add_(0, self._slot_of[self._here(index)], values)
        elif not self.host:
            self.tensors[self._storage.get(name, name)].index_add_(0, self._rows_of(name, index), values)
        else:
            stored = name != "gx" or not self.alias  # aliased dE/dx: summed in the window, in the call's dtype
            self._defer("add", name, self._rows_of(name, index) if name != "gx" else index,
                        values.detach().to(self.store) if stored else values)

    def new_tile(self):
        """The call starts a tile: its first host write applies the earlier tiles' pending ones, by then behind
        the new tile's queued work."""
        self._fresh = True

    def flush(self):
        """Apply the pending host writes, in order (the end of every sweep)."""
        self.settle()
        self._apply(self._pending)
        self._pending.clear()

    def _apply(self, pending):
        for kind, name, index, values, event in pending:
            if event is not None:
                event.synchronize()
            if kind == "close":
                self._close_now(name)
            elif kind == "put":
                self.tensors[self._storage.get(name, name)][index] = values
            else:
                self._add_now(name, index, values)

    def _hand_over(self):
        """``TILE_HOST_WORKER``: the pending writes to the worker thread, applied there in order."""
        import concurrent.futures

        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="fastmace-tile-state")
        self._handed.append(self._executor.submit(self._apply, self._pending))
        self._pending = []

    def settle(self):
        """Wait for the writes handed to the worker thread (``TILE_HOST_WORKER``); its errors raise here."""
        handed, self._handed = self._handed, []
        for future in handed:
            future.result()

    def _defer(self, kind, name, index, values):
        import torch

        if not self.overlap:
            index, values = index.cpu(), values.detach().cpu()
            if kind == "put":
                self.tensors[self._storage.get(name, name)][index] = values
            else:
                self._add_now(name, index, values)
            return
        if self._fresh:
            if self.worker:  # the earlier tiles' writes on the worker thread; the next read waits for them
                self._hand_over()
            else:
                self.flush()
            self._fresh = False
        event = None
        if self.device.type == "cuda" and self.copy_stream is not None:
            index, values, event = self._to_host_aside(index, values.detach())
        elif self.device.type == "cuda":
            index, values = self._to_host(index), self._to_host(values.detach())
            event = torch.cuda.Event()
            event.record()
        else:  # copies: a padded slot's buffers hold the next tile before these writes are applied
            index, values = index.to("cpu", copy=True), values.detach().to("cpu", copy=True)
        self._pending.append((kind, name, index, values, event))

    def _to_host_aside(self, index, values):
        """``TILE_COPY_STREAM``: pinned host copies of ``index`` (device or host) and ``values`` made on the copy
        stream from device clones (the slot's buffers hold the next tile before the copy runs): ``(index, values,
        event)``, the event recorded after the copies."""
        import torch

        current = torch.cuda.current_stream(self.device)
        clones = [t.clone() for t in (index, values)]  # a host index too, as _to_host copies it
        stream = self.copy_stream
        stream.wait_stream(current)
        out = []
        with torch.cuda.stream(stream):
            for tensor in clones:
                if tensor.is_cuda:
                    tensor.record_stream(stream)
                    tensor = self._to_host(tensor)
                out.append(tensor)
            event = torch.cuda.Event()
            event.record(stream)
        return out[0], out[1], event

    @staticmethod
    def _to_host(tensor):
        """A pinned host copy of a device tensor, copied without waiting."""
        import torch

        out = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        out.copy_(tensor, non_blocking=True)
        return out

    def _add_now(self, name, index, values):
        if self.alias and name == "gx":
            slots = self._slot_of[index]
            opening = index[slots < 0]
            if opening.numel():
                top = self._top - opening.numel()
                new = self._free[top:self._top]
                self._top = top
                self._slot_of[opening] = new
                self._window[new] = 0
            self._window.index_add_(0, self._slot_of[index], values)
            return
        self.tensors[self._storage.get(name, name)].index_add_(0, index, values)

    def touch(self, index, tile, n):
        """Sweep 1: tile ``tile`` holds the atoms ``index`` (receivers and halo) of ``n``."""
        self._index(n)
        index = index.cpu()
        fresh = index[self._first[index] < 0]
        self._first[fresh] = tile
        self._last[index] = tile

    def plan(self, tiles):
        """After sweep 1: the atoms each tile closes, and a window of the most atoms open at once."""
        import torch

        self.settle()

        last = self._last.long()
        order = torch.sort(last, stable=True).indices.to(torch.int32)  # kept until the call ends: int32
        self._closing = torch.split(order, torch.bincount(last, minlength=tiles).tolist())
        opened = torch.bincount(self._first.long(), minlength=tiles)
        closed = torch.bincount(last, minlength=tiles)
        # During tile t: the atoms open after it plus those it closes.
        size = int((torch.cumsum(opened - closed, 0) + closed).max().item())
        del last, order, opened, closed
        if not self.host:  # the state on the device: every atom's window row planned here, read without syncs
            self._plan_slots(tiles, size)
        # Read no more this call (8 bytes per atom on the host); the next call's touches start new ones.
        self._first = self._last = None
        self._window = torch.zeros((size, self._width), dtype=self.dtype, device="cpu" if self.host else self.device)
        self._free = torch.arange(size, dtype=torch.int32)
        self._top = size

    def _plan_slots(self, tiles, size):
        """The aliased state on the device: each atom's window row, taken at its first tile of sweep 2 and returned
        after its last (sweep 2 visits the tiles in their order, as ``plan``'s window size assumes), planned on the
        host once; the rows and the closing atoms then on the device. (The host state takes them as writes arrive.)"""
        import torch

        first = self._first.long()
        opening = torch.split(torch.sort(first, stable=True).indices.to(torch.int32),
                              torch.bincount(first, minlength=tiles).tolist())
        slots = torch.full((first.shape[0],), -1, dtype=torch.int32)
        free, top = torch.arange(size, dtype=torch.int32), size
        for tile in range(tiles):
            atoms = opening[tile]
            if atoms.numel():
                slots[atoms] = free[top - atoms.numel():top]
                top -= atoms.numel()
            atoms = self._closing[tile]
            if atoms.numel():
                free[top:top + atoms.numel()] = slots[atoms]
                top += atoms.numel()
        del first, opening, free
        counts = [int(atoms.numel()) for atoms in self._closing]
        self._slot_of = slots.to(self.device)
        self._closing = torch.split(torch.cat(list(self._closing)).to(self.device), counts)

    def close(self, tile):
        """After tile ``tile`` of sweep 2: its closing atoms' dE/dx is final; it takes their rows of x (after the
        tile's own pending writes, with overlap)."""
        if self.overlap:
            self._pending.append(("close", tile, None, None, None))
        else:
            self._close_now(tile)

    def _close_now(self, tile):
        atoms = self._closing[tile]
        if atoms.numel() and not self.host:  # the planned rows: zeroed for the atoms that take them next
            slots = self._slot_of[atoms]
            self.tensors["x"][self._rows_of("x", atoms)] = self._window[slots]
            self._window[slots] = 0
        elif atoms.numel():
            slots = self._slot_of[atoms]
            self.tensors["x"][self._rows_of("x", atoms)] = self._window[slots].to(self.store)
            self._free[self._top:self._top + slots.numel()] = slots
            self._top += slots.numel()
            self._slot_of[atoms] = -1


class _Ghosts:
    """``lammps_class`` of the tile calls: not None, so the interactions cut their outputs to the receivers
    (``truncate_ghosts``); with ``first_layer=True`` their ``handle_lammps`` exchanges nothing."""


_GHOSTS = _Ghosts()


@functools.cache
def _holder_class():
    import torch

    class Holder(torch.nn.Module):
        """Stand-in for a module of interaction 1 in sweep 2: returns its ``value`` buffer, whatever its input
        (a buffer, so a compiled tile body takes it as an input)."""

        def __init__(self):
            super().__init__()
            self.register_buffer("value", None, persistent=False)

        def forward(self, *args, **kwargs):
            return self.value

    return Holder


def _grid_shape(cell, n, receivers, reach=0.0, min_layers=4, k=None, scale=1.0):
    """The tile grid of ``plan_tiles``: blocks per axis from the cell's plane spacings (about cubic blocks of about
    ``receivers`` atoms; ``scale`` times as many per axis; ``k`` given: those), and the fractional thickness of the
    boundary skin along the first axis (0: no split). The first axis splits when ``reach`` > 0, it has at least
    ``min_layers`` layers and a skin is at most half a layer."""
    import torch

    cell64 = torch.as_tensor(cell, dtype=torch.float64)
    volume = abs(float(torch.linalg.det(cell64)))
    recip = torch.linalg.inv(cell64).T
    spacings = [1.0 / float(torch.linalg.norm(recip[i])) for i in range(3)]  # plane spacings, Angstrom
    side = (volume * min(receivers, n) / n) ** (1 / 3)
    if k is None:
        k = [max(1, round(scale * s / side)) for s in spacings]
    k = [int(x) for x in k]
    skin = float(reach) / spacings[0]  # fractional: a plane's distance bounds every distance across it
    return k, (skin if reach > 0 and k[0] >= min_layers and skin * k[0] <= 0.5 else 0.0)


# fit_grid: the finer grids it tries, as factors on the blocks per axis of _grid_shape's rule.
GRID_FIT_SCALES = (1.0, 1.04, 1.08, 1.12, 1.17, 1.22, 1.28)


def fit_grid(positions, cell, receivers, reach=0.0, min_layers=4):
    """The blocks per axis for ``plan_tiles(shape=...)``: of ``_grid_shape``'s rule and finer grids
    (``GRID_FIT_SCALES``), the one whose blocks make the fewest pieces of at most ``receivers`` atoms (the coarser
    one on a tie). The rule aims at blocks of ``receivers`` atoms on average, so on a crystal whose blocks are a whole
    number of unit cells, rounding can leave most blocks just above the cap: each then splits into a full tile and a
    small remainder with a large halo (SrTiO3 at 20M atoms, R 8,192: 2,535 blocks, 4,412 tiles;
    docs/validation/tiled-scale.md). A finer grid whose blocks all fit has fewer tiles."""
    import torch

    n = positions.shape[0]
    cell64 = torch.as_tensor(cell, dtype=torch.float64, device=positions.device)
    best, seen = None, set()
    for scale in GRID_FIT_SCALES:
        k, skin = _grid_shape(cell64.cpu(), n, receivers, reach, min_layers, scale=scale)
        if tuple(k) in seen:
            continue
        seen.add(tuple(k))
        counts, _ = _blocks(positions, cell64, k, skin, counts_only=True)
        pieces = sum(-(-int(c) // int(receivers)) for c in counts)
        if best is None or pieces < best[0]:
            best = (pieces, k)
    return best[1]


def _blocks(positions, cell64, k, skin, counts_only=False):
    """``(counts, block)``: the atoms per block of the ``k`` grid with a boundary skin of ``skin`` along the first
    axis (``plan_tiles``) and each atom's block (None with ``counts_only``).

    The fractional coordinates and the block index are computed in chunks, element-wise (as the tile lists' bins: no
    float64 GEMM, whose emulated forms round coarsely), and the block index in int32 below 2**31 blocks: no
    whole-system float64 transient, 4 bytes per atom of keys (tiled capacity, lever 5; tiled-call-memory.md)."""
    import torch

    from fastmace.neighbors.skin import times_3x3
    from fastmace.neighbors.tile_list import CHUNK

    n = positions.shape[0]
    inverse = torch.linalg.inv(cell64)
    scale = torch.tensor(k, dtype=torch.float64, device=positions.device)
    top = torch.tensor([x - 1 for x in k], device=positions.device)
    layers = k[0] + 2 if skin else k[0]
    blocks = layers * k[1] * k[2]
    dtype = torch.int32 if blocks < 2**31 else torch.int64
    block = None if counts_only else torch.empty(n, dtype=dtype, device=positions.device)
    counts = torch.zeros(blocks, dtype=torch.int64, device=positions.device)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        frac = times_3x3(positions[start:stop].to(torch.float64), inverse)
        torch.remainder(frac, 1.0, out=frac)
        bins = torch.minimum(torch.clamp((frac * scale).long(), min=0), top)
        if skin:  # layer 0 the skin below ``skin``, the last one the skin from 1 - ``skin``, the others one up
            first = frac[:, 0]
            bins[:, 0] = torch.where(first < skin, 0, torch.where(first >= 1 - skin, layers - 1, bins[:, 0] + 1))
        index = (bins[:, 0] * k[1] + bins[:, 1]) * k[2] + bins[:, 2]
        counts += torch.bincount(index, minlength=blocks)
        if block is not None:
            block[start:stop] = index
        del bins, frac, index
    return counts.cpu().tolist(), block


def plan_tiles(positions, cell, row_ptr, receivers, edges, reach=0.0, min_layers=4, shape=None):
    """Receiver tiles: a list of sorted int64 device index tensors covering every atom once.

    Atoms are binned on a k1 x k2 x k3 grid of fractional coordinates, with k_i from the cell's plane spacings so
    that a block is about cubic and holds about ``receivers`` atoms (``shape``: the k_i, as ``fit_grid`` chooses
    them); a block above either cap is split into consecutive pieces (in index order) below both. A rule on the cell
    geometry and the caps only. ``row_ptr`` None: no degrees (the tiles build their own lists), so a block splits
    into consecutive pieces of ``receivers``.

    ``reach`` > 0 (``TileConfig.boundary_skin``): the first and the last layer along the first axis each split into
    a skin of ``reach`` at the periodic boundary and the rest (``_grid_shape``), so k1 + 2 layers; only the skins'
    atoms are within ``reach`` of an atom across the boundary.

    The block index is computed in chunks (``_blocks``), so the plan's whole-system transients stay small
    (tiled-call-memory.md); the values are the unchunked ones.
    """
    import torch

    n = positions.shape[0]
    cell64 = torch.as_tensor(cell, dtype=torch.float64, device=positions.device)
    k, skin = _grid_shape(cell64.cpu(), n, receivers, reach, min_layers, k=shape)
    layers = k[0] + 2 if skin else k[0]
    counts, block = _blocks(positions, cell64, k, skin)
    order = torch.sort(block, stable=True).indices  # within a block: increasing atom index
    del block
    if row_ptr is None:  # no degrees: consecutive pieces of ``receivers`` in every block
        tiles, start = [], 0
        for count in counts:
            tiles.extend(order[piece:min(piece + receivers, start + count)]
                         for piece in range(start, start + count, receivers))
            start += count
        return tiles, (layers, k[1], k[2])
    degree = (row_ptr[1:] - row_ptr[:-1])[order].cpu()
    cumulative = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(degree, 0)]).numpy()
    tiles = []
    start = 0
    for count in counts:
        tiles.extend(order[a:b] for a, b in _split_block(cumulative, start, start + count, receivers, edges))
        start += count
    return tiles, (layers, k[1], k[2])


def _split_block(cumulative, start, stop, receivers, edges):
    """The pieces ``(a, b)`` of the atoms ``start:stop`` (in plan order) below both caps, ``cumulative[i]`` the
    edges of the atoms before ``i``. A piece ends before the first atom that would take it to ``receivers`` atoms or
    past ``edges`` edges, and holds at least one atom: the edge cap cuts at ``j - 1`` for the first ``j`` with
    ``cumulative[j] > cumulative[a] + edges``. One search per piece, not a loop per atom."""
    import numpy as np

    pieces, piece = [], start
    while piece < stop:
        over = int(np.searchsorted(cumulative, cumulative[piece] + edges, side="right"))
        cut = min(piece + receivers, max(over - 1, piece + 1), stop)
        pieces.append((piece, cut))
        piece = cut
    return pieces


def wavefront_schedule(needs, ready, counts, group=1):
    """The wavefront's order of the tile sweeps: ``(steps, peak)``, ``steps`` a list of ``(sweep, tile)`` and
    ``peak`` the most receivers between their sweeps 1 and 3 at once (the row pool's size).

    Sweep 2 visits the tiles in order, ``group`` at a time. Before a group, sweep 1 runs on every tile of its tiles'
    ``needs`` (the tiles that own a receiver or halo atom of ``t``, ascending) it has not run on; after it, sweep 3
    runs on every tile whose ``ready`` (the last sweep-2 tile that holds one of its receivers) is in or before the
    group, in the order of ``ready`` then index. A tile's sweep 1 thus precedes every sweep-2 read of its x and sc,
    and its sweep 3 follows the last addition to its receivers' dE/dx and its own dE/dsc."""
    tiles = len(counts)
    order = sorted(range(tiles), key=lambda u: (int(ready[u]), u))
    done = [False] * tiles
    steps, live, peak, next3 = [], 0, 0, 0
    for start in range(0, tiles, max(1, int(group))):
        stop = min(tiles, start + max(1, int(group)))
        for t in range(start, stop):
            for u in needs[t]:
                u = int(u)
                if not done[u]:
                    done[u] = True
                    steps.append((1, u))
                    live += int(counts[u])
                    peak = max(peak, live)
        steps.extend((2, t) for t in range(start, stop))
        while next3 < tiles and int(ready[order[next3]]) < stop:
            u = order[next3]
            next3 += 1
            steps.append((3, u))
            live -= int(counts[u])
    return steps, peak


def lane_steps(steps, counts):
    """``steps`` (a list of ``(sweep, tile)``) with sweep 2's tiles in pairs for two lanes: ``(items, peak)``, an item
    ``(sweep, tile)`` or ``(2, tile, other)``, and ``peak`` the most receivers between their sweeps 1 and 3 at once.

    Two consecutive sweep-2 tiles become a pair. A sweep-1 step between them runs before the pair (it reads nothing),
    a sweep-3 step between them after it (it follows the first tile's sweep 2 and no later sweep-2 tile adds to its
    receivers). Without the wavefront every sweep runs over all tiles, so its sweep-2 tiles simply pair up."""
    items, held, pending = [], [], None
    for sweep, index in steps:
        if sweep == 2:
            if pending is None:
                pending = index
            else:
                items.append((2, pending, index))
                items.extend(held)
                held, pending = [], None
        elif sweep == 3 and pending is not None:
            held.append((sweep, index))
        else:
            items.append((sweep, index))
    if pending is not None:
        items.append((2, pending))
    items.extend(held)
    live = peak = 0
    for item in items:
        if item[0] == 1:
            live += int(counts[item[1]])
            peak = max(peak, live)
        elif item[0] == 3:
            live -= int(counts[item[1]])
    return items, peak


def _virial(vectors, gradient, mask=None):
    """``TILE_VIRIAL``: dE/d(strain) [3, 3] of edge vectors ``vectors`` [E, 3] strained as ``v + v sym(eps)``, from
    their gradient ``gradient`` [E, 3] at eps = 0: sym(V^T G), as one elementwise product and a column sum (the strain
    leaf's backward is the same product as a split-K GEMM). ``mask`` [E] (a padded slot's edge mask) drops the padding
    edges, as the forces' scatter does."""
    if mask is not None:
        gradient = gradient * mask.unsqueeze(1)
    outer = (vectors.unsqueeze(2) * gradient.unsqueeze(1)).reshape(-1, 9).sum(dim=0).reshape(3, 3)
    return 0.5 * (outer + outer.transpose(0, 1))


def _tile_vectors(pos, ei, shifts, edge_mask, far):
    """A padded slot's edge vectors [E, 3]: ``pos[receiver] - pos[sender] + shift``, and ``far`` (beyond the cutoff)
    for the padding edges (``TILE_EPILOGUE``: one compiled kernel)."""
    import torch

    vectors = pos[ei[1]] - pos[ei[0]] + shifts
    return torch.where(edge_mask.unsqueeze(1) > 0, vectors, far)


def _tile_local(rows64, receiver_rows, local, shifts, cell, inverse, dtype):
    """``TILE_LOCAL_FRAME``, tile-local form: one tile's node rows ``rows64`` [rows, 3] (float64 positions) and its
    edges' float32 ``shifts`` as ``(positions, shifts)`` in ``dtype`` whose ``positions[local[1]] - positions[local[0]]
    + shifts`` are the edge vectors. Each halo row sits at its first edge's periodic image (``p - shift``), every row
    relative to the first receiver, in float64 before the cast; an edge's remaining shift (``shift`` less its sender
    row's) is zero unless one atom reaches the tile through two images. The float64 shift is ``round(shift @ cell^-1)
    @ cell`` (exact: the float32 shift is within its rounding of an integer combination of the cell's rows). Rows
    below ``receiver_rows`` keep their own image."""
    import torch

    count, e = rows64.shape[0], local.shape[1]
    from fastmace.neighbors.skin import times_3x3

    # Element-wise 3 x 3 products: as cuBLAS DGEMMs they cost 0.46 s of a 2.1 s call at 320,000 atoms on the RTX
    # 5060 Ti (local job 52; FP64 GEMM is slow on GeForce parts).
    shift = times_3x3(torch.round(times_3x3(shifts.double(), inverse)), cell)
    image = torch.zeros_like(rows64)
    if e:
        edge = torch.arange(e, device=local.device)
        first = torch.full((count,), e, dtype=edge.dtype, device=local.device).scatter_reduce(
            0, local[0], edge, reduce="amin")
        halo = (first < e) & (torch.arange(count, device=local.device) >= receiver_rows)
        image = torch.where(halo.unsqueeze(1), shift[first.clamp(max=e - 1)], image)
    origin = rows64[:1]
    return (rows64 - image - origin).to(dtype), (shift - image[local[0]]).to(dtype)


def _slot_local_parts(positions64, nodes, node_mask, local, edge_mask, shifts, cell, inverse, receivers, dtype):
    """``_tile_local`` on a padded slot's fixed-size buffers (rows ``nodes`` with ``node_mask``, edges ``local`` and
    ``shifts`` with ``edge_mask``; ``receivers`` slot receiver rows), each part split in two ``dtype`` halves:
    ``(rows, rows_low, remaining, remaining_low)``, the padding rows zero. The vector ``(rows[r] - rows[s] +
    remaining) + (rows_low[r] - rows_low[s] + remaining_low)`` is the float64 vector's rounding: the first sum is exact
    for an edge (nearby rows, Sterbenz), the second carries what float32 rows of up to the tile's extent drop.
    Without the low halves the rows' float32 spacing at R 16,384 (rows up to 56 A) left 4.4e-5 eV/A on the perfect
    40^3 crystal, and a tile that meets itself through the periodic boundary (20^3, R 16,384: 68,911 edges with a
    78 A remaining shift) 1.2e-4 (local job 61).

    Float64 per row: each edge's integer image ``round(shift @ cell^-1)`` in float32 (exact: |unit| small), a halo
    row's image the smallest encoded unit among its edges (deterministic, as every sweep must see the same rows),
    its float64 shift ``unit @ cell`` per row; per edge one float64 ``(unit - row unit) @ cell``. Local job 56 (no
    low halves): 1.003 / 1.003 at 320,000 atoms on the RTX 5060 Ti (three float64 3x3 products per edge: 1.020 /
    1.027)."""
    import torch

    from fastmace.neighbors.skin import times_3x3

    count = nodes.shape[0]
    unit = torch.round(times_3x3(shifts.to(torch.float32), inverse.to(torch.float32)))
    code = unit.to(torch.int32) + 64  # |unit| < 64
    code = (code[:, 0] * 128 + code[:, 1]) * 128 + code[:, 2]
    none = 1 << 21  # above every code
    code = torch.where(edge_mask > 0, code, torch.full_like(code, none))
    best = torch.full((count,), none, dtype=code.dtype, device=local.device).scatter_reduce(
        0, local[0], code, reduce="amin")
    halo = (best < none) & (torch.arange(count, device=local.device) >= receivers)
    best = torch.where(halo, best, torch.full_like(best, (64 * 128 + 64) * 128 + 64))  # unit 0
    row_unit = torch.stack((best // 16384, (best // 128) % 128, best % 128), dim=1).to(torch.float32) - 64
    rows64 = positions64[nodes]
    rows64 = (rows64 - times_3x3(row_unit.to(torch.float64), cell) - rows64[:1]) * node_mask.unsqueeze(1).to(
        torch.float64)
    remaining64 = times_3x3((unit - row_unit[local[0]]).to(torch.float64), cell)
    rows, remaining = rows64.to(dtype), remaining64.to(dtype)
    return rows, (rows64 - rows.to(torch.float64)).to(dtype), remaining, (remaining64 - remaining.to(
        torch.float64)).to(dtype)


def _slot_local(positions64, nodes, node_mask, local, edge_mask, shifts, cell, inverse, receivers, dtype):
    """``_slot_local_parts``' high halves: ``(rows, remaining shifts)``."""
    rows, _, remaining, _ = _slot_local_parts(positions64, nodes, node_mask, local, edge_mask, shifts, cell, inverse,
                                              receivers, dtype)
    return rows, remaining


def _slot_local_vectors(positions64, nodes, node_mask, local, edge_mask, shifts, cell, inverse, receivers, dtype,
                        far):
    """``TILE_LOCAL_FRAME`` inside the tile body: the slot's edge vectors from ``_slot_local_parts`` (``far`` for the
    padding edges, as ``_tile_vectors``), one fixed-shape graph per slot bucket. In the body the frame is part of the
    captured tile (``compiled_graphs``): built before the body as its own compiled call, its per-tile host overhead
    cost 1.047 / 1.048 at 975,560 atoms on the launch-bound A100 (Alex 4983258)."""
    import torch

    rows, low, remaining, remaining_low = _slot_local_parts(positions64, nodes, node_mask, local, edge_mask, shifts,
                                                            cell, inverse, receivers, dtype)
    both = torch.cat([rows, low], dim=1)  # one gather per end
    receiver, sender = both[local[1]], both[local[0]]
    a, b = receiver[:, :3], sender[:, :3]
    # TwoSum: difference + error == a - b exactly (rows of a tile that meets itself through the boundary are up to
    # a box apart, and their float32 difference rounds: 3.8e-6 A at 78 A, local job 62)
    difference = a - b
    a_part = difference + b
    b_part = a_part - difference
    error = (a - a_part) + (b_part - b)
    vectors = (difference + remaining) + (error + (receiver[:, 3:] - sender[:, 3:]) + remaining_low)
    return torch.where(edge_mask.unsqueeze(1) > 0, vectors, far)


def _frame_parts(positions64, width, dtype):
    """``TILE_LOCAL_FRAME``: float64 positions [n, 3] as int16 bins of ``width`` and offsets in ``dtype``."""
    import torch

    b = torch.floor(positions64 / width)
    if b.numel() and float(b.abs().max()) > 32000:
        raise ValueError(f"TILE_LOCAL_FRAME: a coordinate beyond {32000 * width:.0f} A")
    return b.to(torch.int16), (positions64 - b * width).to(dtype)


def _frame_vectors(frame, ei, shifts, edge_mask, far):
    """``TILE_LOCAL_FRAME``: ``_tile_vectors`` from the positions' bins and offsets (``frame``: bins [N, 3] int16,
    offsets [N, 3] float32, the float64 cell, its inverse and the bin width). The edge's unit shift is
    ``round(shift @ cell^-1)`` (exact: the float32 shift is within its rounding of an integer combination of the
    cell's rows), its float64 shift ``unit @ cell`` splits into ``k W + rem``; the bins' integer part is exact in
    float32, and the rest is a sum of terms below 2 W."""
    import torch

    bins, offsets, cell, inverse, width = frame
    from fastmace.neighbors.skin import times_3x3

    # Element-wise 3 x 3 products: as cuBLAS DGEMMs they cost 0.46 s of a 2.1 s call at 320,000 atoms on the RTX
    # 5060 Ti (local job 52; FP64 GEMM is slow on GeForce parts).
    shift = times_3x3(torch.round(times_3x3(shifts.double(), inverse)), cell)
    k = torch.floor(shift / width)
    rem = (shift - k * width).to(offsets.dtype)
    coarse = ((bins[ei[1]] - bins[ei[0]]).double() + k).mul_(width).to(offsets.dtype)
    vectors = coarse + ((offsets[ei[1]] - offsets[ei[0]]) + rem)
    if edge_mask is None:
        return vectors
    return torch.where(edge_mask.unsqueeze(1) > 0, vectors, far)


def _split_streams(device):
    """Two CUDA streams of two green contexts holding disjoint halves of ``device``'s SMs (one split of its SM
    resource, rounded to the split's granularity: 2 SMs before sm_90, 8 from it), with the contexts that keep them
    alive: ``(streams, contexts)``. Raises where the driver or the CUDA Python bindings cannot make them."""
    import torch
    from torch.cuda import green_contexts as gc

    drv, check = gc._drv, gc._check_cuda_bindings
    index = device.index if device.index is not None else torch.cuda.current_device()
    handle = check(drv.cuDeviceGet(index))
    sm = check(drv.cuDeviceGetDevResource(handle, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    step = 8 if torch.cuda.get_device_capability(index)[0] >= 9 else 2
    half = (sm.sm.smCount // 2) // step * step
    groups, count, _ = check(drv.cuDevSmResourceSplitByCount(2, sm, 0, half))
    if count != 2:
        raise RuntimeError(f"the SM split gave {count} groups of {half} SMs, not 2")
    streams, contexts = [], []
    for group in list(groups)[:2]:
        desc = check(drv.cuDevResourceGenerateDesc([group], 1))
        green = check(drv.cuGreenCtxCreate(desc, handle, drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        context = gc.GreenContext.__new__(gc.GreenContext)
        context._init_from_cuda_objects(index, green, check(drv.cuCtxFromGreenCtx(green)))
        contexts.append(context)
        streams.append(context.Stream())
    return tuple(streams), contexts


_UNIT_WEIGHTS = (9, 3, 1)  # a unit shift in {-1, 0, 1}^3 as one byte: sum((u + 1) * w), 0..26


def _pack_units(unit):
    """A tile's unit shifts [E, 3] (integer valued, float) for the list cache: one byte each when all lie in
    {-1, 0, 1}^3 (every cell longer than the search radius), else int16; None when they exceed int16."""
    import torch

    if unit.numel() == 0 or float(unit.abs().max()) <= 1:
        weights = torch.tensor(_UNIT_WEIGHTS, dtype=unit.dtype, device=unit.device)
        return ((unit + 1) * weights).sum(1).to(torch.uint8)
    if float(unit.abs().max()) < 2**15:
        return unit.to(torch.int16)
    return None


def _filter32_band(cutoff, reach):
    """``TILE_FILTER32``'s band: a bound on |d2_32 - d2_64| near the cutoff, for float32 copies of coordinates and
    shifts up to ``reach`` in magnitude, times four. Each vector component carries at most 6 * reach * 2^-24 (the
    inputs' rounding and the subtraction's and addition's), and d2 = |v|^2 near |v| = cutoff moves by at most
    2 sqrt(3) cutoff e + 3 e^2 for a component error e, plus the float32 sum's 4 d2 2^-24."""
    e = 6.0 * (abs(reach) + cutoff) * 2.0 ** -24
    return 4.0 * (2.0 * 3.0 ** 0.5 * cutoff * e + 3.0 * e * e + 4.0 * cutoff * cutoff * 2.0 ** -24)


def _filter32(positions32, positions64, shifts27, band, cutoff, sender, receiver, unit, codes, float_dtype):
    """``TILE_FILTER32``: the kept lists' cutoff test of the float64 one (``positions64[receiver] -
    positions64[sender] + shift`` squared and summed below cutoff^2), decided in float32 except within ``band``,
    where the float64 test runs on those edges alone. Returns ``(sender, receiver, unit, shifts)`` as the float64
    path does: the shifts are ``shifts27`` rows (``times_3x3`` of the one-byte codes) in ``float_dtype``."""
    import torch

    shift = shifts27[codes]  # float64 [E, 3], each row bitwise times_3x3 of its unit shift
    v32 = positions32[receiver] - positions32[sender] + shift.to(torch.float32)
    d2 = (v32 * v32).sum(dim=1)
    limit = cutoff ** 2
    keep = d2 < limit
    close = ((d2 - limit).abs() <= band).nonzero().squeeze(1)
    if close.numel():
        v = positions64[receiver[close]] - positions64[sender[close]] + shift[close]
        keep[close] = (v * v).sum(dim=1) < limit
    return sender[keep], receiver[keep], unit[keep], shift[keep].to(float_dtype)


def _unpack_units(packed, dtype, device):
    """The float unit shifts [E, 3] of ``_pack_units``' result, exactly."""
    import torch

    if packed.dtype != torch.uint8:
        return packed.to(dtype)
    codes = torch.arange(27, device=device)
    table = torch.stack([codes // 9, codes // 3 % 3, codes % 3], dim=1).to(dtype) - 1
    return table[packed.long()]


def _pack_senders(sender):
    """A tile's senders [E] (int64 atom indices) for the host lists: ``(dictionary, codes)``, the tile's distinct
    senders (int32, ascending) and each edge's rank among them as int16 (offset by -32768), while they number at
    most 65,536 (a tile's receivers and halo: every tile up to the 16,384 cap); else an empty dictionary and the
    int32 senders. 2 bytes per edge instead of 4 (the kept lists: about 40 % of their host memory)."""
    import torch

    dictionary, codes = torch.unique(sender, sorted=True, return_inverse=True)
    if dictionary.shape[0] <= 2**16:
        return dictionary.to(torch.int32), (codes - 2**15).to(torch.int16)
    return dictionary.new_empty(0, dtype=torch.int32), sender.to(torch.int32)


def _unpack_senders(dictionary, codes):
    """The int64 senders of ``_pack_senders``' result, exactly."""
    import torch

    if codes.dtype != torch.int16:
        return codes.long()
    return dictionary.long()[codes.long() + 2**15]


def moved_within(positions, reference, limit, chunk=None):
    """Whether no atom of ``positions`` (any device) is more than ``limit`` from its row of ``reference`` (host
    float64): the squared distances in float64 by chunks of ``chunk`` atoms (default the tile lists' ``CHUNK``),
    stopping at the first chunk past it. The same decision as on the whole arrays, with transients of one chunk
    instead of three whole-system float64 copies on the host (72 bytes per atom; tiled capacity)."""
    import torch

    if chunk is None:
        from fastmace.neighbors.tile_list import CHUNK as chunk
    bound = float(limit) ** 2
    for start in range(0, int(positions.shape[0]), int(chunk)):
        stop = min(start + int(chunk), int(positions.shape[0]))
        here = positions[start:stop].detach().to("cpu", torch.float64)
        if float(((here - reference[start:stop]) ** 2).sum(dim=1).max()) > bound:
            return False
    return True


def _pinned_block(nbytes):
    """The bytes PyTorch's pinned host allocator holds for a pinned tensor of ``nbytes``: its block, ``nbytes``
    rounded up to a power of two (CachingHostAllocator)."""
    return 1 << max(int(nbytes) - 1, 0).bit_length() if nbytes > 0 else 0


def _kept_bytes(kept):
    """The host bytes a kept-lists dict holds: its reference positions and its tiles' lists (and their A/B mirrors),
    a pinned tensor by its allocator block (``_pinned_block``)."""
    def held(t):
        nbytes = t.numel() * t.element_size()
        return _pinned_block(nbytes) if t.is_pinned() else nbytes

    entries = list(kept["superset"].values()) + list(kept.get("mirrors", {}).values())
    return int(sum(held(t) for entry in entries for t in entry) + kept["reference"].numel() * 8)


HEAP_TRIMS = [0, 0.0]  # the process's malloc_trim calls and their seconds (the call's stats: heap_trims, heap_trim_s)


def _trim_heap():
    """``malloc_trim(0)``: glibc's free heap (every arena) back to the system (``TILE_HEAP_TRIM``). True if glibc
    returned memory; False where it did not, or without glibc."""
    if not TILE_HEAP_TRIM:
        return False
    import ctypes

    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (OSError, AttributeError):  # not glibc (macOS, Windows, musl)
        return False
    trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
    started = time.perf_counter()
    returned = bool(trim(0))
    HEAP_TRIMS[0] += 1
    HEAP_TRIMS[1] += time.perf_counter() - started
    return returned


def _release_kept(kept):
    """Kept lists dropped or replaced: their tensors go, and pinned ones (``TILE_PINNED_SUPERSET``) back to the
    operating system. PyTorch's pinned allocator keeps a freed block for reuse, resident, so without this the bytes
    the host budget counts as freed (``_auto_store``, ``list_reuse_floor``) stayed in the process: at 179.7M SrTiO3
    atoms on the RTX 4000 Ada the host peak grew by 7 GB (two list builds, each 4 GiB of pinned lists, the blocks
    rounded up to powers of two)."""
    if kept is None:
        return
    kept["superset"].clear()
    kept.get("mirrors", {}).clear()
    kept["complete"] = False
    _trim_heap()  # the pageable lists' heap pages (TILE_HEAP_TRIM)
    if kept.get("pinned_bytes"):
        kept["pinned_bytes"] = 0
        import torch

        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()  # the free pinned blocks (also staging buffers no longer in use)


def _host_available():
    """The host's available memory in bytes: ``MemAvailable``, or less where the process's memory cgroup leaves less
    (``_cgroup_room``: a batch job's limit, e.g. 231.5 GB for one A100-80 on an Alex node, which /proc/meminfo does not
    show); None where /proc/meminfo is missing."""
    available = None
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except OSError:
        return None
    room = _cgroup_room()
    if available is None or room is None:
        return available
    return min(available, room)


def _cgroup_room(root="/sys/fs/cgroup", own="/proc/self/cgroup"):
    """The bytes the process's memory cgroups leave it: over the cgroup and its ancestors with a limit, the least of
    the limit less the usage (less the reclaimable inactive file cache). cgroup v2 (``memory.max``,
    ``memory.current``, ``memory.stat``) or v1 (``memory.limit_in_bytes``, ``memory.usage_in_bytes``); None
    without a limit or outside Linux."""
    import os

    try:
        with open(own) as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None

    def read(path):
        try:
            with open(path) as handle:
                return handle.read()
        except OSError:
            return None

    def inactive_file(directory, key):
        stat = read(os.path.join(directory, "memory.stat")) or ""
        for line in stat.splitlines():
            name, _, value = line.partition(" ")
            if name == key:
                return int(value)
        return 0

    rooms = []
    for line in lines:
        hierarchy, controllers, path = (line.split(":", 2) + ["", ""])[:3]
        v2 = hierarchy == "0" and controllers == ""
        if not v2 and "memory" not in controllers.split(","):
            continue
        base = root if v2 else os.path.join(root, "memory")
        parts = [part for part in path.split("/") if part]
        for depth in range(len(parts), -1, -1):
            directory = os.path.join(base, *parts[:depth])
            if v2:
                limit = read(os.path.join(directory, "memory.max"))
                usage = read(os.path.join(directory, "memory.current"))
                key = "inactive_file"
            else:
                limit = read(os.path.join(directory, "memory.limit_in_bytes"))
                usage = read(os.path.join(directory, "memory.usage_in_bytes"))
                key = "total_inactive_file"
            if limit is None or usage is None or limit.strip() == "max" or int(limit) >= 2**60:
                continue
            used = int(usage) - inactive_file(directory, key)
            rooms.append(int(limit) - max(used, 0))
    return min(rooms) if rooms else None


def _narrow_counts(counts):
    """Per-receiver edge counts at the narrowest integer dtype that holds them (uint8, int16 or int32)."""
    import torch

    top = int(counts.max()) if counts.numel() else 0
    return counts.to(torch.uint8 if top < 2**8 else torch.int16 if top < 2**15 else torch.int32)


class TiledForceField:
    """The tiled three-sweep force call around an eager ``MACEForceField`` (``field``, ``compile_mode="none"``).

    ``field`` keeps its neighbour list, edges, species tables and rewritten model; this class only schedules the
    model's modules. ``evaluate(atoms, stress)`` is the one-off evaluation of SPEC §7.1; ``__call__(positions)``
    the force call on the field's current edges (after ``rebuild``, ``compact`` and ``use_edges``).

    ``TileConfig.mode``: ``"eager"`` runs each tile at its own shape; ``"padded"`` and ``"graphs"`` run every tile in
    one padded slot of fixed receiver, halo and edge counts (``_Slot``), eagerly or as one captured CUDA graph per
    sweep that reads the slot's static buffers. The slot grows, and the graphs are captured again, only when a tile
    does not fit it.
    """

    def __init__(self, field, tiles: TileConfig | None = None):
        import torch

        self.field = field
        self.config = tiles or TileConfig()
        model = getattr(field.model, "_orig_mod", field.model)
        # A compiled force field's model is the simplified, traceable copy (``compile_model``): the tile bodies
        # run its modules (``_orig_mod``), eagerly or, in mode "compiled", under their own torch.compile. An eager
        # field's e3nn modules cannot be compiled (their TorchScript parts would differentiate to zero).
        if self.config.mode == "compiled" and field.compile_mode == "none":
            raise ValueError("TileConfig.mode='compiled' needs a compiled force field (compile_mode='default')")
        if type(model).__name__ != "ScaleShiftMACE" or len(model.interactions) != 2:
            raise ValueError("TiledForceField supports two-interaction ScaleShiftMACE models only")
        if hasattr(model, "joint_embedding") or getattr(field, "is_polar", False):
            raise ValueError("TiledForceField does not support joint embeddings or PolarMACE")
        if self.config.mode == "graphs" and field.device.type != "cuda":
            raise ValueError("TileConfig.mode='graphs' needs a CUDA device")
        if self.config.compiled_graphs is True and field.device.type != "cuda":
            raise ValueError("TileConfig.compiled_graphs needs a CUDA device")
        # rewrites.radial.nested: the tile calls take the plain products, so the plain tables come back (one table)
        from fastmace.rewrites.radial import nested

        nested.release(model.modules())
        self.model = model
        self.device = field.device
        self.dtype = field.dtype
        self._big = torch.tensor(torch.iinfo(torch.int64).max, device=self.device)
        skip = model.interactions[1].skip_tp
        required = [p for p in inspect.signature(skip.forward).parameters.values()
                    if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        self._skip_takes_attrs = len(required) >= 2
        # Padding edges: a vector beyond the cutoff, so their radial envelope, and every term it multiplies, is 0.
        r_max = float(getattr(model, "r_max", 10.0))
        self._far = torch.tensor([2.0 * r_max + 1.0, 0.0, 0.0], dtype=self.dtype, device=self.device)
        self._holders = (_holder_class()(), _holder_class()())
        # The reassociated first interactions read species tables evaluated at the tile's own row count (the
        # field's ``species_tables``): bitwise the features the tile's embedding gives its atoms.
        self._reassociations = [entry[1] for entry in field._tables]
        # TileConfig.species="index": each tile's one-hot rows from the field's species index (the one-hot's
        # column of every atom), bitwise the stored rows; the field's device one-hot goes to the host.
        self._species = None
        if self.config.neighbours == "global" and not getattr(field, "global_list", True):
            raise ValueError("TileConfig.neighbours='global' reads the force field's own neighbour list, which this "
                             "one was built without (global_list=False): use neighbours='tiles'")
        if self.config.species == "index":
            from fastmace.model import energy as energy64

            if energy64.NODE_E0_KEY not in field.data:
                raise ValueError("TileConfig.species='index' needs the force field's float64 per-atom E0 "
                                 "(its one-hot node_attrs are not kept on the device)")
            stored = field.data["node_attrs"]
            self._species, self._width, self._attrs_dtype = field._species_index, stored.shape[1], stored.dtype
            field.release_node_attrs()
            field.release_whole_system()  # the per-atom buffers only the whole-system call reads
            self._species = field._species_index  # now uint8 (int32 with 256 or more columns)
        scale = self.config.backward_chunk_scale
        if scale is None:
            name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None
            scale = TILE_CHUNK_SCALE.get(name, 1)
        self.backward_chunk_scale = int(scale)
        # TILED_SPEED_GPUS: where the tiled speed switches left at None are on
        self._speed_gpu = self.device.type == "cuda" and torch.cuda.get_device_name(self.device) in TILED_SPEED_GPUS
        # TileConfig.compiled_graphs: None is on for COMPILED_GRAPH_GPUS; the memory guard's verdict (None until the
        # first compiled call measured), its measurement (workspace, available, slots) and the reason
        self._graph_gpu = (self.device.type == "cuda"
                           and torch.cuda.get_device_name(self.device) in COMPILED_GRAPH_GPUS)
        self._graphs_fit, self._graph_memory, self.graphs_state = None, None, None
        for module in model.modules():  # from each module's own value: a second tiled field scales it once
            if getattr(module, "chunked_backward_edges", 0) > 0:
                base = getattr(module, "_tile_chunk_base", module.chunked_backward_edges)
                module._tile_chunk_base = base
                module.chunked_backward_edges = base * self.backward_chunk_scale
        self._compiled = {}
        self._slots = {}  # edge bucket -> padded slot (with its graphs)
        self._edge_step = None
        # TILE_LARGE_CAPS: a default budget of a host-state call takes the large caps too, where they fit
        if getattr(self.config, "_budget_default", False) and self.config.state == "host" and self._large_caps():
            self.config = dataclasses.replace(self.config, budget_receivers=tuple(sorted(
                {*map(int, self.config.budget_receivers), *TILE_LARGE_CAP_RECEIVERS})))
        self.receivers = int(self.config.receivers)  # the receiver cap (budget_receivers may raise it once)
        # mode "auto": a compiled force field's first call runs eager tiles and measures its peak; the later calls
        # compile when the compiled residency and a margin fit (``_choose_mode``). An eager field stays eager.
        self._auto = self.config.mode == "auto" and field.compile_mode != "none"
        if self.config.mode != "auto":
            self.mode, self.mode_reason = self.config.mode, "configured"
        elif self._auto:
            self.mode, self.mode_reason = "eager", "compiled force field: the first call measures its memory"
        else:
            self.mode, self.mode_reason = "eager", "the force field is not compiled"
        self._released = False  # the allocator's cache, released once before the first tile
        self._budgeted = not self.config.budget_receivers
        self._budget_raised, self._fallback = False, None  # the budget raised the cap; the last step down
        self.slot_changes = 0  # slots created or grown (each change compiles and captures again)
        self._kept, self._reusing, self._tile_index = None, False, {}  # list_reuse: the kept lists and plan
        self._probe = 0  # TILE_FIRST_CALL_PROBE: the tiles this measuring call runs (0: all)
        self._grid_fit = None  # (key, blocks per axis) of fit_grid's last choice
        self._reuse_ok = None
        self.list_reuse_state = "off" if not (self.config.list_reuse and self.config.neighbours == "tiles") else None
        self.state_dtype_state = None  # TileConfig.state_dtype "auto": the host state's dtype and why
        self.list_cache_state = "on" if self.config.list_cache else "off"
        self.list_rebuilds = 0  # list_reuse: the calls that searched (and planned) again
        self._graph_key = None
        self._pool = None
        self._stream = None
        self._cache_paused = False  # the wavefront's search pass: the tile lists are not kept
        self._graph_takes = False  # TILE_FILL_SKIP: the graph cache takes the build under way
        self._wave_auto = {}  # TileConfig.wavefront "auto": the decision per atom count
        self._wave_short = {}  # ... and whether the plain state would not fit the host at all
        # TileConfig.lanes: the second lane's padded slots (by edge bucket), its green-context streams (with their
        # contexts) once made, whether the first call's memory model fits a second workspace, and the decision.
        self._lane_slots, self._lane_streams, self._lanes_fit = {}, None, None
        self._plan_skin = 0.0  # TileConfig.boundary_skin: the reach the current plan split at (0: none)
        self._prep_stream = None  # TileConfig.prefetch's side stream
        self._copy_stream = None  # TILE_COPY_STREAM's side stream
        self._pieces, self._piece_bytes, self._piece_budget, self._piece_count = None, 0, 0, 0  # TILE_GRAPH_CACHE
        self._list_cache = None  # TileConfig.list_cache (neighbours "tiles"; set per call by tiles())
        self._per_receiver = None  # the last measured tile workspace per receiver (bytes; _memory)
        self.lanes_state = "off" if self.config.lanes == "off" else None
        self.last = {}

    @property
    def compute_stress(self):
        return self.field.compute_stress

    # ------------------------------------------------------------------ one tile

    def _tile_graph(self, receivers, source, local_of, halo_start=None):
        """Local graph of one tile: ``(nodes [R + H], ei [2, Ec], shifts [Ec, 3], local edge_index [2, Ec], R, H)``.

        ``source(receivers)`` gives the edges into the receivers, receiver-sorted: ``(ei, shifts)`` with ``ei`` the
        global (sender, receiver) pairs. The halo's local indices start at ``halo_start`` (default: after the
        receivers; a padded slot puts them after its receiver slots)."""
        import torch

        r = receivers.shape[0]
        start = r if halo_start is None else int(halo_start)
        ei, shifts = source(receivers)
        senders = ei[0]
        local_of[receivers] = torch.arange(r, dtype=local_of.dtype, device=self.device)
        outside = senders[local_of[senders] < 0]
        halo = torch.unique(outside)
        local_of[halo] = torch.arange(start, start + halo.shape[0], dtype=local_of.dtype, device=self.device)
        local = torch.stack([local_of[senders], local_of[ei[1]]]).long()  # local_of is int32
        local_of[receivers] = -1
        local_of[halo] = -1
        return torch.cat([receivers, halo]), ei, shifts, local, r, halo.shape[0]

    def _piece(self, index, receivers, source, local_of, halo_start=None):
        """``_tile_graph`` of tile ``index``, from the call's graph cache (``TILE_GRAPH_CACHE``) when it holds it;
        a first build is kept there while the cache's budget allows."""
        import torch

        cache = self._pieces
        entry = cache.get(index) if cache is not None else None
        if entry is None:
            self._graph_takes = (cache is not None and self._piece_bytes < self._piece_budget
                                 and bool(self._speed(TILE_FILL_SKIP)))
            try:
                piece = self._tile_graph(receivers, source, local_of, halo_start)
            finally:
                self._graph_takes = False
            if cache is not None:
                nodes, ei, shifts, local, r, h = piece
                size = nodes.numel() * nodes.element_size() + local.numel() * 4 + shifts.numel() * shifts.element_size()
                if self._piece_bytes + size <= self._piece_budget and local.numel() < 2**31:
                    start = r if halo_start is None else int(halo_start)
                    # local relative to a halo start of r: halo row r + k (int32: every index is below R + H)
                    relative = (local if start == r else torch.where(local >= start, local - (start - r), local))
                    cache[index] = (nodes, relative.to(torch.int32), shifts, r, h)
                    self._piece_bytes += size
                    self._piece_count += 1
            return piece
        nodes, relative, shifts, r, h = entry
        local = relative.long()
        ei = nodes[local]  # the global (sender, receiver) pairs, as the list gave them
        start = r if halo_start is None else int(halo_start)
        if start != r:
            local = torch.where(local >= r, local + (start - r), local)
        return nodes, ei, shifts, local, r, h

    def _kept_on_host(self, kept, tensors):
        """A tile's kept lists on the host: pinned (``TILE_PINNED_SUPERSET``) while the pinned ones total at most
        ``TILE_PINNED_SUPERSET_BYTES`` (counted by the pinned allocator's blocks, ``_pinned_block``), else
        pageable."""
        size = sum(_pinned_block(t.numel() * t.element_size()) for t in tensors)
        pinned = kept.get("pinned_bytes", 0)
        if (self.device.type == "cuda" and self._speed(TILE_PINNED_SUPERSET)
                and pinned + size <= TILE_PINNED_SUPERSET_BYTES):
            kept["pinned_bytes"] = pinned + size
            import torch

            out = tuple(torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in tensors)
            for host, t in zip(out, tensors):
                host.copy_(t)  # blocking: the host keeps it past this call
            return out
        return tuple(t.cpu() for t in tensors)

    def _kept_form(self, kept, index, superset):
        """Tile ``index``'s kept lists as ``TILE_PINNED_SUPERSET`` wants them now. They are stored as the switch was
        when they were kept; where it changed since (a same-process A/B's other side), a mirror in the other form is
        made once and kept beside them (never with the switch left as it is)."""
        if not superset or self.device.type != "cuda":
            return superset
        want = bool(self._speed(TILE_PINNED_SUPERSET))
        if superset[0].is_pinned() == want:
            return superset
        mirrors = kept.setdefault("mirrors", {})
        mirror = mirrors.get(index)
        if mirror is None:
            mirror = self._kept_on_host(kept, superset) if want else tuple(t.clone() for t in superset)
            if mirror[0].is_pinned() != want:  # past the pinned cap: pageable, as stored
                return superset
            mirrors[index] = mirror
        return mirror

    def _graph_budget(self, wave, measure=False):
        """The call's graph cache (``TILE_GRAPH_CACHE``): empty with its byte budget, or None (off, under the
        wavefront without ``TILE_GRAPH_CACHE_WAVE``, off a CUDA device, or in a call that measures its workspace, where the kept graphs would count
        as tile workspace and lower the receiver cap). The budget is the fraction of the memory free at the call's
        start less the tiles' predicted workspace (the last measured workspace per receiver times the cap, plus
        10 %, as ``_choose_receivers``), so the graphs never take what the tiles need."""
        self._pieces, self._piece_bytes, self._piece_budget, self._piece_count = None, 0, 0, 0
        fraction = self._speed(TILE_GRAPH_CACHE, TILE_GRAPH_CACHE_FRACTION)
        wave_ok = not wave or self._speed(TILE_GRAPH_CACHE_WAVE)
        free = self._free_bytes() if fraction and wave_ok and not measure else None
        if free is not None:
            workspace = 1.1 * self._per_receiver * self.receivers if self._per_receiver else 0
            self._pieces, self._piece_budget = {}, int(float(fraction) * max(0.0, free - workspace))

    def _large_caps(self):
        """``TILE_LARGE_CAPS``: the constant when set, else on for ``TILE_LARGE_CAP_GPUS``."""
        import torch

        if TILE_LARGE_CAPS is not None:
            return bool(TILE_LARGE_CAPS)
        return self.device.type == "cuda" and torch.cuda.get_device_name(self.device) in TILE_LARGE_CAP_GPUS

    def _speed(self, value, on=True):
        """A tiled speed switch's value (``TILE_VIRIAL``, ``TILE_PREFETCH_DEVICE``, ``TILE_GRAPH_CACHE``, ...): the
        constant when set, else ``on`` on ``TILED_SPEED_GPUS`` and off elsewhere."""
        if value is not None:
            return value
        return on if getattr(self, "_speed_gpu", False) else type(on)(0)

    def _free_bytes(self):
        """Device memory free for the graph cache: the driver's free memory and the allocator's unused reserve
        (None off a CUDA device)."""
        import torch

        if self.device.type != "cuda":
            return None
        free, _ = torch.cuda.mem_get_info(self.device)
        return free + torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)

    def _source(self, positions, n):
        """``(source, row_ptr, edges per atom)`` of this call's edges: the force field's list, or (neighbours
        "tiles") a cell binning of ``positions`` (float64) that builds each tile's edges."""
        import torch

        data = self.field.data
        if self.config.neighbours == "tiles":
            from fastmace.neighbors.skin import times_3x3
            from fastmace.neighbors.tile_list import TileNeighbours

            # The kept lists hold every tile's (55.8 GiB at 122M SrTiO3 atoms beside a 55 GiB wavefront pool), and
            # a call that keeps them searches once: dropping them under the wavefront searched every tile twice per
            # call (the search pass and sweep 1), 9.2 s against 1.8 s each at 5M atoms (+18 % of a call, RTX 4000
            # Ada). The host floor drops them where the host runs short; ``list_reuse=False`` keeps only the
            # band's (the call's cache, from a tile's sweep 1 to its sweep 3).
            skin, cutoff = self._reuse_skin(), float(self.field.neighbours.cutoff)
            self._reusing = False
            kept = (self._keep_lists(positions, n, TileNeighbours.like(self.field.neighbours).cell, skin)
                    if skin > 0 else None)
            bins = TileNeighbours.like(self.field.neighbours, cutoff=cutoff + (skin if kept is not None else 0.0))
            if not self._reusing:
                bins.bin(positions)
            # Float64 positions for the kept lists' distance test only (24 bytes per atom on the device).
            positions64 = positions.to(device=self.device, dtype=torch.float64) if kept is not None else None
            # TILE_FILTER32: float32 positions (12 bytes per atom) and the band of squared lengths tested again
            # in float64; the float64 shifts of the 27 one-byte unit codes
            fast = kept is not None and self._reusing and self._speed(TILE_FILTER32)
            if fast:
                positions32 = positions64.to(torch.float32)
                codes = torch.arange(27, device=self.device)
                units27 = torch.stack([codes // 9, codes // 3 % 3, codes % 3], dim=1).to(torch.float64) - 1
                shifts27 = times_3x3(units27, bins.cell.to(device=self.device, dtype=torch.float64))
                reach = float(positions64.abs().max()) + float(shifts27.abs().max()) if n else 0.0
                band = _filter32_band(cutoff, reach)
            # With kept lists, the cache holds the tiles' filtered lists from their sweep 1 (under the wavefront until
            # their sweep 3: the band), so sweeps 2 and 3 skip the superset's copy and float64 filter. At the host
            # limit (the floor reached while it fills) the cache is dropped for this and later calls: the later sweeps
            # search or filter again, bitwise the same edges.
            self._list_hits = 0
            self._list_cache = ({} if self.config.list_cache and n < 2**31 and self.list_cache_state == "on"
                                else None)

            def within(receivers):
                """The tile's edges within the cutoff from the kept ones within cutoff + skin (searched there on a
                call that searches): a fresh search's float64 distance test, its order kept."""
                index = self._tile_index[receivers.data_ptr()]
                superset = kept["superset"].get(index)
                if superset is None:
                    # Float64 positions: the search's own vectors are the filter's below, so they give its mask.
                    exact = bins.positions.dtype == torch.float64
                    sender, receiver, unit, shifts, *inner = bins.edges(receivers, inner=cutoff if exact else None)
                    packed = _pack_units(unit)
                    if packed is None:  # beyond int16 shifts: not kept, searched again next call
                        kept["complete"] = False
                    elif not self._host_allows(f"keeping tile {index}'s lists"):
                        kept["complete"] = False
                        kept["superset"].clear()
                    elif kept["complete"]:
                        counts = torch.bincount(torch.searchsorted(receivers, receiver),
                                                minlength=receivers.shape[0])
                        kept["superset"][index] = self._kept_on_host(kept, (*_pack_senders(sender),
                                                                            _narrow_counts(counts), packed))
                    if exact:
                        inner = inner[0]
                        return sender[inner], receiver[inner], unit[inner], shifts[inner]
                else:
                    dictionary, codes, counts, packed = (t.to(self.device, non_blocking=t.is_pinned())
                                                         for t in self._kept_form(kept, index, superset))
                    sender = _unpack_senders(dictionary, codes)
                    receiver = receivers.repeat_interleave(counts.long(), output_size=sender.shape[0])
                    unit = _unpack_units(packed, bins.float_dtype, self.device)
                    if fast and packed.dtype == torch.uint8:
                        return _filter32(positions32, positions64, shifts27, band, cutoff, sender, receiver, unit,
                                         packed.long(), bins.float_dtype)
                shifts64 = times_3x3(unit.to(torch.float64), bins.cell)
                vectors = positions64[receiver] - positions64[sender] + shifts64
                keep = (vectors * vectors).sum(dim=1) < cutoff ** 2
                del vectors
                return sender[keep], receiver[keep], unit[keep], shifts64[keep].to(bins.float_dtype)

            def tiles(receivers):
                """The tile's (global edge index, shifts): searched once per call, then from the host cache (int32
                senders, the receivers' edge counts at their narrowest dtype, the unit shifts packed)."""
                key = receivers.data_ptr()  # the call's tile tensors live through all its sweeps
                cache = self._list_cache  # (None once a fallback or the host floor dropped it)
                if cache is not None and key in cache:
                    self._list_hits += 1
                    dictionary, codes, counts, packed = (t.to(self.device) for t in cache[key])
                    sender = _unpack_senders(dictionary, codes)
                    receiver = receivers.repeat_interleave(counts.long(), output_size=sender.shape[0])
                    unit = _unpack_units(packed, bins.float_dtype, self.device)
                    return torch.stack([sender, receiver]), bins.shifts(unit)
                sender, receiver, unit, shifts = within(receivers) if kept is not None else bins.edges(receivers)
                packed = (_pack_units(unit) if cache is not None and self.list_cache_state == "on"
                          and not self._cache_paused and not self._graph_takes else None)
                # (the wavefront's search pass keeps none, nor a build the graph cache takes: TILE_FILL_SKIP)
                if packed is not None and not self._host_keeps_cache(cache):
                    packed = None
                if packed is not None:
                    counts = torch.bincount(torch.searchsorted(receivers, receiver), minlength=receivers.shape[0])
                    cache[key] = tuple(t.cpu() for t in (*_pack_senders(sender), _narrow_counts(counts), packed))
                return torch.stack([sender, receiver]), shifts

            # No degrees before the tiles' builds: the tiles are planned by receivers only.
            return tiles, None, None
        edge_index, shifts = data["edge_index"], data["shifts"]
        row_ptr = torch.searchsorted(edge_index[1], torch.arange(n + 1, device=self.device))

        def global_list(receivers):
            starts = row_ptr[receivers]
            counts = row_ptr[receivers + 1] - starts
            total = int(counts.sum())
            offsets = torch.cumsum(counts, 0) - counts
            edge_ids = (torch.repeat_interleave(starts - offsets, counts, output_size=total)
                        + torch.arange(total, device=self.device))
            return edge_index[:, edge_ids], shifts[edge_ids]

        return global_list, row_ptr, edge_index.shape[1] / max(n, 1)

    def _reuse_skin(self):
        """``list_reuse``'s skin: the force field's list's, when the lists can be kept (tile lists, at least three
        bins of cutoff + skin per axis); else 0. A cell that changes (NPT, a strained one-off call) searches again:
        ``_keep_lists`` compares the cell every call."""
        neighbours = self.field.neighbours
        skin = float(getattr(neighbours, "skin", 0.0) or 0.0)
        if not (self.config.list_reuse and self.config.neighbours == "tiles" and skin > 0):
            return 0.0
        if self._reuse_ok is None:
            from fastmace.neighbors.tile_list import TileNeighbours

            try:
                TileNeighbours.like(neighbours, cutoff=float(neighbours.cutoff) + skin)
                self._reuse_ok, self.list_reuse_state = True, "on"
            except ValueError:  # fewer than three bins per axis at cutoff + skin
                self._reuse_ok = False
                self.list_reuse_state = "off: fewer than three bins of cutoff + skin per cell axis"
        return skin if self._reuse_ok else 0.0

    def _host_allows(self, what):
        """Whether the host keeps ``list_reuse_floor`` bytes available (MemAvailable; always off Linux). If not,
        reuse is off from now on, with the reason in ``list_reuse_state``."""
        available = _host_available()
        if available is None or available >= int(self.config.list_reuse_floor):
            return True
        _release_kept(self._kept)
        self._reuse_ok, self._kept = False, None
        self.list_reuse_state = (f"off: {available / 2**30:.1f} GiB available on the host when {what}, below the "
                                 f"floor of {int(self.config.list_reuse_floor) / 2**30:.1f} GiB")
        return False

    def _host_keeps_cache(self, cache):
        """Whether the host keeps ``list_reuse_floor`` bytes available (MemAvailable; always off Linux) with the
        list cache filling. If not, the cache is emptied and off from now on (the later sweeps search again),
        with the reason in ``list_cache_state``."""
        available = _host_available()
        if available is None or available >= int(self.config.list_reuse_floor):
            return True
        cache.clear()
        _trim_heap()
        self.list_cache_state = (f"off: {available / 2**30:.1f} GiB available on the host while the list cache "
                                 f"filled, below the floor of {int(self.config.list_reuse_floor) / 2**30:.1f} GiB")
        return False

    def _keep_lists(self, positions, n, cell, skin):
        """The kept lists for this call: reused (``self._reusing``) while the atom count, cell and receiver cap
        (the budget may raise it after the first call) are those of the search and no atom has moved more than
        skin / 2 since it; else a new, empty set (this call searches and plans)."""
        import torch

        kept = self._kept
        reuse = (kept is not None and kept["complete"] and kept["n"] == n and kept["receivers"] == self.receivers
                 and torch.equal(kept["cell"], cell.cpu())
                 and moved_within(positions, kept["reference"], skin / 2))
        if not reuse:
            _release_kept(kept)  # the old reference and lists go before the new reference is copied
            self._kept = kept = None
            if not self._host_allows("keeping the reference positions"):
                return None
            here = positions.detach().to("cpu", torch.float64)
            kept = self._kept = dict(n=n, cell=cell.cpu(), receivers=self.receivers, reference=here, superset={},
                                     tiles=None, grid=None, complete=True)
            self.list_rebuilds += 1
        self._reusing = reuse
        return kept

    def _edges(self, vectors, node_attrs, local, strain):
        """Spherical harmonics, radial features and cutoff of the tile's edges from their vectors."""
        import torch

        model = self.model
        if strain is not None:
            vectors = vectors + vectors @ (0.5 * (strain + strain.transpose(0, 1)))
        lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        edge_attrs = model.spherical_harmonics(vectors)
        edge_feats, cutoff = model.radial_embedding(lengths, node_attrs, local, model.atomic_numbers)
        return lengths, edge_attrs, edge_feats, cutoff

    def _layer(self, index, node_feats, node_attrs, edge_attrs, edge_feats, cutoff, local, r):
        """Interaction ``index`` and its product on a tile: the receivers' output features [R, ...]."""
        model = self.model
        feats, sc = model.interactions[index](
            node_attrs=node_attrs, node_feats=node_feats, edge_attrs=edge_attrs, edge_feats=edge_feats,
            edge_index=local, cutoff=cutoff, first_layer=True, lammps_class=_GHOSTS,
            lammps_natoms=(r, node_feats.shape[0] - r))
        return model.products[index](node_feats=feats, sc=sc, node_attrs=node_attrs[:r])

    def _consumers(self, h1, attrs):
        """Interaction 1's two reads of h1: ``(linear_up(h1), skip_tp(h1))`` (``skip_tp`` with the species when
        it takes them)."""
        interaction = self.model.interactions[1]
        sc = interaction.skip_tp(h1, attrs) if self._skip_takes_attrs else interaction.skip_tp(h1)
        return interaction.linear_up(h1), sc

    def _attrs(self, nodes):
        """The one-hot species rows of ``nodes``: the field's, or (``species="index"``) built from its index."""
        import torch

        if self._species is None:
            return self.node_attrs[nodes]
        return torch.nn.functional.one_hot(self._species[nodes].long(), self._width).to(self._attrs_dtype)

    @staticmethod
    def _masked(value, mask):
        return value if mask is None else value * mask.unsqueeze(-1) if value.dim() > 1 else value * mask

    def _first(self, tile, state, adjoint=False, strain=None):
        """Interaction 0 on a tile. Sweep 1 (``adjoint`` False, no autograd): ``(es, x, sc)``, the receivers' node
        energies before scale and shift and interaction 1's inputs. Sweep 3 (``adjoint`` True): ``(es, edge-vector
        gradient, strain gradient or None)`` of interaction 0's energy plus <dE/dx, x> + <dE/dsc, sc>. Padding
        receivers (``recv_mask`` 0) give exact zeros; with ``TILE_EPILOGUE`` the edge-vector gradient is masked too."""
        import torch

        model = self.model
        nodes, local, r, vectors, recv_mask, _, edge_mask = tile
        attrs = self._attrs(nodes)
        receivers = nodes[:r]
        virial = adjoint and strain is not None and self._speed(TILE_VIRIAL)
        fused = self._speed(TILE_EPILOGUE)
        with torch.enable_grad() if adjoint else torch.no_grad():
            leaf = vectors.detach().requires_grad_(adjoint)
            eps = (torch.zeros((3, 3), dtype=self.dtype, device=self.device, requires_grad=True)
                   if adjoint and strain is not None and not virial else None)
            if fused and not adjoint:
                return self._forward("1m")(leaf, eps, attrs, local, r, recv_mask)
            if fused:
                es, objective = self._forward("3o")(leaf, eps, attrs, local, r, recv_mask,
                                                   state.rows("gx", receivers), state.rows("gsc", receivers))
            else:
                es, x, sc = self._forward(3 if adjoint else 1)(leaf, eps, attrs, local, r)
                heads = self.heads[:r]
                if not adjoint:
                    return self._masked(es, recv_mask), self._masked(x, recv_mask), self._masked(sc, recv_mask)
                objective = (self._masked(model.scale_shift(es, heads), recv_mask).sum()
                             + (self._masked(state.rows("gx", receivers), recv_mask) * x).sum()
                             + (self._masked(state.rows("gsc", receivers), recv_mask) * sc).sum())
            wanted = [leaf] + ([eps] if eps is not None else [])
            grads = torch.autograd.grad(objective, wanted)
        es = self._masked(es.detach(), recv_mask)
        if fused:
            g_vec, g_eps = self._epilogue("e")(vectors, grads[0], edge_mask, virial)
            return es, g_vec, g_eps if virial else grads[1] if eps is not None else None
        if virial:
            return es, grads[0], _virial(vectors, grads[0], edge_mask)
        return es, grads[0], grads[1] if eps is not None else None

    def _second(self, tile, state, strain):
        """Interaction 1 forward and backward on the stored ``linear_up(h1)`` and ``skip_tp(h1)`` (sweep 2):
        ``(es, edge-vector gradient, dE/dx [R + H], dE/dsc [R], strain gradient or None)``."""
        import torch

        model = self.model
        interaction = model.interactions[1]
        nodes, local, r, vectors, recv_mask, node_mask, edge_mask = tile
        attrs = self._attrs(nodes)
        virial = strain is not None and self._speed(TILE_VIRIAL)
        with torch.enable_grad():
            leaf = vectors.detach().requires_grad_(True)
            eps = (torch.zeros((3, 3), dtype=self.dtype, device=self.device, requires_grad=True)
                   if strain is not None and not virial else None)
            x = state.rows("x", nodes).detach().requires_grad_(True)
            sc = state.rows("sc", nodes[:r]).detach().requires_grad_(True)
            modules = interaction._modules
            saved = modules["linear_up"], modules["skip_tp"]
            holder_x, holder_sc = self._holders
            holder_x.value, holder_sc.value = x, sc
            modules["linear_up"], modules["skip_tp"] = holder_x, holder_sc
            try:  # their real counterparts ran in sweep 1, on the h1 that is not kept
                objective, es = self._forward(2)(leaf, x, eps, attrs, local, recv_mask, r)
            finally:
                modules["linear_up"], modules["skip_tp"] = saved
                holder_x.value = holder_sc.value = None
            wanted = [leaf, x, sc] + ([eps] if eps is not None else [])
            grads = torch.autograd.grad(objective, wanted)
        if self._speed(TILE_EPILOGUE):
            es, g_vec, g_x, g_sc, g_eps = self._epilogue("2")(vectors, es.detach(), grads[0], grads[1], grads[2],
                                                              node_mask, recv_mask, edge_mask, virial)
            return es, g_vec, g_x, g_sc, g_eps if virial else grads[3] if eps is not None else None
        g_eps = _virial(vectors, grads[0], edge_mask) if virial else grads[3] if eps is not None else None
        return (self._masked(es.detach(), recv_mask), grads[0], self._masked(grads[1], node_mask),
                self._masked(grads[2], recv_mask), g_eps)

    def _first_forward(self, leaf, eps, attrs, local, r):
        """Interaction 0's forward on a tile: ``(es, x, sc)`` (sweeps 1 and 3; compiled in mode "compiled")."""
        model = self.model
        lengths, edge_attrs, edge_feats, cutoff = self._edges(leaf, attrs, local, eps)
        node_feats = model.node_embedding(attrs)
        h1 = self._layer(0, node_feats, attrs, edge_attrs, edge_feats, cutoff, local, r)
        heads = self.heads[:r]
        es = model.readouts[0](h1, heads)[self.arange[:r], heads]
        if hasattr(model, "pair_repulsion"):
            es = model.pair_repulsion_fn(lengths, attrs, local, model.atomic_numbers)[:r] + es
        x, sc = self._consumers(h1, attrs[:r])
        return es, x, sc

    def _second_forward(self, leaf, x, eps, attrs, local, recv_mask, r):
        """Sweep 2's forward up to its objective (interaction 1 reads the holders' x and sc): ``(objective, es)``.
        Compiled as one function in mode "compiled"."""
        model = self.model
        _, edge_attrs, edge_feats, cutoff = self._edges(leaf, attrs, local, eps)
        out = self._layer(1, x, attrs, edge_attrs, edge_feats, cutoff, local, r)
        heads = self.heads[:r]
        es = model.readouts[1](out, heads)[self.arange[:r], heads]
        return self._masked(model.scale_shift(es, heads), recv_mask).sum(), es

    def _first_forward_masked(self, leaf, eps, attrs, local, r, recv_mask):
        """``TILE_EPILOGUE``: sweep 1's forward with the padding receivers' masks: ``(es, x, sc)``."""
        es, x, sc = self._first_forward(leaf, eps, attrs, local, r)
        return self._masked(es, recv_mask), self._masked(x, recv_mask), self._masked(sc, recv_mask)

    def _first_objective(self, leaf, eps, attrs, local, r, recv_mask, gx, gsc):
        """``TILE_EPILOGUE``: sweep 3's forward up to its objective, interaction 0's energy (after scale and shift)
        plus <dE/dx, x> + <dE/dsc, sc> over the real receivers: ``(es, objective)``."""
        es, x, sc = self._first_forward(leaf, eps, attrs, local, r)
        objective = (self._masked(self.model.scale_shift(es, self.heads[:r]), recv_mask).sum()
                     + (self._masked(gx, recv_mask) * x).sum() + (self._masked(gsc, recv_mask) * sc).sum())
        return es, objective

    @staticmethod
    def _edge_epilogue(vectors, g_vec, edge_mask, virial):
        """``TILE_EPILOGUE``: sweep 3's edge-vector gradient without its padding edges and (``virial``) the strain
        gradient of ``_virial``: ``(g_vec, g_eps or None)``."""
        g_vec = TiledForceField._masked(g_vec, edge_mask)
        return g_vec, _virial(vectors, g_vec) if virial else None

    @staticmethod
    def _second_epilogue(vectors, es, g_vec, g_x, g_sc, node_mask, recv_mask, edge_mask, virial):
        """``TILE_EPILOGUE``: sweep 2's outputs without their padding rows and edges and (``virial``) the strain
        gradient: ``(es, g_vec, g_x, g_sc, g_eps or None)``."""
        g_vec = TiledForceField._masked(g_vec, edge_mask)
        masked = TiledForceField._masked
        return (masked(es, recv_mask), g_vec, masked(g_x, node_mask), masked(g_sc, recv_mask),
                _virial(vectors, g_vec) if virial else None)

    def _epilogue(self, key):
        """``TILE_EPILOGUE``'s epilogue ``key`` ("e": sweep 3, "2": sweep 2): plain, or compiled in mode "compiled"."""
        plain = {"e": self._edge_epilogue, "2": self._second_epilogue}[key]
        return self._compiled_once(f"epilogue{key}", plain)

    def _forward(self, sweep):
        """The sweep's forward: plain, or (mode "compiled") compiled on first use, once per sweep (sweeps 1 and 3
        apart: without and with autograd; "1m" and "3o" are ``TILE_EPILOGUE``'s); each slot's shapes are one
        specialisation."""
        plain = {1: self._first_forward, 2: self._second_forward, 3: self._first_forward,
                 "1m": self._first_forward_masked, "3o": self._first_objective}[sweep]
        return self._compiled_once(sweep, plain)

    def _compiled_once(self, key, plain):
        """``plain``, or (mode "compiled") ``plain`` compiled on its first use under ``key``."""
        if self.mode != "compiled":
            return plain
        compiled = self._compiled.get(key)
        if compiled is None:
            import torch

            from fastmace.rewrites import inductor as inductor_records

            # One specialisation per slot (edge bucket) and sweep; dynamo would run eager beyond its limit.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            options = inductor_records.inductor_options(self.model)
            compiled = torch.compile(plain, dynamic=False, fullgraph=False, **({"options": options} if options else {}))
            self._compiled[key] = compiled
        return compiled

    # ------------------------------------------------------------------ padded slot

    def _new_slot(self, sizes):
        """Static buffers of one padded tile: ``sizes = (R, H, E)`` receiver slots (one more than any tile, so a
        padding receiver always exists), halo rows and edges."""
        import torch

        receivers, halo, edges = sizes
        i64 = torch.int64
        new = functools.partial(torch.zeros, device=self.device)
        tables = self.field.species_tables(receivers + halo) if self._reassociations else []
        return dict(sizes=sizes, nodes=new(receivers + halo, dtype=i64), tables=tables,
                    node_mask=new(receivers + halo, dtype=self.dtype), recv_mask=new(receivers, dtype=self.dtype),
                    ei=new((2, edges), dtype=i64), shifts=new((edges, 3), dtype=self.dtype),
                    local=new((2, edges), dtype=i64), edge_mask=new(edges, dtype=self.dtype),
                    count=new((), dtype=i64), out={}, graphs={})

    def _fill(self, slot, piece, g=None, host_nodes=None):
        """Copy one tile (``_tile_graph``'s output at the slot's halo start) into the slot's static buffers.

        Padding: receivers and halo rows read atom 0 and are masked; padding edges join the last receiver slot
        (always padding) to itself, after the tile's edges (so the receivers stay sorted), beyond the cutoff and
        beyond the tile's edge count (the streaming kernels' ``streaming_count``).

        TILE_HOST_ROWS (``g``'s tile-local frame on the host): the slot's float64 rows ``rows64`` are gathered on the
        host in the slot's layout (padding rows: atom 0's, as ``positions64[nodes]`` reads them) and copied in; the
        body reads them in place of ``positions64[nodes]``, the same values."""
        import torch

        nodes, ei, shifts, local, r, h = piece
        receivers, _, edges = slot["sizes"]
        e = ei.shape[1]
        frame = None if g is None else g.get("local_frame")
        if frame is not None and _tile_list().rows_on_host(frame[0], self.device):
            here = nodes.cpu() if host_nodes is None else host_nodes
            layout = torch.zeros(slot["nodes"].shape[0], dtype=torch.int64)
            layout[:r] = here[:r]
            layout[receivers:receivers + h] = here[r:]
            if "rows64" not in slot:  # fixed addresses (the captured bodies read them)
                slot["rows64"] = torch.zeros((layout.shape[0], 3), dtype=torch.float64, device=self.device)
                slot["row_ids"] = torch.arange(layout.shape[0], device=self.device)
            slot["rows64"].copy_(frame[0][layout])
        slot["nodes"].zero_()
        slot["nodes"][:r] = nodes[:r]
        slot["nodes"][receivers:receivers + h] = nodes[r:]
        slot["node_mask"].zero_()
        slot["node_mask"][:r] = 1
        slot["node_mask"][receivers:receivers + h] = 1
        slot["recv_mask"].zero_()
        slot["recv_mask"][:r] = 1
        # The edge buffers: the tile's edges, then the padding tail (the edges' rows are written, not cleared first).
        slot["ei"][:, :e] = ei
        slot["ei"][:, e:].zero_()
        slot["shifts"][:e] = shifts
        slot["shifts"][e:].zero_()
        slot["local"][:, :e] = local
        slot["local"][:, e:].fill_(receivers - 1)
        slot["edge_mask"][:e] = 1
        slot["edge_mask"][e:].zero_()
        slot["count"].fill_(e)

    def _body(self, sweep, slot, g):
        """One sweep on the slot's tile, reading only static buffers; writes ``slot["out"]`` (graph-capturable)."""
        import torch

        receivers = slot["sizes"][0]
        nodes, ei = slot["nodes"], slot["ei"]
        if g.get("local_frame") is not None:  # TILE_LOCAL_FRAME, tile-local: rows and remaining shifts in the body
            positions64, cell, inverse = g["local_frame"]
            vectors_of = (self._compiled_once("local_vectors", _slot_local_vectors) if self._speed(TILE_EPILOGUE)
                          else _slot_local_vectors)
            rows_of = nodes
            if "rows64" in slot:  # TILE_HOST_ROWS: the slot's own rows (``_fill``), read in place
                positions64, rows_of = slot["rows64"], slot["row_ids"]
            vectors = vectors_of(positions64, rows_of, slot["node_mask"], slot["local"], slot["edge_mask"],
                                 slot["shifts"], cell, inverse, receivers, self.dtype, self._far)
        elif g.get("frame") is not None:  # TILE_LOCAL_FRAME, bins and offsets (host positions)
            vectors = _frame_vectors(g["frame"], ei, slot["shifts"], slot["edge_mask"], self._far)
        elif self._speed(TILE_EPILOGUE):
            vectors = self._compiled_once("vectors", _tile_vectors)(g["pos"], ei, slot["shifts"], slot["edge_mask"],
                                                                   self._far)
        else:
            vectors = _tile_vectors(g["pos"], ei, slot["shifts"], slot["edge_mask"], self._far)
        tile = (nodes, slot["local"], receivers, vectors, slot["recv_mask"], slot["node_mask"], slot["edge_mask"])
        strain = True if g["stress"] else None
        if sweep == 1:
            values = self._first(tile, g["state"])
            names = ("es", "x", "sc")
        elif sweep == 2:
            es, g_vec, g_x, g_sc, g_eps = self._second(tile, g["state"], strain)
            if not self._speed(TILE_EPILOGUE):  # else its epilogue masked the edge gradient
                g_vec = self._masked(g_vec, slot["edge_mask"])
            values = (es, g_vec, g_x, g_sc) + ((g_eps,) if strain else ())
            names = ("es", "g_vec", "g_x", "g_sc", "g_eps")[:len(values)]
        else:
            _, g_vec, g_eps = self._first(tile, g["state"], adjoint=True, strain=strain)
            if not self._speed(TILE_EPILOGUE):
                g_vec = self._masked(g_vec, slot["edge_mask"])
            values = (g_vec,) + ((g_eps,) if strain else ())
            names = ("g_vec", "g_eps")[:len(values)]
        if not self._captured():  # fixed addresses only for the graphs' replays: this sweep's outputs themselves,
            slot["out"] = {sweep: dict(zip(names, values))}  # no copy, and no other sweep's kept beside them
            return
        # the graphs' outputs apart from an uncaptured call's (which replaces ``slot["out"]``): their addresses stay
        out = slot.setdefault("graph_out", {}).setdefault(sweep, {})
        for name, value in zip(names, values):
            if name not in out:
                out[name] = torch.empty_like(value)
            out[name].copy_(value)
        slot["out"] = {sweep: out}

    def _scatter(self, sweep, slot, g, piece, host_nodes=None):
        """Add the slot's outputs into the global buffers (outside the graphs: warm-ups and captures have no side
        effects). Padding rows read atom 0 and add exact zeros to it; with the state aliased, only the tile's real
        rows are written (dE/dsc overwrites sc, and dE/dx sums in the window of the tile's own atoms)."""
        out = slot["out"][sweep]
        receivers = slot["sizes"][0]
        recv = slot["nodes"][:receivers]
        state = g["state"]
        if sweep == 1:
            state = self._state(g, out["x"], out["sc"])
            g["node_es"].index_add_(0, recv, out["es"])
            if state.pooled:  # only the real receivers have pooled rows (their x and sc, written once)
                r = piece[4]
                index = recv[:r] if host_nodes is None else host_nodes[:r]  # host nodes: no sync for the pool rows
                state.put("x", index, out["x"][:r])
                state.put("sc", index, out["sc"][:r])
                return
            state.add("x", recv, out["x"])
            state.add("sc", recv, out["sc"])
            return
        if sweep == 2 and state.alias:
            import torch

            nodes, r, h = piece[0], piece[4], piece[5]
            rows = torch.cat([torch.arange(r, device=self.device),
                              torch.arange(receivers, receivers + h, device=self.device)])
            g["node_es"].index_add_(0, recv, out["es"])
            state.add("gx", nodes, out["g_x"][rows])
            state.put("gsc", nodes[:r] if host_nodes is None else host_nodes[:r], out["g_sc"][:r])
        elif sweep == 2:
            g["node_es"].index_add_(0, recv, out["es"])
            state.add("gx", slot["nodes"], out["g_x"])
            state.add("gsc", recv, out["g_sc"])
        ei = slot["ei"]
        g["forces"].index_add_(0, ei[1], out["g_vec"], alpha=-1)
        g["forces"].index_add_(0, ei[0], out["g_vec"])
        if "g_eps" in out:
            g["strain_grad"].add_(out["g_eps"].double())

    def _captured(self):
        """Whether the slots' sweep bodies run as captured CUDA graphs: mode "graphs" (eager bodies), or compiled
        bodies with ``TileConfig.compiled_graphs``."""
        return self.mode == "graphs" or (self.mode == "compiled" and self._graphs_wanted() and self._graphs_fit is True)

    def _graphs_wanted(self):
        """Whether ``TileConfig.compiled_graphs`` asks for captured compiled bodies (before the memory guard): True,
        or None on ``COMPILED_GRAPH_GPUS`` with a configuration that allows them."""
        config = self.config
        if config.compiled_graphs is None:
            return (getattr(self, "_graph_gpu", False) and config.state == "device" and config.lanes != "on"
                    and not config.alias_state and config.mode in ("compiled", "auto"))
        return bool(config.compiled_graphs)

    def _choose_graphs(self, memory, slots, measured_receivers):
        """The memory guard of ``TileConfig.compiled_graphs`` after the first compiled call (uncaptured): on when
        ``slots`` capture pools of its tile workspace (scaled to the cap the budget chose) and the warm-ups' fit
        ``available`` (``compiled_graphs_fit``), else off for the field's life."""
        if memory is None:
            self._graphs_fit, self.graphs_state = False, "off: no device memory measurement"
            return
        workspace, available = memory
        workspace = workspace * self.receivers / max(1, int(measured_receivers))
        self._graph_memory = (workspace, available, int(slots))
        self._graphs_fit = compiled_graphs_fit(workspace, available, slots)
        verdict = "fit" if self._graphs_fit else "do not fit"
        self.graphs_state = (f"{'on' if self._graphs_fit else 'off'}: {int(slots)} capture pools and the warm-ups of "
                             f"{workspace / 2**30:.2f} GiB (x{COMPILED_GRAPH_MARGIN}) {verdict} "
                             f"{available / 2**30:.2f} GiB")

    def _graphs_slots_fit(self, slots):
        """A captured call with more slots than the guard measured: whether their pools still fit (else graphs off,
        the captured ones dropped)."""
        if self._graph_memory is None or slots <= self._graph_memory[2]:
            return True
        workspace, available, _ = self._graph_memory
        if compiled_graphs_fit(workspace, available, slots):
            self._graph_memory = (workspace, available, int(slots))
            return True
        self._drop_graphs(f"off: {int(slots)} slots' capture pools of {workspace / 2**30:.2f} GiB do not fit "
                          f"{available / 2**30:.2f} GiB")
        return False

    def _drop_graphs(self, reason):
        """``TileConfig.compiled_graphs`` off for the field's life: the captured graphs, their outputs and pool
        released."""
        import torch

        self._graphs_fit, self.graphs_state = False, reason
        for slot in self._slots.values():
            slot["graphs"].clear()
            slot.pop("graph_out", None)
            slot.pop("out", None)
        self._pool = None
        self._graph_key = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _run_slot(self, sweep, slot, g):
        """The sweep's body on the filled slot: eagerly, or by replaying its graph (captured on first use)."""
        import torch

        if not self._captured():
            self._body(sweep, slot, g)
            return
        graph = slot["graphs"].get(sweep)
        if graph is None:
            from fastmace import _patches

            _patches.set_capture_safe_mask_head(True)  # MACE's readout mask assigns by index (as GraphEngine)
            if self._stream is None:
                self._stream = torch.cuda.Stream()
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                for _ in range(2):  # library workspaces and the allocator, outside the capture
                    self._body(sweep, slot, g)
            torch.cuda.current_stream().wait_stream(self._stream)
            if self._pool is None or not any(s["graphs"] for s in self._slots.values()):
                # a new pool once no graph holds the old one (PyTorch releases a private pool with its last graph,
                # and a capture into the released handle fails an allocator assertion)
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool, stream=self._stream):
                self._body(sweep, slot, g)
            slot["graphs"][sweep] = graph
        graph.replay()
        slot["out"] = {sweep: slot["graph_out"][sweep]}

    # ------------------------------------------------------------------ the call

    def _bind(self, count=None):
        field = self.field
        # species tables; then every tile edge is a real edge. No whole-system BLOCK_ORDER order: a tile has
        # fewer rows than N, so its kernels never read it (they launch with their rows' identity)
        field._bind_streaming_count(block_order=bool(TILE_WHOLE_BLOCK_ORDER))
        saved = []
        for module in field._streaming:
            saved.append((module, module._buffers.get("streaming_count")))
            module.register_buffer("streaming_count", self._big if count is None else count, persistent=False)
        return saved

    def __call__(self, positions):
        """Energy [1] (float64), forces [N, 3] and stress [3, 3] or None at float64 ``positions``.

        A call that runs out of device memory at a receiver cap the budget raised plans again at the next lower cap
        of ``(receivers, *budget_receivers)`` and runs again; the lower cap stays (the budget's measurement is the
        first, eager call's, and a compiled tile's workspace can exceed it: tiled-omat.md)."""
        import torch

        probe = None
        if self._probe_wanted():
            self._probe = max(1, int(TILE_PROBE_TILES))
        while True:
            fallback = None
            try:
                out = self._call(positions)
                if getattr(self, "_probe", 0):
                    self._probe = 0
                    if "probe" in self.last:  # its tiles measured, the mode and cap chosen: the call again
                        del out
                        probe = self._after_probe()
                        continue
                if probe is not None:
                    self.last["probe"] = probe
                if self._fallback is not None:
                    self.last["budget_fallback"] = self._fallback
                return out
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error)
                lower = [int(r) for r in (self.config.receivers, *self.config.budget_receivers)
                         if int(r) < self.receivers]
                graphs = self.mode == "compiled" and self._captured()
                lanes = self.lanes_state == "on" and self.config.lanes == "auto"
                if not (oom and (graphs or lanes or (self._budget_raised and lower))):
                    raise
                fallback = (self.receivers, self.receivers if graphs or lanes else max(lower),
                            str(error).splitlines()[0][:200], graphs)
            # Outside the handler, so the failed call's frames (and their tensors) are released.
            old, new, why, graphs = fallback
            if graphs:  # TileConfig.compiled_graphs goes first: the same call again without captures
                self._drop_graphs(f"off: out of device memory in a captured call ({why})")
                self._g = None
                self._list_cache = None
                self._fallback = f"compiled graphs off after running out of device memory ({why})"
                continue
            if new == old:  # the lanes go first: one tile workspace again at the same cap
                self._lanes_fit = False
                self._lane_slots.clear()
                self._g = None
                self._list_cache = None
                self._fallback = f"lanes off after running out of device memory ({why})"
                torch.cuda.empty_cache()
                continue
            self.receivers = new
            self._budget_raised = new > int(self.config.receivers)
            self._slots.clear()
            self._lane_slots.clear()
            self._edge_step = None
            self._graph_key = None
            self._g = None  # the global buffers and the state are allocated again, zeroed
            self._list_cache = None
            self._fallback = f"receiver cap {old} -> {new} after running out of device memory ({why})"
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _probe_wanted(self):
        """``TILE_FIRST_CALL_PROBE`` for this call: mode "auto"'s measuring first call (a compiled force field's)."""
        if not getattr(self, "_auto", False) or TILE_FIRST_CALL_PROBE is False:
            return False
        if TILE_FIRST_CALL_PROBE:
            return True
        if self.device.type != "cuda":
            return False
        import torch

        return torch.cuda.get_device_name(self.device) in TILE_PROBE_GPUS

    def _after_probe(self):
        """``TILE_FIRST_CALL_PROBE``: the probe's report; its call's buffers and partial kept lists dropped (the call
        again allocates and searches anew)."""
        import torch

        last = self.last
        probe = dict(last["probe"], mode=self.mode, mode_reason=self.mode_reason, receivers=self.receivers,
                     search_s=last.get("search_s"), tiles_s=last.get("tiles_s"))
        if "tiles_peak" in last:
            probe["workspace"] = last["tiles_peak"] - last["loop_base"]
        _release_kept(self._kept)
        self._kept, self._reusing = None, False
        self._g = None
        self._list_cache = None
        self._pieces = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return probe

    def _call(self, positions):
        import torch

        field = self.field
        data = field.data
        stress = field.compute_stress
        padded = self.mode != "eager"
        with torch.no_grad():
            if _tile_list().rows_on_host(positions, self.device):  # a chunk at a time: no float64 copy on the device
                for start in range(0, positions.shape[0], _tile_list().CHUNK):
                    stop = start + _tile_list().CHUNK
                    field.positions[start:stop].copy_(positions[start:stop].to(self.device))
            else:
                field.positions.copy_(positions)
        pos = field.positions.detach()
        n = pos.shape[0]
        self.node_attrs = data["node_attrs"] if self._species is None else None
        batch = data["batch"]
        if batch.dim() == 1 and batch.stride(0) == 0:  # one structure (release_whole_system): one head for all
            heads = (data["head"].reshape(-1)[:1] if "head" in data
                     else torch.zeros(1, dtype=torch.int64, device=batch.device)).long().expand(n)
        else:
            heads = (data["head"][batch] if "head" in data else torch.zeros_like(batch)).long()
        source, row_ptr, degree = self._source(positions, n)
        caps = self.config
        edge_cap = caps.edges if caps.edges is not None else (
            2**62 if degree is None else max(1, int(math.ceil(2 * self.receivers * degree))))
        cell = data["cell"].reshape(3, 3) if "cell" in data else torch.as_tensor(field.neighbours.cell)
        if self._reusing:  # the kept plan, back on the device for this call
            tiles, grid = [t.to(self.device) for t in self._kept["tiles"]], self._kept["grid"]
        else:
            reach, layers = self._boundary_reach(n)
            shape = self._fitted_grid(pos, cell, n, reach, layers)
            tiles, grid = plan_tiles(pos, cell, row_ptr, self.receivers, edge_cap, reach, layers, shape=shape)
            split = _grid_shape(cell.detach().cpu(), n, self.receivers, reach, layers, k=shape)[1] > 0
            self._plan_skin = reach if split else 0.0
            if self._kept is not None:
                self._kept["tiles"], self._kept["grid"] = [t.cpu() for t in tiles], grid
        self._tile_index = {t.data_ptr(): i for i, t in enumerate(tiles)}
        local_of = torch.full((n,), -1, dtype=torch.int32, device=self.device)  # int32: 4 bytes per atom
        # The first padded (compiled) call also measures: the lanes' fit is its own tile workspace, not the eager
        # first call's scaled to the cap (eager tiles hold several times more: R 16,384 was refused with 12 GiB free).
        lanes_unmeasured = self._lanes_fit is None and padded and not self._captured()
        # TileConfig.compiled_graphs: the first compiled call runs uncaptured and measures (the memory guard)
        graphs_unmeasured = self.mode == "compiled" and self._graphs_fit is None and self._graphs_wanted()
        if graphs_unmeasured:
            self.graphs_state = "measuring: the first compiled call runs uncaptured"
        graphs_state = self.graphs_state if self._graphs_wanted() else (
            "off" if self.config.compiled_graphs is not None else
            "off: not measured faster on this GPU or configuration" if self.mode == "compiled" else None)
        measure = (not self._budgeted or self._auto or lanes_unmeasured or graphs_unmeasured) \
            and self.device.type == "cuda"
        if measure:  # the workspace of this call: its peak above what stays allocated after it
            torch.cuda.reset_peak_memory_stats(self.device)
        stats = dict(tiles=len(tiles), grid=grid, edge_cap=edge_cap, max_receivers=0, max_halo=0, max_edges=0,
                     list_reused=self._reusing, list_rebuilds=self.list_rebuilds,
                     list_reuse=self.list_reuse_state,
                     receivers=self.receivers,
                     edges=0, mode=self.mode, mode_reason=self.mode_reason, neighbours=caps.neighbours,
                     species=caps.species)
        wave = self._wavefront_on(grid, n)
        self._graph_budget(wave, measure)
        started = time.perf_counter()  # the search pass before the sweeps (its syncs make the wall time its cost)
        if wave:
            keys, steps, pool_rows, first, last = self._wavefront(tiles, source, local_of, n, padded)
        else:
            keys = self._slot_keys(tiles, source, local_of, n) if padded else None
            steps, pool_rows = [(sweep, index) for sweep in (1, 2, 3) for index in range(len(tiles))], 0
        stats["search_s"] = time.perf_counter() - started if padded or wave else 0.0
        if self._probe and wave and not padded:  # TILE_FIRST_CALL_PROBE: the largest tiles' sweeps only
            sizes = [int(t.shape[0]) for t in tiles]
            chosen = set(sorted(range(len(tiles)), key=lambda i: (-sizes[i], i))[:self._probe])
            steps = [kept_item for kept_item in ((item[0], *[i for i in item[1:] if i in chosen]) for item in steps)
                     if len(kept_item) > 1]
            stats["probe"] = dict(tiles=sorted(chosen), steps=len(steps), largest=max(sizes, default=0))
        lanes = self._lanes_on(padded, n)
        if lanes:  # sweep 2 in pairs: its sweep-3 steps between a pair's tiles wait for the pair (a larger pool)
            steps, lane_peak = lane_steps(steps, [int(t.shape[0]) for t in tiles])
            pool_rows = max(pool_rows, lane_peak) if wave else pool_rows
        stats["lanes"] = self.lanes_state
        stats["wavefront"] = dict(on=wave, pool_rows=pool_rows, pool_fraction=pool_rows / max(n, 1),
                                  boundary_skin=self._plan_skin)  # the split's reach (Angstrom), 0: none
        if padded:
            stats["slots"] = {key: self._slots[key]["sizes"] for key in sorted(set(keys))}
            stats["slot_needs"] = {key: self.slot_needs[key] for key in sorted(self.slot_needs)}
            stats["slot_changes"] = self.slot_changes
        if not self._released and self.device.type == "cuda":
            # The force field's construction leaves its peak in PyTorch's cache; libraries that allocate outside
            # it (cuEquivariance's cudaMallocAsync) would find the device full in the first tile.
            torch.cuda.empty_cache()
        self._released = True
        saved = self._bind()
        bound = None
        try:
            g = self._globals(n, stress, pos, heads)
            g["local_frame"], g["frame"] = (self._frames(positions, n) if self._frame_wanted(positions, n)
                                            else (None, None))
            stats["local_frame"] = ("tiles" if g["local_frame"] is not None else
                                    "bins" if g["frame"] is not None else None)
            self.heads, self.arange = g["heads"], g["arange"]
            if padded and self.mode == "compiled" and self._captured():  # the guard: pools for every slot in use
                self._graphs_slots_fit(len(set(keys)))
                graphs_state = self.graphs_state
            stats["compiled_graphs"] = graphs_state
            if padded and self._captured():  # (a call without graphs leaves the captured ones alone)
                species = self.node_attrs if self._species is None else self._species
                # (and the body's edge-vector form: TILE_LOCAL_FRAME is read per call)
                frame = g.get("local_frame")
                key = (id(g), pos.data_ptr(), species.data_ptr(), stress, None if frame is None else frame[0].data_ptr())
                if key != self._graph_key:  # the graphs read these addresses
                    for slot in self._slots.values():
                        slot["graphs"].clear()
                    self._graph_key = key
            forces, node_es, strain_grad, state = g["forces"], g["node_es"], g["strain_grad"], g["state"]
            aliased = self.config.alias_state
            state.worker = state.overlap and bool(self._speed(TILE_HOST_WORKER))
            aside = state.overlap and self.device.type == "cuda" and bool(self._speed(TILE_COPY_STREAM))
            if aside and self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(self.device)
            state.copy_stream = self._copy_stream if aside else None
            stats["copy_stream"] = aside
            stats["host_worker"] = state.worker
            if aliased:
                state.pool(pool_rows, n)
                if wave:
                    state.touched(first, last)
                    del first, last
            planned, previous = False, None
            if measure:  # the tiles' workspace alone: the per-atom buffers of the call stay out of it (see _memory)
                stats["loop_base"] = int(torch.cuda.memory_allocated(self.device))
                stats["peak_before_tiles"] = int(torch.cuda.max_memory_allocated(self.device))
                torch.cuda.reset_peak_memory_stats(self.device)
            streams = self._lane_streams[0] if lanes and self.device.type == "cuda" else None
            lane_strain = [torch.zeros_like(strain_grad) for _ in range(2)] if lanes else None

            # TileConfig.prefetch: the next tile's graph (and, within a run of sweep 2 or 3, its host rows) on a side
            # stream; the tiles' receivers on the host (the pool's assign and release index with them, no sync).
            ahead = self.config.prefetch and padded and (state.host or self._speed(TILE_PREFETCH_DEVICE))
            prep = None
            if ahead and self.device.type == "cuda":
                if self._prep_stream is None:
                    self._prep_stream = torch.cuda.Stream(self.device)
                prep = self._prep_stream
                prep.wait_stream(torch.cuda.current_stream())  # the call's positions, bins and plan
            host_tiles = [t.cpu() for t in tiles] if ahead and wave else None
            order = [(item[0], index, (k if len(item) == 3 or (lanes and item[0] == 2) else None))
                     for item in steps for k, index in enumerate(item[1:])]
            prepared = {}

            def prepare(position):
                """Build tile ``order[position]``'s graph ahead; fetch its rows when the sweep before it is its own."""
                sweep, index, lane = order[position]
                slot = self._lane_slot(keys[index]) if lane == 1 else self._slots[keys[index]]
                # (device state, TILE_PREFETCH_DEVICE: the graph only; the body reads its rows)
                rows = state.host and position > 0 and order[position - 1][0] == sweep and sweep in (2, 3)
                with torch.cuda.stream(prep) if prep is not None else contextlib.nullcontext():
                    piece = self._piece(index, tiles[index], source, local_of, slot["sizes"][0])
                    nodes = piece[0].cpu()  # waits for the prefetch stream only
                    fetched = {}
                    if rows:  # the slot's layout (_fill): receivers, then the halo after the receiver slots
                        size, halo = slot["sizes"][0], slot["sizes"][1]
                        r, h = piece[4], piece[5]
                        layout = torch.zeros(size + halo, dtype=torch.int64)
                        layout[:r] = nodes[:r]
                        layout[size:size + h] = nodes[r:]
                        names = (("x", layout), ("sc", layout[:size])) if sweep == 2 else (
                            ("gx", layout[:size]), ("gsc", layout[:size]))
                        fetched = {name: state.fetch(name, rows_) for name, rows_ in names}
                    event = None
                    if prep is not None:
                        event = torch.cuda.Event()
                        event.record(prep)
                prepared[position] = (piece, nodes, fetched, event)

            def tile(sweep, index, lane=None, position=None):
                """One sweep on one tile; ``lane`` 0 or 1: sweep 2 of a pair, on that lane's stream and slots."""
                nonlocal bound
                receivers = tiles[index]
                if sweep == 1 and wave:
                    state.assign(receivers if host_tiles is None else host_tiles[index])
                state.new_tile()
                slot = self._lane_slot(keys[index]) if lane == 1 else self._slots[keys[index]] if padded else None
                ready = prepared.pop(position, None)
                host_nodes, fetched = None, {}
                if ready is None:
                    piece = self._piece(index, receivers, source, local_of, slot["sizes"][0] if padded else None)
                else:
                    piece, host_nodes, fetched, event = ready
                    if event is not None:
                        torch.cuda.current_stream().wait_event(event)
                        for tensor in piece[:4]:  # made on the prefetch stream
                            tensor.record_stream(torch.cuda.current_stream())
                self._note(stats, piece)
                if sweep == 1:
                    stats["edges"] += piece[1].shape[1]
                    if aliased and not wave:
                        state.touch(piece[0] if host_nodes is None else host_nodes, index, n)
                if padded:
                    if bound is not slot:  # the streaming kernels read this slot's edge count and tables
                        for module in field._streaming:
                            module.register_buffer("streaming_count", slot["count"], persistent=False)
                        self._bind_tables(slot["tables"])
                        bound = slot
                    if slot.get("event") is not None:  # a lane still reads this slot's buffers
                        torch.cuda.current_stream().wait_event(slot["event"])
                        slot["event"] = None
                    self._fill(slot, piece, g, host_nodes)
                    state.prefetched = fetched  # rows() takes these instead of gathering them now
                    following = position is not None and position + 1 < len(order)
                    if lane is None:
                        self._run_slot(sweep, slot, g)
                        if following:
                            prepare(position + 1)  # while this tile runs (its scatter only queues)
                        self._scatter(sweep, slot, g, piece, host_nodes)
                    else:  # the lane's own strain gradient (a plain add_); forces and node energies add atomically
                        lane_g = dict(g, strain_grad=lane_strain[lane])
                        if streams is None:  # off a CUDA device: the lanes one after the other
                            self._run_slot(sweep, slot, lane_g)
                            if following:
                                prepare(position + 1)
                            self._scatter(sweep, slot, lane_g, piece, host_nodes)
                        else:
                            stream = streams[lane]
                            stream.wait_stream(torch.cuda.current_stream())  # the slot's filled buffers
                            with torch.cuda.stream(stream):
                                for tensor in fetched.values():
                                    tensor.record_stream(stream)
                                self._run_slot(sweep, slot, lane_g)
                            if following:
                                prepare(position + 1)
                            with torch.cuda.stream(stream):
                                self._scatter(sweep, slot, lane_g, piece, host_nodes)
                                slot["event"] = torch.cuda.Event()
                                slot["event"].record(stream)
                            for tensor in piece[:4]:  # read by the lane's scatter
                                tensor.record_stream(stream)
                    state.prefetched = {}
                else:
                    self._eager_tile(sweep, piece, g)
                if sweep == 2 and aliased:
                    state.close(index)
                del piece
                if sweep == 3 and wave:  # its reads are done (rows() copies on the host before it returns)
                    state.release(receivers if host_tiles is None else host_tiles[index])
                    if self._pieces is not None:  # TILE_GRAPH_CACHE_WAVE: no sweep reads this tile's graph again
                        self._pieces.pop(index, None)
                    if self._list_cache is not None:
                        self._list_cache.pop(receivers.data_ptr(), None)  # no sweep reads this tile's lists again

            loop_started = time.perf_counter()
            position = 0
            for item in steps:
                sweep = item[0]
                if sweep != previous and sweep != 1:
                    state.flush()  # sweep 2 reads what sweep 1 wrote, sweep 3 what sweep 2 wrote (sweep 1 reads none)
                if sweep == 2 and aliased and not planned:
                    state.plan(len(tiles))  # after the touches (sweep 1, or the wavefront's search pass)
                    planned = True
                if streams is not None and sweep != 2 and previous == 2:  # the whole device again: after the lanes
                    for stream in streams:
                        torch.cuda.current_stream().wait_stream(stream)
                previous = sweep
                for k, index in enumerate(item[1:]):
                    lane = k if len(item) == 3 or (lanes and sweep == 2) else None
                    tile(sweep, index, lane=lane, position=position if ahead else None)
                    position += 1
            if streams is not None:
                for stream in streams:
                    torch.cuda.current_stream().wait_stream(stream)
            if lanes:
                for part in lane_strain:
                    strain_grad.add_(part)
            state.flush()
            stats["tiles_s"] = time.perf_counter() - loop_started  # the host's time in the sweeps (queued work aside)
            stats["graph_cache"] = dict(tiles=self._piece_count, bytes=self._piece_bytes,
                                        budget=self._piece_budget)
            self._pieces = None
            if measure:
                stats["tiles_peak"] = int(torch.cuda.max_memory_allocated(self.device))
            source = None  # the tile lists' cell bins (per-atom device arrays) before the energy's transients
            stats["global_bytes"] = state.bytes()
            stats["heap_trims"], stats["heap_trim_s"] = HEAP_TRIMS[0], round(HEAP_TRIMS[1], 3)
            list_cache = getattr(self, "_list_cache", None)
            stats["list_cache_bytes"] = int(sum(t.numel() * t.element_size() for entry in (list_cache or {}).values()
                                                for t in entry))
            kept = self._kept
            stats["list_kept_bytes"] = 0 if kept is None else int(
                sum(t.numel() * t.element_size() for entry in kept["superset"].values() for t in entry)
                + kept["reference"].numel() * kept["reference"].element_size())
            # TILE_PINNED_SUPERSET: the pinned allocator's blocks of the kept lists (powers of two)
            stats["list_kept_pinned_bytes"] = 0 if kept is None else int(kept.get("pinned_bytes", 0))
            stats["list_cache_hits"] = getattr(self, "_list_hits", 0)  # tile visits served from the call's cache
            stats["list_reuse"] = self.list_reuse_state
            stats["list_cache"] = self.list_cache_state
            self._list_cache = None
            stats["host_bytes"] = state.bytes() if state.host else 0
            stats["state_dtype"] = str(state.store).replace("torch.", "") if state.host else None
            stats["state_dtype_reason"] = self.state_dtype_state
            with torch.no_grad():
                node_inter = self.model.scale_shift(node_es, self.heads[:n])
                from fastmace.model import energy as energy64

                if energy64.NODE_E0_KEY in data:
                    e0 = field.node_e0()  # the stored column, or looked up from the species index
                else:  # a float64 model summed natively: its own table
                    rows = torch.arange(n, device=self.device)
                    e0 = self.model.atomic_energies_fn(self.node_attrs)[rows, self.heads[:n]].double()
                energy = (e0 + node_inter.double()).sum().reshape(1)
                out_stress = None
                if stress:
                    volume = torch.linalg.det(cell.to(torch.float64)).abs()
                    out_stress = (strain_grad / volume).to(self.dtype)  # MACE: dE/d(strain) / V
            self.last = stats
            out = ForceOutput(energy, forces, out_stress)
            # Handed out: the caller holds the forces, not the call (else a caller that drops them still has them
            # on the device through the next call, beside its new buffer: 12 bytes per atom of that call's peak).
            g["forces"] = None
            memory = None
            if self._auto or not self._budgeted:
                memory = self._memory()  # None without a CUDA device: the mode and the cap stay
                self._choose_mode(memory)
                if not self._budgeted:
                    self._budgeted = True
                    if self.config.mode != "auto" or self.mode == "compiled":  # "auto": eager tiles take no budget
                        self._choose_receivers(memory)
            elif lanes_unmeasured and measure:  # TileConfig.lanes: a second workspace of this padded call's
                memory = self._memory()
                workspace, available = memory
                self._lanes_fit = 2 * 1.1 * workspace <= available
            if graphs_unmeasured and measure:  # TileConfig.compiled_graphs: the guard, from this call's workspace
                self._choose_graphs(memory if memory is not None else self._memory(), len(stats["slots"]),
                                    stats["receivers"])
                stats["compiled_graphs"] = self.graphs_state
            return out
        finally:
            for module, count in saved:
                if count is not None:
                    module.register_buffer("streaming_count", count, persistent=False)

    def _lanes_on(self, padded, n):
        """Whether this call runs sweep 2 on two lanes (``TileConfig.lanes``), the reason in ``lanes_state``."""
        import torch

        config = self.config.lanes
        if config == "off":
            self.lanes_state = "off"
            return False
        if self.config.alias_state and self.config.state == "device" and self.device.type == "cuda":
            self.lanes_state = "off: the aliased device state reads its window on the call's stream"
            return False
        if not padded or self._captured():
            self.lanes_state = f"off: {self.mode} tiles" + (" in CUDA graphs" if self.mode == "compiled" else "")
            return False
        if self.device.type != "cuda":
            self.lanes_state = "on: one after the other (no CUDA device)" if config == "on" else "off: no CUDA device"
            return config == "on"
        if config == "auto":
            name = torch.cuda.get_device_name(self.device)
            if name not in LANE_GPUS:
                self.lanes_state = f"off: not measured faster on {name}"
                return False
            if not self._lanes_fit:
                self.lanes_state = ("off: no memory measurement" if self._lanes_fit is None else
                                    "off: a second tile workspace does not fit the device")
                return False
        backend = torch.cuda.get_allocator_backend()
        if backend != "cudaMallocAsync":
            reason = (f"needs PyTorch's cudaMallocAsync allocator backend (PYTORCH_CUDA_ALLOC_CONF="
                      f"backend:cudaMallocAsync before CUDA starts), not {backend}")
            if config == "on":
                raise ValueError(f"TileConfig.lanes='on' {reason}")
            self.lanes_state = f"off: {reason}"
            return False
        if self._lane_streams is None:
            try:
                self._lane_streams = _split_streams(self.device)
            except Exception as error:  # noqa: BLE001  (no green contexts here: off, with the driver's reason)
                if config == "on":
                    raise
                self._lane_streams = False
                self.lanes_state = f"off: green contexts: {type(error).__name__}: {error}"[:300]
        if self._lane_streams is False:
            return False
        self.lanes_state = "on"
        return True

    def _lane_slot(self, key):
        """The second lane's padded slot of edge bucket ``key``: the first lane's sizes, its own buffers."""
        sizes = self._slots[key]["sizes"]
        slot = self._lane_slots.get(key)
        if slot is None or slot["sizes"] != sizes:
            slot = self._lane_slots[key] = self._new_slot(sizes)
        return slot

    def _memory(self):
        """``(workspace, available)`` bytes after a measured call, or None without a CUDA device: the call's peak
        above what stays allocated, and the device's free memory plus the allocator's cached, unallocated
        blocks."""
        import torch

        if self.device.type != "cuda":
            return None
        free, _ = torch.cuda.mem_get_info(self.device)
        last = self.last
        if "tiles_peak" in last:
            # The tiles' peak above what was allocated when they started (the call's per-atom buffers: tile lists,
            # local indices, forces), and what was left beside those buffers. The call's own transients outside the
            # tiles (the energy's float64 sums, the wavefront's search pass) scale with the atoms, not the cap: the
            # first call held them at this atom count, apart from the tiles' workspace (tiled-wavefront.md).
            base = last["loop_base"]
            memory = last["tiles_peak"] - base, free + torch.cuda.memory_reserved(self.device) - base
        else:
            allocated = torch.cuda.memory_allocated(self.device)
            memory = (torch.cuda.max_memory_allocated(self.device) - allocated,
                      free + torch.cuda.memory_reserved(self.device) - allocated)
        self._per_receiver = memory[0] / self.receivers  # at this call's cap (TILE_GRAPH_CACHE's budget)
        return memory

    def _choose_mode(self, memory):
        """Mode "auto" after a compiled force field's first (eager) call: compiled tiles when the memory left at
        the call's peak (``available`` less the call's workspace) holds their residency and a margin, else eager.
        Chosen once."""
        if not self._auto:
            return
        self._auto = False
        if memory is None:
            self.mode_reason = "no device memory measurement: eager"
            return
        workspace, available = memory
        headroom = available - workspace
        need = COMPILED_RESIDENCY + COMPILED_MARGIN
        verdict = ">=" if headroom >= need else "<"
        self.mode = "compiled" if headroom >= need else "eager"
        self.mode_reason = (f"{headroom / 2**30:.2f} GiB left at the first call's peak {verdict} the compiled "
                            f"residency and margin ({need / 2**30:.2f} GiB)")

    def _choose_receivers(self, memory):
        """``budget_receivers``: the largest cap whose predicted workspace (the measured workspace per receiver,
        times the cap, plus 10 %) fits ``available``; the padded slots are planned again for it."""
        if memory is None:
            return
        workspace, available = memory
        per_receiver = workspace / self.receivers
        candidates = (self.receivers, *map(int, self.config.budget_receivers))
        host = _host_available() if any(r in TILE_LARGE_CAP_RECEIVERS for r in candidates) else None
        floor = int(self.config.list_reuse_floor)

        def host_fits(r):  # TILE_LARGE_CAP_HOST_BYTES: the large caps' host buffers
            if r not in TILE_LARGE_CAP_RECEIVERS or r <= self.receivers or host is None:
                return True
            return host - floor >= 1.1 * TILE_LARGE_CAP_HOST_BYTES * (r - self.receivers)

        fits = [r for r in candidates if 1.1 * per_receiver * r <= available and host_fits(r)]
        chosen = max(fits, default=self.receivers)
        if chosen != self.receivers:
            self._budget_raised = chosen > int(self.config.receivers)
            self.receivers = chosen
            self._slots.clear()  # their sizes and edge buckets were for the old cap
            self._lane_slots.clear()
            self._edge_step = None
            self._graph_key = None

    def _frame_wanted(self, positions, n):
        """``TILE_LOCAL_FRAME`` for this call: None decides by the structure's extent (``forces.float64_edges_wanted``),
        once per atom count and cell (one host read of the positions' largest |coordinate|)."""
        import numpy as np

        from fastmace.model import forces as forces_module

        if TILE_LOCAL_FRAME is not None:
            return bool(TILE_LOCAL_FRAME)
        if str(self.dtype) == "torch.float64":  # float64 positions: nothing to fix
            return False
        cell = getattr(self.field.neighbours, "cell", None)
        if cell is None:
            cell = self.field.data["cell"].reshape(3, 3).detach().cpu().numpy()
        cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        key = (n, cell.tobytes())
        kept = getattr(self, "_frame_choice", None)
        if kept is None or kept[0] != key:
            largest = _tile_list().abs_max(positions) if n else 0.0
            extent = max(largest, forces_module.structure_extent(np.zeros((0, 3)), cell))
            kept = self._frame_choice = (key, forces_module.float64_edges_wanted(None, extent))
        return kept[1]

    def _frames(self, positions, n):
        """``TILE_LOCAL_FRAME``: ``(local_frame, frame)``, one of them set. Tile-local positions (``(positions64,
        cell, inverse)``) where the call's float64 positions are on the device (no stored bytes), else the bins and
        offsets of ``_local_frame``; neither without a full periodic cell."""
        import torch

        cell = getattr(self.field.neighbours, "cell", None)
        if cell is None:
            cell = self.field.data["cell"].reshape(3, 3)
        cell = torch.as_tensor(cell, dtype=torch.float64).reshape(3, 3)
        if abs(float(torch.linalg.det(cell))) < 1e-9:
            return None, None
        if _tile_list().rows_on_host(positions, self.device) and not TILE_HOST_ROWS:
            return None, self._local_frame(positions, n)
        kept = getattr(self, "_frame_cell", None)  # fixed addresses: the captured tile bodies read them
        if kept is None or not torch.equal(kept[0].cpu(), cell):
            device_cell = cell.to(self.device)
            kept = self._frame_cell = (cell.clone(), device_cell, torch.linalg.inv(device_cell))
        # TILE_HOST_ROWS: host positions stay there; each slot gets its rows (``_fill``), an eager tile gathers them
        return (positions if positions.dtype == torch.float64 else positions.to(torch.float64), kept[1], kept[2]), None

    def _local_frame(self, positions, n):
        """``TILE_LOCAL_FRAME``: the call's positions as int16 bins of ``TILE_LOCAL_FRAME_WIDTH`` and offsets in the
        field's dtype (kept between calls), from the float64 ``positions`` a chunk at a time (no whole float64 copy
        on the device), with the float64 cell and its inverse."""
        import torch

        width = float(TILE_LOCAL_FRAME_WIDTH)
        cell = getattr(self.field.neighbours, "cell", None)
        if cell is None:
            cell = self.field.data["cell"].reshape(3, 3)
        cell = torch.as_tensor(cell, dtype=torch.float64).reshape(3, 3)
        if abs(float(torch.linalg.det(cell))) < 1e-9:  # no full periodic cell: the plain float32 vectors
            return None
        kept = getattr(self, "_frame_buffers", None)
        if kept is None or kept[0].shape[0] != n:  # fixed addresses between calls (the CUDA graphs read them)
            kept = (torch.empty((n, 3), dtype=torch.int16, device=self.device),
                    torch.empty((n, 3), dtype=self.dtype, device=self.device),
                    torch.empty((3, 3), dtype=torch.float64, device=self.device),
                    torch.empty((3, 3), dtype=torch.float64, device=self.device))
            self._frame_buffers = kept
        bins, offsets, cell_kept, inverse_kept = kept
        cell_kept.copy_(cell)
        inverse_kept.copy_(torch.linalg.inv(cell))
        chunk = 1 << 20
        for start in range(0, n, chunk):
            b, o = _frame_parts(positions[start:start + chunk].to(device=self.device, dtype=torch.float64), width,
                                self.dtype)
            bins[start:start + chunk] = b
            offsets[start:start + chunk] = o
        return bins, offsets, cell_kept, inverse_kept, width

    def _globals(self, n, stress, pos, heads):
        """The call's global buffers, kept between calls at fixed addresses (the graphs read them) and zeroed.

        A new set (another atom count) drops the captured graphs. The state (``x``, ``sc`` and their adjoints) is
        allocated from sweep 1's first outputs."""
        import torch

        g = getattr(self, "_g", None)
        shared = heads.dim() == 1 and heads.stride(0) == 0  # one head for every atom: a zero-stride view
        if g is None or g["n"] != n or (g["head"] is not None) != shared:
            # n + 1 rows: a padded slot has at most n + 1 receiver rows (one padding row past the largest tile);
            # the tiles read the first rows of ``arange`` only (at most the largest receiver cap and its padding
            # row), and one shared head as a view of one element (tiled capacity, lever 5).
            caps = (int(self.config.receivers), *map(int, self.config.budget_receivers))
            head = heads[:1].clone() if shared else None
            g = dict(n=n, pos=pos, head=head,
                     heads=head.expand(n + 1) if shared else torch.cat([heads, heads[-1:]]),
                     arange=torch.arange(min(n, max(caps)) + 1, device=self.device),
                     node_es=torch.zeros((n,), dtype=self.dtype, device=self.device),
                     strain_grad=torch.zeros((3, 3), dtype=torch.float64, device=self.device),
                     state=_State(self.device, self.dtype, self.config.state == "host",
                                  alias=self.config.alias_state, overlap=self.config.overlap,
                                  store=getattr(torch, self.config.state_dtype)
                                  if self.config.state_dtype not in (None, "auto") else None))
            self._g = g
        elif g["head"] is not None and shared:
            g["head"].copy_(heads[:1])
        else:
            g["heads"].copy_(torch.cat([heads, heads[-1:]]))
        g["stress"] = stress
        # A new force buffer every call, handed out as the call's forces (no copy; the scatter runs outside the
        # graphs, so its address may change): 12 bytes per atom less on the device than a kept buffer and its copy.
        g["forces"] = None  # a call that raised left its buffer here
        g["forces"] = torch.zeros((n, 3), dtype=self.dtype, device=self.device)
        for tensor in [g["node_es"], g["strain_grad"]]:
            tensor.zero_()
        g["state"].zero()
        return g

    def _state(self, g, x, sc):
        """Allocate ``x``, ``sc`` and their adjoints [N, width] on sweep 1's first outputs."""
        state = g["state"]
        if not state.tensors and state.host and state.alias and self.config.state_dtype == "auto":
            state.store, self.state_dtype_state = self._auto_store(state, g["n"], x.shape[1] + sc.shape[1])
        state.allocate(g["n"], dict(x=x.shape[1], sc=sc.shape[1], gx=x.shape[1], gsc=sc.shape[1]))
        return state

    def _auto_store(self, state, n, width):
        """``TileConfig.state_dtype="auto"``: (the host state's dtype, the reason). The call's dtype while its rows
        (x and sc, the pool's under the wavefront) and the alias window (about a quarter more, in the call's dtype)
        leave ``list_reuse_floor`` available on the host; else without the kept lists if that makes them fit; else
        float16 (the kept lists too go if even that does not fit)."""
        import torch

        available = _host_available()
        if available is None:
            return self.dtype, "the call's dtype (host memory unknown)"
        rows = state._pool_rows if state.pooled else n
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        window = 0.25 * rows * width * itemsize
        full, half = rows * width * itemsize + window, rows * width * 2 + window
        room = available - int(self.config.list_reuse_floor)
        if full <= room:
            return self.dtype, "the call's dtype: fits"
        # A call that reuses the kept lists did not bin the atoms: it needs them, so they stay (float16 instead).
        kept = None if getattr(self, "_reusing", False) else self._kept
        lists = 0 if kept is None else _kept_bytes(kept)  # what _drop_kept returns to the host
        if lists and full <= room + lists:
            self._drop_kept(f"the host state ({full / 2**30:.1f} GiB) fits only without them")
            return self.dtype, f"the call's dtype after dropping the kept lists ({lists / 2**30:.1f} GiB)"
        if lists and half > room:
            self._drop_kept(f"the float16 host state ({half / 2**30:.1f} GiB) fits only without them")
        return torch.float16, (f"float16: {full / 2**30:.1f} GiB of state in the call's dtype against "
                               f"{room / 2**30:.1f} GiB above the floor")

    def _drop_kept(self, why):
        """The kept lists go for this call and the later ones (``list_reuse_state`` says why)."""
        _release_kept(self._kept)  # (this call's sweeps hold the dict: emptied, they search again, bitwise)
        self._reuse_ok, self._kept = False, None
        self.list_reuse_state = f"off: {why}"

    @staticmethod
    def _note(stats, piece):
        nodes, ei, _, _, r, h = piece
        stats["max_receivers"] = max(stats["max_receivers"], r)
        stats["max_halo"] = max(stats["max_halo"], h)
        stats["max_edges"] = max(stats["max_edges"], ei.shape[1])

    def _fitted_grid(self, pos, cell, n, reach, layers):
        """The plan's blocks per axis (``TileConfig.grid_fit``; None: the rule's). Chosen once per receiver cap, atom
        count, cell and boundary skin, so the plan of an MD run keeps its grid while the atoms move."""
        if not self.config.grid_fit:
            return None
        key = (self.receivers, n, tuple(float(x) for x in cell.detach().cpu().reshape(-1)), float(reach), int(layers))
        if self._grid_fit is None or self._grid_fit[0] != key:
            self._grid_fit = (key, fit_grid(pos, cell, self.receivers, reach, layers))
        return self._grid_fit[1]

    def _boundary_reach(self, n):
        """``TileConfig.boundary_skin``: the plan's ``reach`` and ``min_layers`` (``plan_tiles``); no split without
        the wavefront, and with "auto" only where its grid would have the layers anyway."""
        if not self.config.boundary_skin or not self._wave_expected(n):
            return 0.0, 0
        reach = float(self.field.neighbours.cutoff) + self._reuse_skin()
        return reach, self._wave_layers(n)

    def _wave_layers(self, n):
        """The fewest grid layers the wavefront runs on: ``WAVEFRONT_MIN_LAYERS`` with "on" or where the plain state
        would not fit the host, else ``WAVEFRONT_LAYERS``."""
        if self.config.wavefront == "on" or self._wave_short.get(n):
            return WAVEFRONT_MIN_LAYERS
        return WAVEFRONT_LAYERS

    def _wavefront_on(self, grid, n):
        """Whether this call runs the wavefront schedule (``TileConfig.wavefront``)."""
        config = self.config
        if config.wavefront == "off" or not config.alias_state:
            return False
        if config.wavefront == "on":
            return True
        return self._wave_expected(n) and int(grid[0]) >= self._wave_layers(n)

    def _wave_expected(self, n):
        """``TileConfig.wavefront`` without the grid's layer test: whether a call at ``n`` atoms runs the wavefront
        when its grid has the layers."""
        config = self.config
        if config.wavefront == "off" or not config.alias_state:
            return False
        if config.wavefront == "on":
            return True
        if config.state == "device":  # "auto" decides on the host's memory; the device pool runs with "on"
            return False
        # "auto": only where the plain state (x and sc, and the alias window, about a quarter more) would leave the
        # host below the floor: the schedule costs about 3 % of a call (its search pass; 5M SrTiO3, both Ada GPUs).
        # Decided on the first call at an atom count, then kept.
        decided = self._wave_auto.get(n)
        if decided is None:
            width, available = self._state_width(), _host_available()
            if width is None or available is None:
                decided, short = n >= 2**20, False
            else:
                itemsize = 8 if str(self.dtype) == "torch.float64" else 4
                # TileConfig.state_dtype: 16-bit rows when set to one ("auto" stores the call's dtype while it fits)
                stored = 2 if self.config.state_dtype not in (None, "auto") else itemsize
                plain = n * width * (stored + 0.25 * itemsize)
                decided, short = plain + int(config.list_reuse_floor) > available, plain > available
            self._wave_auto[n], self._wave_short[n] = decided, short
        return decided

    def _state_width(self):
        """Floats per atom of the plain host state (x and sc): from the allocated state, else the model's
        ``linear_up`` and ``skip_tp`` output widths; None when unknown."""
        g = getattr(self, "_g", None)
        if g is not None and g["state"].tensors:
            return sum(int(t.shape[1]) for name, t in g["state"].tensors.items() if name in ("x", "sc"))
        interaction = self.model.interactions[1]
        try:
            return int(interaction.linear_up.irreps_out.dim) + int(interaction.skip_tp.irreps_out.dim)
        except AttributeError:
            return None

    def _wavefront(self, tiles, source, local_of, n, padded):
        """The wavefront's search pass: every tile's graph built once (without keeping its lists), recording each
        atom's first and last tile and the tiles each tile reads x from. ``(keys, steps, peak, first, last)``: the
        padded slots' keys (None eager), the schedule and its row pool size, and the host int32 first and last tile
        of every atom (the alias window's plan)."""
        import torch

        owner = torch.empty(n, dtype=torch.int32, device=self.device)
        for index, receivers in enumerate(tiles):
            owner[receivers] = index
        first = torch.full((n,), -1, dtype=torch.int32)
        last = torch.full((n,), -1, dtype=torch.int32)
        needs = [None] * len(tiles)

        def visit(index, piece):
            nodes = piece[0]
            needs[index] = torch.unique(owner[nodes]).cpu().tolist()
            nodes = nodes.cpu()
            first[nodes[first[nodes] < 0]] = index
            last[nodes] = index

        self._cache_paused = True
        try:
            if padded:
                keys = self._slot_keys(tiles, source, local_of, n, visit)
            else:
                keys = None
                for index, receivers in enumerate(tiles):
                    visit(index, self._piece(index, receivers, source, local_of))
        finally:
            self._cache_paused = False
        owner = None  # 4 bytes per atom on the device, before the sweeps
        # Each tile's last reader among its own atoms (its receivers): no whole-system int64 copies on the host.
        ready = [int(last[t.cpu()].max()) if t.numel() else -1 for t in tiles]
        steps, peak = wavefront_schedule(needs, ready, [int(t.shape[0]) for t in tiles],
                                         self.config.wavefront_group)
        return keys, steps, peak, first, last

    def _slot_keys(self, tiles, source, local_of, n, visit=None):
        """Each tile's slot: the edge bucket ``ceil(edges / step)``, with ``step`` set on the first call from the
        largest tile (``TileConfig.bucket`` of its edges, at least 256, a multiple of 256). A slot holds one
        receiver row more than the largest of its tiles with ``TileConfig.headroom``, at most the receiver cap (the
        extra row is the padding receiver), their halo with the same headroom and its bucket's edge count. A slot
        grows (and its graphs and compiled shapes go) only when a tile does not fit it: the fit test takes the tiles'
        own sizes, the headroom is for sizing only (in MD the tiles' sizes move a little every step). An exceeded
        size grows by at least a quarter of the largest slot's in that size: the large slot's sizes drift by a few per
        cent in MD, while a small slot (the remainder pieces of blocks just above the cap) can triple
        (tiled-slot-growth.md). A tile whose bucket has no slot runs in a neighbouring bucket's slot when it fits: in MD
        a tile's edges drift across a bucket's edge by a few per cent, which would otherwise compile a new slot. The
        edges take no headroom: a slot's padded edges cost every call (+4.8 % edges, +2.5 % step at 1M;
        tiled-1m-speed.md, "Slots in MD")."""
        sizes = []
        for index, receivers in enumerate(tiles):
            piece = self._piece(index, receivers, source, local_of)
            _, ei, _, _, r, h = piece
            sizes.append((r, h, int(ei.shape[1])))
            if visit is not None:  # the wavefront's search pass
                visit(index, piece)
            del piece
        if self._edge_step is None:
            largest = max(e for _, _, e in sizes)
            self._edge_step = max(256, 256 * math.ceil(float(self.config.bucket) * largest / 256))
        step = self._edge_step
        keys = [max(1, math.ceil(e / step)) for _, _, e in sizes]
        for index, (key, size) in enumerate(zip(keys, sizes)):
            if key in self._slots:
                continue
            raw = (size[0] + 1,) + size[1:]
            near = [k for k in (key - 1, key + 1)
                    if k in self._slots and all(a <= b for a, b in zip(raw, self._slots[k]["sizes"]))]
            if near:
                keys[index] = near[0]
        head = 1.0 + float(self.config.headroom)
        self.slot_needs = {}  # each bucket's largest tile sizes this call, before headroom (the call's report)
        for key in set(keys):
            members = [size for size, k in zip(sizes, keys) if k == key]
            self.slot_needs[key] = tuple(max(size[i] for size in members) for i in range(3))
            # Receivers with headroom too (tiles are planned again every call, so their sizes move a little), but
            # never past the cap, which no tile exceeds: full tiles keep their slot.
            rows = min(self.receivers, n,  # nor the atom count
                       math.ceil(max(r for r, _, _ in members) * head))
            need = (rows + 1, max(1, math.ceil(max(h for _, h, _ in members) * head)),
                    key * step)
            raw = (self.slot_needs[key][0] + 1,) + self.slot_needs[key][1:]  # one padding row
            slot = self._slots.get(key)
            if slot is None or any(a > b for a, b in zip(raw, slot["sizes"])):
                if slot is None:
                    grown = need
                else:  # an exceeded size grows by at least a quarter of the largest slot's (receivers at most the
                    # cap), so a slot changes a few times, not once per tile a little larger than the last
                    rows_cap = min(self.receivers, n) + 1
                    largest = [max(other["sizes"][i] for other in self._slots.values()) for i in range(3)]
                    grown = tuple(b if a <= b else max(c, b + math.ceil(0.25 * top))
                                  for a, b, c, top in zip(raw, slot["sizes"], need, largest))
                    grown = (min(grown[0], max(rows_cap, need[0])), grown[1], grown[2])
                self._slots[key] = self._new_slot(grown)
                self.slot_changes += 1
        return keys

    def _bind_tables(self, tables):
        for reassociation, table in zip(self._reassociations, tables):
            reassociation.bind(table)

    def _eager_tile(self, sweep, piece, g):
        """One sweep on one unpadded tile (``mode="eager"``)."""
        nodes, ei, shifts, local, r, _ = piece
        if sweep != 2 and self._reassociations:  # interaction 0 runs on the tile's R + H rows
            self._bind_tables(self.field.species_tables(nodes.shape[0]))
        state = g["state"]
        if g.get("local_frame") is not None:  # TILE_LOCAL_FRAME, tile-local
            positions64, cell, inverse = g["local_frame"]
            if _tile_list().rows_on_host(positions64, self.device):  # TILE_HOST_ROWS: the tile's rows from the host
                rows64 = positions64[nodes.cpu()].to(self.device)
            else:
                rows64 = positions64[nodes]
            rows, remaining = _tile_local(rows64, r, local, shifts, cell, inverse, self.dtype)
            vectors = rows[local[1]] - rows[local[0]] + remaining
        elif g.get("frame") is not None:  # TILE_LOCAL_FRAME, bins and offsets (host positions)
            vectors = _frame_vectors(g["frame"], ei, shifts, None, None)
        else:
            vectors = g["pos"][ei[1]] - g["pos"][ei[0]] + shifts
        tile = (nodes, local, r, vectors, None, None, None)
        receivers = nodes[:r]
        strain = True if g["stress"] else None
        g_eps = None
        if sweep == 1:
            es, x, sc = self._first(tile, state)
            state = self._state(g, x, sc)
            g["node_es"][receivers] += es
            state.put("x", receivers, x)
            state.put("sc", receivers, sc)
            return
        if sweep == 2:
            es, g_vec, g_x, g_sc, g_eps = self._second(tile, state, strain)
            g["node_es"][receivers] += es
            state.add("gx", nodes, g_x)
            state.put("gsc", receivers, g_sc)
        else:
            _, g_vec, g_eps = self._first(tile, state, adjoint=True, strain=strain)
        g["forces"].index_add_(0, ei[1], g_vec, alpha=-1)
        g["forces"].index_add_(0, ei[0], g_vec)
        if g_eps is not None:
            g["strain_grad"].add_(g_eps.double())

    def evaluate(self, atoms: Any, stress: bool = False):
        """Energy (eV), forces (eV/Å, float64) and the 3x3 stress (eV/Å³) or None of ``atoms`` (one-off)."""
        import torch

        if stress and not self.field.compute_stress:
            raise ValueError("stress=True needs a force field built with compute_stress=True")
        out = self(self.field._prepare_one_off(atoms))
        energy = float(out.energy.reshape(-1)[0])
        forces = out.forces.detach().to(torch.float64).cpu().numpy()
        values = out.stress.detach().to(torch.float64).cpu().numpy() if stress else None
        return energy, forces, values
