"""Kernel-lab variants of the w' backward for H200 ideas 37 and 38 (plan revision 15, 6 November 2026).

Text transforms of the plan's w' render for ``tools/omat0_kernel_lab.py``'s ``wprime_variants``. Hook:

    def _ideas3738_variants(kernel):
        import lab_ideas3738  # tools/ is on sys.path when the lab runs as a script
        return lab_ideas3738.ideas3738_variants(kernel)

and ``**_ideas3738_variants(kernel),`` at the end of the dict ``wprime_variants`` returns.

Idea 38, step 1 (dout locality bound):
- ``FF-dout-row0`` (falsifier): every ``dout<n>[int64_t(receiver) * D + ...]`` load (40 in the OMAT-0 render) reads
  receiver 0's row instead, so all dout loads of all edges hit one 20 KB set of rows (L1/L2 resident). The earlier
  ``FF-dout-one-row`` changed the unused ``output_base`` and compiled to the base's SASS (Codex review, correction
  A). This one builds at fewer registers (72 against 96 on sm_90, Codex), so its resource-matched control is the
  lab's ``maxreg72`` (the real addresses at ``--maxrregcount=72``).

Idea 37 (w' split by spherical-harmonic degree, Codex idea 1):
- ``harmonic`` and ``harmonic-rot`` (rounding): one 128-thread block per edge; warp d owns the paths whose edge
  harmonic has degree d (OMAT-0: path degrees 0, 1, 1, 0, 2, 2, 1, 3, 3, 2) and only its 2d + 1 edge_gradient
  components. Each warp writes its 512 dx partials and its dr partial to shared memory (4 x 512 + 4 floats), one
  barrier, then the block adds the four partials and issues the usual 512 dx atomics per edge; dsh components are
  written by their degree's warp (disjoint). ``harmonic-rot`` rotates the degree-to-warp assignment by a hash of the
  block (Codex's ``harmonic_cta_rotated``). The text follows Codex's ``compile_research.py`` (degree bodies) and
  ``compile_combined.py`` (wrapper) exactly; on the H200 round-9 render (SHA-256 5565cb33...) ``harmonic-rot``
  reproduces Codex's prototype byte for byte (SHA-256 0232182c...). Launch: grid = edges, 128 threads (the lab
  marker ``// lab launch threads=128 persistent=0 cohort=128``).

CPU check (no GPU, no torch): ``python -I -B lab_ideas3738.py RENDER.cu OUTDIR`` writes every variant's source.
"""
import re
import sys

_DOUT = r"(dout\d\[)int64_t\(receiver\)"


def _dout_row0(source):
    s, n = re.subn(_DOUT, r"\g<1>int64_t(0)", source)
    if not n:
        raise ValueError("variant pattern found 0 times: dout<n>[int64_t(receiver) ...]")
    return s


def _path_blocks(src):
    """(start, end, degree) of the top-level path scopes inside the channel loop (Codex's split)."""
    start = src.index("    for (int channel = lane;")
    brace = src.index("{", start)
    depth, blocks, b = 0, [], None
    for i in range(brace + 1, len(src)):
        if src[i] == "{":
            if depth == 0:
                b = i
            depth += 1
        elif src[i] == "}":
            if depth == 0:
                break
            depth -= 1
            if depth == 0:
                blocks.append((b, i + 1))
    info = []
    for a, b in blocks:
        components = sorted({int(x) for x in re.findall(r"edge_gradient\[(\d+)\]", src[a:b])})
        if not components:
            raise ValueError("harmonic: a path scope without edge_gradient components")
        degree = int(components[0] ** 0.5)
        if not all(degree * degree <= x < (degree + 1) ** 2 for x in components):
            raise ValueError("harmonic: a path scope mixes harmonic degrees")
        info.append((a, b, degree))
    return info


def _degree_source(src, info, degree):
    variant = src
    for a, b, d in reversed(info):
        if d != degree:
            variant = variant[:a] + variant[b:]
    loop = "for (int component = 0; component < EDGE_DIM; ++component)"
    if variant.count(loop) != 2:
        raise ValueError(f"harmonic: component loop found {variant.count(loop)} times")
    return variant.replace(loop, f"for (int component = {degree * degree}; component < {(degree + 1) ** 2}; "
                                 "++component)")


def harmonic(source, rotate=True):
    """Idea 37's combined four-warp w' (Codex's ``harmonic_cta[_rotated]``) from the plan's render."""
    info = _path_blocks(source)
    degrees = sorted({d for *_, d in info})
    if degrees != [0, 1, 2, 3]:
        raise ValueError(f"harmonic: needs harmonic degrees 0..3 (four warps), found {degrees}")
    header, rest = source.split('extern "C" __global__ void oeq_projected_backward_wprime(', 1)
    params = rest.split(") {", 1)[0]
    names = [re.search(r"(\w+)\s*$", p.strip()).group(1) for p in params.split(",")]
    functions = []
    for degree in range(4):
        body = _degree_source(source, info, degree).split(") {", 1)[1]
        body = body.replace("int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + logical_group",
                            "int64_t(blockIdx.x)")
        pat = r"atomicAdd\(\s*dx \+ input_base \+ (\d+) \+ channel,\s*scalar_t\((input_gradient_\d+_\d+)\)\);"
        body, n = re.subn(pat, lambda m: f"partial_dx[{degree} * INPUT_DIM + {m[1]} + channel] = scalar_t({m[2]});",
                          body)
        if n != 4:
            raise ValueError(f"harmonic: {n} dx atomics in the degree-{degree} body (expected 4)")
        if body.count("dr[e] = scalar_t(radial_gradient);") != 1:
            raise ValueError("harmonic: dr store not found")
        body = body.replace("dr[e] = scalar_t(radial_gradient);", f"partial_dr[{degree}] = scalar_t(radial_gradient);")
        functions.append(f"__device__ __forceinline__ void body{degree}({params}, scalar_t* partial_dx, "
                         f"scalar_t* partial_dr) {{" + body)
    wrapper = f'extern "C" __global__ void oeq_projected_backward_wprime({params}) {{\n'
    wrapper += """
    const int e = blockIdx.x;
    if (e >= edge_count) return;
    const int sender = senders[e], receiver = receivers[e];
    if (sender < 0 || sender >= node_count || receiver < 0 || receiver >= node_count) {
        if (threadIdx.x < EDGE_DIM) dsh[e * EDGE_DIM + threadIdx.x] = 0;
        if (threadIdx.x == 0) dr[e] = 0;
        return;
    }
    __shared__ scalar_t partial_dx[4 * INPUT_DIM];
    __shared__ scalar_t partial_dr[4];
    const int degree = threadIdx.x / 32;
"""
    args = ", ".join(names + ["partial_dx", "partial_dr"])
    for d in range(4):
        wrapper += f'    {"if" if d == 0 else "else if"} (degree == {d}) body{d}({args});\n'
    wrapper += """
    __syncthreads();
    for (int q = threadIdx.x; q < INPUT_DIM; q += blockDim.x) {
        scalar_t total = partial_dx[q];
        for (int d = 1; d < 4; ++d) total += partial_dx[d * INPUT_DIM + q];
        atomicAdd(dx + int64_t(sender) * INPUT_DIM + q, total);
    }
    if (threadIdx.x == 0) dr[e] = ((partial_dr[0] + partial_dr[1]) + partial_dr[2]) + partial_dr[3];
}
"""
    s = header + "\n".join(functions) + wrapper
    if rotate:
        s = s.replace("const int degree = threadIdx.x / 32;",
                      "const int degree = (threadIdx.x / 32 + (blockIdx.x ^ (blockIdx.x >> 7))) & 3;")
    return s


def _launch_marker(s):
    return s + "\n// lab launch threads=128 persistent=0 cohort=128\n"


# ------------------------------------------------------------------------------------- idea 38: receiver tiles
_INDEX = """    const int64_t e =
        int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + logical_group;
    if (e >= edge_count) return;
"""
_DOUT_LOAD = r"\bdout(\d)\[int64_t\(receiver\) \* (\d+)([^\[\]]*)\]"


def stage_tiles(source):
    """Idea 38: one block per receiver tile (consecutive edges of one receiver, at most T of them; the tile starts are
    the extra argument ``fm_tiles`` [tiles + 1], the last entry the edge count). The block copies the receiver's dout
    row (all OUTPUT_DIM floats, 20 KB for OMAT-0) into shared memory with float4 loads, one barrier, then each warp
    processes the tile's edges logical_group, logical_group + warps, ... with the unchanged per-edge body, reading
    dout from shared memory. Each edge's arithmetic is the base's: dsh and dr bitwise, dx atomics in another order."""
    dims = {}
    for n, d, _ in re.findall(_DOUT_LOAD, source):
        if dims.setdefault(int(n), int(d)) != int(d):
            raise ValueError("stage_tiles: one dout block with two row lengths")
    if not dims:
        raise ValueError("variant pattern found 0 times: dout<n>[int64_t(receiver) * D ...]")
    offsets, total = {}, 0
    for n in sorted(dims):
        if dims[n] % 4:
            raise ValueError("stage_tiles: a dout row length not a multiple of 4")
        offsets[n], total = total, total + dims[n]
    s = source
    if s.count("    scalar_t* dr) {") != 1:
        raise ValueError("stage_tiles: kernel signature end not found")
    s = s.replace("    scalar_t* dr) {", "    scalar_t* dr,\n    const int32_t* __restrict__ fm_tiles) {")
    copy = "".join(
        f"        for (int i = threadIdx.x; i < {dims[n] // 4}; i += blockDim.x)\n"
        f"            reinterpret_cast<float4*>(fm_d + {offsets[n]})[i] =\n"
        f"                __ldg(reinterpret_cast<const float4*>(dout{n} + int64_t(fm_r) * {dims[n]}) + i);\n"
        for n in sorted(dims))
    prologue = f"""    __shared__ __align__(16) scalar_t fm_d[{total}];
    const int64_t fm_t0 = fm_tiles[blockIdx.x];
    const int64_t fm_t1 = fm_tiles[blockIdx.x + 1];
    {{
        const int32_t fm_r = receivers[fm_t0];
        if (fm_r >= 0 && fm_r < node_count) {{
{copy}        }}
    }}
    __syncthreads();
    for (int64_t e = fm_t0 + logical_group; e < fm_t1; e += blockDim.x / LOGICAL_GROUP_SIZE) {{
"""
    if s.count(_INDEX) != 1:
        raise ValueError("stage_tiles: edge index not found")
    head, body = s.split(_INDEX)
    returns = body.count("        return;\n")
    if returns != 1:
        raise ValueError(f"stage_tiles: {returns} early returns in the body (expected 1)")
    body = body.replace("        return;\n", "        continue;\n").rstrip()
    if not body.endswith("}"):
        raise ValueError("stage_tiles: kernel end not found")
    body = body[:-1] + "    }  // lab tile edge loop\n}\n"
    body, n = re.subn(_DOUT_LOAD, lambda m: f"fm_d[{offsets[int(m.group(1))]}{m.group(3)}]", body)
    if n != len(re.findall(_DOUT_LOAD, source)):
        raise ValueError("stage_tiles: not every dout load was rewritten")
    return head + prologue + body + "\n// lab launch threads=128 persistent=0 tiles=1\n"


def tile_starts(receivers, tile):
    """int32 [tiles + 1]: starts of the runs of equal receiver, each cut into pieces of at most ``tile`` edges, then
    the edge count (torch, on the receivers' device)."""
    import torch
    r = receivers.long()
    count = r.numel()
    index = torch.arange(count, device=r.device)
    new_run = torch.ones(count, dtype=torch.bool, device=r.device)
    new_run[1:] = r[1:] != r[:-1]
    run_start = torch.cummax(torch.where(new_run, index, torch.zeros_like(index)), 0).values
    starts = torch.nonzero((index - run_start) % tile == 0).flatten()
    return torch.cat([starts, torch.tensor([count], device=r.device)]).to(torch.int32).contiguous()


def tile_args(tile):
    def transform(args):
        return list(args) + [tile_starts(args[4], tile)]  # args[4]: receivers
    return transform


def ideas3738_variants(kernel=None):
    return {
        **{f"stage-t{t}": ("exact", stage_tiles, (), tile_args(t)) for t in (8, 16, 32)},
        "FF-dout-row0": ("falsifier", _dout_row0, ()),
        "harmonic": ("rounding", lambda s: _launch_marker(harmonic(s, rotate=False)), ()),
        "harmonic-rot": ("rounding", lambda s: _launch_marker(harmonic(s, rotate=True)), ()),
    }


if __name__ == "__main__":
    from pathlib import Path
    render = Path(sys.argv[1]).read_text()
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    print("dout loads:", len(re.findall(_DOUT, render)))
    print("path degrees:", [d for *_, d in _path_blocks(render)])
    for name, (kind, transform, extra, *_) in ideas3738_variants().items():
        try:
            (out / f"{name}.cu").write_bytes(transform(render).encode())
            print(name, kind, extra, "written")
        except ValueError as error:
            print(name, "does not apply:", error)
    (out / "harmonic-rot-nomarker.cu").write_bytes(harmonic(render, rotate=True).encode())
