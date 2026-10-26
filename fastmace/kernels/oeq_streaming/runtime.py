"""NVRTC compilation and launch of the vendored streaming kernels, wrapped as PyTorch custom ops.

render_kernels() builds the schedule and the CUDA source (pure Python; needs
OpenEquivariance for the problem description and Jinja2). StreamingKernels
compiles that source with NVRTC through ``cuda.bindings`` and launches it on
PyTorch's current stream, so the launches can be captured in a CUDA graph.
Nothing here imports OpenEquivariance or ``cuda.bindings`` at module import.
"""

import ctypes
import itertools

import numpy as np
import torch

LAYOUTS = ("mul_ir", "ir_mul")

# Kernel id -> StreamingKernels. The custom ops take the id, not the object, because a
# custom op's arguments must be tensors or plain values. Entries are compiled kernels,
# added by StreamingKernels and never changed; this is a registry, not a switch.
_REGISTRY = {}
_IDS = itertools.count()


def _check(result):
    error = result[0]
    if int(error) != 0:
        raise RuntimeError(f"CUDA/NVRTC call failed: {error}")
    return result[1:] if len(result) > 2 else (result[1] if len(result) == 2 else None)


def render_kernels(problem, dtype=np.float32, num_threads=128, layout="mul_ir"):
    """(schedule, kernel) for one uvu problem (OEq TPProblem): kernel.jit_kernel is the CUDA source."""
    from fastmace.kernels.oeq_streaming.schedule import FactorizedLaunchConfig, factorized_schedule_from_problem

    if layout not in LAYOUTS:
        raise ValueError("layout must be mul_ir or ir_mul")
    schedule = factorized_schedule_from_problem(problem)
    fields = {f: getattr(schedule, f) for f in schedule.__dataclass_fields__}
    fields.update(launch_config=FactorizedLaunchConfig(num_threads=num_threads), layout=layout)
    schedule = type(schedule)(**fields)
    kernel = schedule.kernel(np.dtype(dtype).type, backend="cuda",
                             forward_jvp_active=(False, False, False),
                             backward_jvp_active=(False, False, False, False))
    return schedule, kernel


def compile_cubin(source, capability, extra_options=()):
    """Compile CUDA source to a cubin for compute capability (major, minor) with NVRTC."""
    from cuda.bindings import nvrtc

    major, minor = capability
    program = _check(nvrtc.nvrtcCreateProgram(source.encode(), b"oeq_streaming.cu", 0, [], []))
    options = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17", b"-default-device"]
    options += [option.encode() for option in extra_options]  # e.g. register caps (timing screens)
    result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if int(result[0]) != 0:
        size = _check(nvrtc.nvrtcGetProgramLogSize(program))
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError("NVRTC compilation failed:\n" + log.decode(errors="replace")[-4000:])
    size = _check(nvrtc.nvrtcGetCUBINSize(program))
    cubin = b" " * size
    _check(nvrtc.nvrtcGetCUBIN(program, cubin))
    return cubin


class StreamingKernels:
    """Compiled forward/backward kernels for one uvu problem (OEq TPProblem) and dtype."""

    def __init__(self, problem, dtype=np.float32, num_threads=128, layout="mul_ir", receiver_backward=False,
                 extra_options=()):
        from cuda.bindings import driver

        schedule, kernel = render_kernels(problem, dtype, num_threads, layout)
        self.layout = layout
        self.schedule = schedule
        self.source = kernel.jit_kernel
        attributes = kernel.ffi_attributes
        self.input_dim = attributes["input_dim"]
        self.edge_dim = attributes["edge_dim"]
        self.weight_dim = attributes["weight_dim"]
        self.output_dim = attributes["output_dim"]
        self.channels = attributes["channels"]
        self.num_threads = attributes["num_threads"]
        self.cohort = attributes["logical_cohort_width"]
        self.shared_memory = attributes["shared_memory_bytes"]
        # Receiver-owned backward: one thread per (receiver, channel), warps aligned to nodes.
        self.receiver_backward = bool(receiver_backward)
        if self.receiver_backward and self.num_threads % 32:
            raise ValueError("receiver_backward needs a block size that is a multiple of 32")
        cubin = compile_cubin(self.source, torch.cuda.get_device_capability(), extra_options)
        torch.cuda.current_stream()  # make the primary context current
        self._module = _check(driver.cuModuleLoadData(cubin))
        self._forward = _check(driver.cuModuleGetFunction(self._module, b"oeq_projected_forward"))
        self._backward = _check(driver.cuModuleGetFunction(self._module, b"oeq_projected_backward"))
        self._backward_receiver = _check(driver.cuModuleGetFunction(self._module, b"oeq_projected_backward_receiver"))
        self.id = next(_IDS)
        _REGISTRY[self.id] = self

    def _launch(self, function, blocks, values, types):
        from cuda.bindings import driver
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(driver.cuLaunchKernel(function, blocks, 1, 1, self.num_threads, 1, 1, self.shared_memory,
                                     stream, (tuple(values), tuple(types)), 0))

    def forward(self, x, sh, weights, senders, row_ptr, out):
        nodes, edges = x.shape[0], sh.shape[0]
        blocks = (nodes * self.channels + self.num_threads - 1) // self.num_threads
        if blocks:
            p = ctypes.c_void_p
            self._launch(self._forward, blocks,
                         (nodes, edges, x.data_ptr(), sh.data_ptr(), weights.data_ptr(), senders.data_ptr(),
                          row_ptr.data_ptr(), out.data_ptr()),
                         (ctypes.c_longlong, ctypes.c_longlong, p, p, p, p, p, p))

    def backward_receiver(self, x, sh, weights, senders, row_ptr, dout, dx, dsh, dweights):
        """Receiver-owned reverse pass; dx and dsh must be zero."""
        nodes, edges = x.shape[0], sh.shape[0]
        blocks = (nodes * self.channels + self.num_threads - 1) // self.num_threads
        if blocks:
            p = ctypes.c_void_p
            self._launch(self._backward_receiver, blocks,
                         (nodes, edges, x.data_ptr(), sh.data_ptr(), weights.data_ptr(), senders.data_ptr(),
                          row_ptr.data_ptr(), dout.data_ptr(), dx.data_ptr(), dsh.data_ptr(), dweights.data_ptr()),
                         (ctypes.c_longlong, ctypes.c_longlong, p, p, p, p, p, p, p, p, p))

    def backward(self, x, sh, weights, senders, receivers, dout, dx, dsh, dweights):
        nodes, edges = x.shape[0], sh.shape[0]
        blocks = (edges * self.cohort + self.num_threads - 1) // self.num_threads
        if blocks:
            p = ctypes.c_void_p
            self._launch(self._backward, blocks,
                         (nodes, edges, x.data_ptr(), sh.data_ptr(), weights.data_ptr(), senders.data_ptr(),
                          receivers.data_ptr(), dout.data_ptr(), dx.data_ptr(), dsh.data_ptr(),
                          dweights.data_ptr()),
                         (ctypes.c_longlong, ctypes.c_longlong, p, p, p, p, p, p, p, p, p))

    def backward_edges(self, x, sh, weights, senders, receivers, dout, dx, dsh, dweights, start, stop):
        """``backward`` for the edges ``start:stop`` only: the edge tensors are addressed at ``start``, and
        ``dweights`` is a buffer whose row 0 is edge ``start`` (``chunked_conv_backward``, opt-6 OPT6-T02)."""
        count = stop - start
        blocks = (count * self.cohort + self.num_threads - 1) // self.num_threads
        if blocks:
            p, item = ctypes.c_void_p, x.element_size()
            self._launch(self._backward, blocks,
                         (x.shape[0], count, x.data_ptr(), sh.data_ptr() + start * self.edge_dim * item,
                          weights.data_ptr() + start * self.weight_dim * item, senders.data_ptr() + start * 4,
                          receivers.data_ptr() + start * 4, dout.data_ptr(), dx.data_ptr(),
                          dsh.data_ptr() + start * self.edge_dim * item, dweights.data_ptr()),
                         (ctypes.c_longlong, ctypes.c_longlong, p, p, p, p, p, p, p, p, p))


@torch.library.custom_op("fastmace::streaming_conv_forward", mutates_args=())
def streaming_conv_forward(x: torch.Tensor, sh: torch.Tensor, weights: torch.Tensor, senders: torch.Tensor,
                           row_ptr: torch.Tensor, kernel_id: int) -> torch.Tensor:
    kernels = _REGISTRY[kernel_id]
    x, sh, weights = x.contiguous(), sh.contiguous(), weights.contiguous()
    out = x.new_empty((x.shape[0], kernels.output_dim))
    kernels.forward(x, sh, weights, senders.contiguous(), row_ptr.contiguous(), out)
    return out


@streaming_conv_forward.register_fake
def _(x, sh, weights, senders, row_ptr, kernel_id):
    return x.new_empty((x.shape[0], _REGISTRY[kernel_id].output_dim))


@torch.library.custom_op("fastmace::streaming_conv_backward", mutates_args=())
def streaming_conv_backward(x: torch.Tensor, sh: torch.Tensor, weights: torch.Tensor, senders: torch.Tensor,
                            receivers: torch.Tensor, row_ptr: torch.Tensor, dout: torch.Tensor,
                            kernel_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kernels = _REGISTRY[kernel_id]
    x, sh, weights, dout = x.contiguous(), sh.contiguous(), weights.contiguous(), dout.contiguous()
    dx = torch.zeros_like(x)  # sender gradients are added atomically
    if kernels.receiver_backward:
        dsh = torch.zeros_like(sh)  # reduced per warp, added atomically
        # Edges outside every row (padding) keep a zero weight gradient.
        dweights = torch.zeros_like(weights)
        kernels.backward_receiver(x, sh, weights, senders.contiguous(), row_ptr.contiguous(), dout, dx, dsh,
                                  dweights)
    else:
        dsh = torch.empty_like(sh)
        dweights = torch.empty_like(weights)
        kernels.backward(x, sh, weights, senders.contiguous(), receivers.contiguous(), dout, dx, dsh, dweights)
    return dx, dsh, dweights


@streaming_conv_backward.register_fake
def _(x, sh, weights, senders, receivers, row_ptr, dout, kernel_id):
    return torch.empty_like(x), torch.empty_like(sh), torch.empty_like(weights)


def l2_chunk_edges(device, weight_dim, itemsize):
    """Edges per chunk of ``chunked_conv_backward``: a chunk's weight gradient fills a quarter of the L2, rounded
    down to a multiple of 128 edges (at least 128). The quarter leaves room for the weights and sender features
    that stream through the L2 beside it; on the RTX 5060 Ti (32 MiB, Q = 1280) that is 1,536 edges, the best of
    the OPT6-T01 screen (docs/validation/opt6-T01.md)."""
    l2 = int(torch.cuda.get_device_properties(device).L2_cache_size)
    rows = l2 // (4 * int(weight_dim) * int(itemsize))
    return max(128, rows // 128 * 128)


_SIDE_STREAMS = {}  # device index -> the side stream of the chunked backward's input gradients


def _side_stream(device):
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    if index not in _SIDE_STREAMS:
        _SIDE_STREAMS[index] = torch.cuda.Stream(device=index)
    return _SIDE_STREAMS[index]


@torch.library.custom_op("fastmace::streaming_conv_folded_backward", mutates_args=())
def streaming_conv_folded_backward(x: torch.Tensor, sh: torch.Tensor, weights: torch.Tensor, senders: torch.Tensor,
                                   receivers: torch.Tensor, dout: torch.Tensor, weight: torch.Tensor,
                                   main: torch.Tensor, last: torch.Tensor, kernel_id: int, chunk_edges: int,
                                   tile_rule: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The edge-owned backward of ``weights = a @ weight.T`` and the streaming convolution, returning (dx, dsh, da).

    The edges run in chunks of ``chunk_edges``. Each chunk's weight gradient goes into one of two chunk-sized
    slots, and the folded layer's input gradient reads it at once, from the L2, on a side stream while the
    next chunk's backward runs (one stream left the small input-gradient grids exposed: NbCl5 2.83 against
    2.10 ms at 1,536 edges, docs/validation/opt6-T01.md). Every edge's values are those of
    ``streaming_conv_backward`` followed by ``folded_linear_input_grad`` (the same kernels, the same reduction
    order); only the order of the atomic sender-gradient additions differs, as between any two runs.
    """
    from fastmace.kernels.triton.skinny_linear import input_grad_into

    kernels = _REGISTRY[kernel_id]
    x, sh, weights, dout = x.contiguous(), sh.contiguous(), weights.contiguous(), dout.contiguous()
    senders, receivers = senders.contiguous(), receivers.contiguous()
    edges = sh.shape[0]
    dx = torch.zeros_like(x)  # sender gradients are added atomically (all on the current stream)
    dsh = torch.empty_like(sh)
    da = weights.new_empty((edges, weight.shape[1]))
    rows = max(1, min(int(chunk_edges), edges))
    slots = weights.new_empty((2, rows, weights.shape[1]))
    current, side = torch.cuda.current_stream(x.device), _side_stream(x.device)
    side.wait_stream(current)
    freed = [None, None]  # per slot: the side stream's event after its input gradient read the slot
    for chunk, start in enumerate(range(0, edges, rows)):
        stop, slot = min(edges, start + rows), chunk % 2
        if freed[slot] is not None:
            current.wait_event(freed[slot])
        kernels.backward_edges(x, sh, weights, senders, receivers, dout, dx, dsh, slots[slot], start, stop)
        written = torch.cuda.Event()
        written.record(current)
        with torch.cuda.stream(side):
            side.wait_event(written)
            input_grad_into(slots[slot, :stop - start], weight, main, last, da[start:stop], tile_rule)
            freed[slot] = torch.cuda.Event()
            freed[slot].record(side)
    current.wait_stream(side)
    return dx, dsh, da


@streaming_conv_folded_backward.register_fake
def _(x, sh, weights, senders, receivers, dout, weight, main, last, kernel_id, chunk_edges, tile_rule):
    return torch.empty_like(x), torch.empty_like(sh), sh.new_empty((sh.shape[0], weight.shape[1]))


class _FoldedStreamingConvolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, weight, main, last, x, sh, senders, receivers, row_ptr, kernel_id, chunk_edges, tile_rule):
        weights = a @ weight.t()  # folded_linear's forward
        ctx.save_for_backward(x, sh, weights, senders, receivers, weight, main, last)
        ctx.kernel_id, ctx.chunk_edges, ctx.tile_rule = kernel_id, chunk_edges, tile_rule
        return streaming_conv_forward(x, sh, weights, senders, row_ptr, kernel_id)

    @staticmethod
    def backward(ctx, dout):
        x, sh, weights, senders, receivers, weight, main, last = ctx.saved_tensors
        dx, dsh, da = streaming_conv_folded_backward(x, sh, weights, senders, receivers, dout, weight, main, last,
                                                     ctx.kernel_id, ctx.chunk_edges, ctx.tile_rule)
        return da, None, None, None, dx, dsh, None, None, None, None, None, None


def folded_streaming_convolution(a, weight, x, sh, senders, receivers, row_ptr, kernels, chunk_edges,
                                 tile_rule="fixed", parts=None):
    """``streaming_convolution(x, sh, a @ weight.T, ...)`` with the chunked backward (``chunked_conv_backward``).

    ``a`` is the folded radial layer's input [E, P + 1], ``weight`` its [W, b] [Q, P + 1] (no gradient: MD
    forces only); ``parts`` are weight's contiguous halves (``folded_constants``), else they are sliced here.
    The gradient reaches a, x and sh; the [E, Q] weight gradient is never formed in full.
    """
    if weight.requires_grad:
        raise ValueError("folded_streaming_convolution: the folded weight must not require a gradient")
    if parts is None:
        p = weight.shape[1] - 1
        parts = (weight[:, :p].contiguous(), weight[:, p].contiguous())
    return _FoldedStreamingConvolution.apply(a, weight, parts[0], parts[1], x, sh, senders, receivers, row_ptr,
                                             kernels.id, int(chunk_edges), tile_rule)


class _StreamingConvolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sh, weights, senders, receivers, row_ptr, kernel_id):
        ctx.save_for_backward(x, sh, weights, senders, receivers, row_ptr)
        ctx.kernel_id = kernel_id
        return streaming_conv_forward(x, sh, weights, senders, row_ptr, kernel_id)

    @staticmethod
    def backward(ctx, dout):
        x, sh, weights, senders, receivers, row_ptr = ctx.saved_tensors
        dx, dsh, dweights = streaming_conv_backward(x, sh, weights, senders, receivers, row_ptr, dout,
                                                    ctx.kernel_id)
        return dx, dsh, dweights, None, None, None, None


def streaming_convolution(x, sh, weights, senders, receivers, row_ptr, kernels):
    """OEq-equivalent message [nodes, output_dim] (mul_ir); senders/receivers/row_ptr int32, row_ptr [nodes + 1]."""
    return _StreamingConvolution.apply(x, sh, weights, senders, receivers, row_ptr, kernels.id)
