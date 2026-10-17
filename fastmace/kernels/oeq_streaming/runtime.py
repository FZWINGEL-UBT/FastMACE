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
