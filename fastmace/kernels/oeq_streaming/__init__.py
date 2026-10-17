"""Receiver-streaming UVU convolution (OpenEquivariance PR 223 kernels) for PyTorch.

streaming_convolution(x, sh, weights, senders, receivers, row_ptr, kernels)
computes OEq's TensorProductConv message for one uvu problem in the mul_ir
layout: one thread per (receiver, channel) scans the receiver's edges (CSR
row_ptr over receiver-sorted edges; edges outside every row contribute
nothing) and writes its outputs once; the backward runs one 32-thread group
per edge (weight gradients written directly, sender-feature gradients added
atomically). Kernels are rendered from the vendored template, compiled with
NVRTC and launched on PyTorch's current stream (CUDA-graph capturable).
See NOTICE.md and LICENSE (BSD 3-Clause) for the origin of the vendored files;
both are kept byte-identical to the research engine's copies.

Importing this package imports neither OpenEquivariance nor ``cuda.bindings``;
``fastmace.kernels.streaming_available()`` says whether the kernels can run.
"""

from fastmace.kernels.oeq_streaming.runtime import StreamingKernels, render_kernels, streaming_convolution

__all__ = ["StreamingKernels", "render_kernels", "streaming_convolution"]
