import torch
import numpy as np
from fastmace.kernels.oeq_streaming.runtime import StreamingKernels, streaming_convolution
from openequivariance.core.e3nn_lite import TPProblem

# Load model to extract exact Interaction 2 TPProblem
model_path = "/home/fzwin/models/mace-mh-1.model"
model = torch.load(model_path, map_location="cuda", weights_only=False)
interaction = model.interactions[1] # Interaction 2
conv = interaction.conv_tp
problem = conv.input_args["problem"]
ir_mul = conv.layout_transpose_in is not None
layout = "ir_mul" if ir_mul else "mul_ir"

print(f"Problem: irreps_in1={problem.irreps_in1}, irreps_in2={problem.irreps_in2}, irreps_out={problem.irreps_out}")
print(f"layout={layout}")

# Build kernels for edge-owned (default) and receiver-owned
k_edge = StreamingKernels(problem, np.float32, layout=layout, receiver_backward=False)
k_recv = StreamingKernels(problem, np.float32, layout=layout, receiver_backward=True)

# Generate mock data matching NbCl5 3x3x3 dimensions
nodes = 1944
edges = 56320
sh_dim = sum(mul * ir.dim for mul, ir in problem.irreps_in2)
w_dim = problem.weight_numel

x = torch.randn(nodes, problem.irreps_in1.dim, device="cuda", dtype=torch.float32, requires_grad=True)
sh = torch.randn(edges, sh_dim, device="cuda", dtype=torch.float32, requires_grad=True)
weights = torch.randn(edges, w_dim, device="cuda", dtype=torch.float32, requires_grad=True)

# Receiver-sorted edges with CSR row_ptr
edges_per_node = edges // nodes
row_ptr_list = [0]
senders_list = []
receivers_list = []
for i in range(nodes):
    count = edges_per_node + (1 if i < (edges % nodes) else 0)
    row_ptr_list.append(row_ptr_list[-1] + count)
    receivers_list.extend([i] * count)
    # random senders
    senders_list.extend(np.random.randint(0, nodes, size=count).tolist())

row_ptr = torch.tensor(row_ptr_list, device="cuda", dtype=torch.int32)
senders = torch.tensor(senders_list, device="cuda", dtype=torch.int32)
receivers = torch.tensor(receivers_list, device="cuda", dtype=torch.int32)

dout = torch.randn(nodes, problem.irreps_out.dim, device="cuda", dtype=torch.float32)

# Warmup
for _ in range(5):
    out1 = streaming_convolution(x, sh, weights, senders, receivers, row_ptr, k_edge)
    out1.backward(dout, retain_graph=True)
    out2 = streaming_convolution(x, sh, weights, senders, receivers, row_ptr, k_recv)
    out2.backward(dout, retain_graph=True)
torch.cuda.synchronize()

reps = 30
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Benchmark edge-owned backward
out1 = streaming_convolution(x, sh, weights, senders, receivers, row_ptr, k_edge)
start.record()
for _ in range(reps):
    x.grad = None
    sh.grad = None
    weights.grad = None
    out1.backward(dout, retain_graph=True)
end.record()
torch.cuda.synchronize()
t_edge = start.elapsed_time(end) * 1000.0 / reps

# Benchmark receiver-owned backward
out2 = streaming_convolution(x, sh, weights, senders, receivers, row_ptr, k_recv)
start.record()
for _ in range(reps):
    x.grad = None
    sh.grad = None
    weights.grad = None
    out2.backward(dout, retain_graph=True)
end.record()
torch.cuda.synchronize()
t_recv = start.elapsed_time(end) * 1000.0 / reps

print(f"\n=== Streaming Convolution Backward on NbCl5 3x3x3 ===")
print(f"Edge-owned backward (current default): {t_edge:8.2f} µs")
print(f"Receiver-owned backward (test option): {t_recv:8.2f} µs")
print(f"Speedup: {(t_edge / t_recv - 1.0) * 100.0:+6.1f} %")
print(f"Time saved per force call: {t_edge - t_recv:+8.2f} µs")
