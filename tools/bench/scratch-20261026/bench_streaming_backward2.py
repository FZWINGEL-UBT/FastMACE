import torch
import fastmace as fm
import numpy as np

# Load model using FastMACE's ForceField
ff = fm.ForceField("/home/fzwin/models/mace-mh-1.model", backend="hybrid", precision="float32")
model = ff.model
interaction = model.interactions[1] # Interaction 2
k_edge = interaction.streaming_kernels
print("Existing k_edge:", k_edge)
print("Problem channels:", k_edge.channels, "threads:", k_edge.num_threads)

# Create receiver-owned counterpart for the exact same problem
k_recv = fm.kernels.oeq_streaming.StreamingKernels(
    k_edge.problem, dtype=np.float32, layout=k_edge.layout, receiver_backward=True
)
print("Built k_recv successfully!")

nodes = 1944
edges = 56320
x = torch.randn(nodes, k_edge.problem.irreps_in1.dim, device="cuda", dtype=torch.float32, requires_grad=True)
sh = torch.randn(edges, k_edge.problem.irreps_in2.dim, device="cuda", dtype=torch.float32, requires_grad=True)
weights = torch.randn(edges, k_edge.problem.weight_numel, device="cuda", dtype=torch.float32, requires_grad=True)

edges_per_node = edges // nodes
row_ptr_list = [0]
senders_list = []
receivers_list = []
for i in range(nodes):
    count = edges_per_node + (1 if i < (edges % nodes) else 0)
    row_ptr_list.append(row_ptr_list[-1] + count)
    receivers_list.extend([i] * count)
    senders_list.extend(np.random.randint(0, nodes, size=count).tolist())

row_ptr = torch.tensor(row_ptr_list, device="cuda", dtype=torch.int32)
senders = torch.tensor(senders_list, device="cuda", dtype=torch.int32)
receivers = torch.tensor(receivers_list, device="cuda", dtype=torch.int32)
dout = torch.randn(nodes, k_edge.problem.irreps_out.dim, device="cuda", dtype=torch.float32)

from fastmace.kernels.oeq_streaming import streaming_convolution

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
