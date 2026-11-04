import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/extension.h>

torch::Tensor test_add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}
"""

module = load_inline(
    name="test_inline",
    cpp_sources=cpp_source,
    functions=["test_add"],
    verbose=True,
)

a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
print("Result:", module.test_add(a, b))
