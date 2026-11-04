import torch

for k in dir(torch.backends.cuda):
    if "workspace" in k.lower():
        print(k)

print("Current cublaslt workspace:", torch.backends.cuda.cublaslt_workspace_size())
try:
    torch.backends.cuda.cublaslt_workspace_size(33554432)
    print("Set via call:", torch.backends.cuda.cublaslt_workspace_size())
except Exception as e:
    print("Call failed:", e)

try:
    torch.backends.cuda.set_cublaslt_workspace_size(33554432)
    print("Set via set function:", torch.backends.cuda.cublaslt_workspace_size())
except Exception as e:
    print("set_cublaslt_workspace_size failed:", e)
