"""compute-sanitizer 2026.2 runs only (PYTHONPATH=jobs/cs_site with CS_MEMINFO="free,total" bytes): the sanitizer
aborts the host process ("double free or corruption") inside cudaMemGetInfo, which the planner reads
(torch.cuda.mem_get_info). Answer it with the values measured without the sanitizer before the run."""
import os

if os.environ.get("CS_MEMINFO"):
    import torch.cuda
    import torch.cuda.memory

    _free, _total = (int(v) for v in os.environ["CS_MEMINFO"].split(","))

    def mem_get_info(device=None):
        return _free, _total

    torch.cuda.mem_get_info = mem_get_info
    torch.cuda.memory.mem_get_info = mem_get_info
