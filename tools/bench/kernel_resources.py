"""Local memory of the NVRTC kernels one MD plan builds (plans/capacity.md, the conductor's check 2 after CAP-T03).

Under WSL2 the driver reserves a kernel's local memory (its stack frame) for every thread the GPU can hold, keeps
the reservation for the process's lifetime and does not give it back when the stack limit is lowered
(docs/validation/capacity-T03.md). This tool builds the default plan's ``fm.Simulation`` for one supercell, runs two
steps, then lists

- FastMACE's NVRTC kernels (``fastmace.kernels.oeq_streaming``: the streaming convolution, the fused folded forward,
  the wprime backward): ``cuFuncGetAttribute`` of the loaded functions (local bytes per thread, registers, static
  shared memory) and ptxas's report for the same source (``--ptxas-options=-v``: stack frame, spill stores and loads);
- OpenEquivariance's JIT kernels in the model (objects with a ``jit_kernel`` source): ptxas's report with OEq's own
  options (``--use_fast_math``); OEq keeps its function handles in C++.

The WSL2 reservation of a kernel is its local bytes per thread times the threads per SM times the SMs
(``reservation_bytes``); the tool ranks by it. It also reads the stack limit (``cudaLimitStackSize``) after the
build and the steps, sets it down to the largest local size of the loaded FastMACE kernels, and records the device
memory outside PyTorch's allocator (``total - free - reserved``) before and after (``stack``). With
``--smem-spilling`` it compiles each source again with the PTX
pragma ``enable_smem_spilling`` at the start of every kernel (CUDA 13) and reports the same numbers. Writes
``<output>/kernels-<name>-<precision>-<atoms>.json``. Run inside a local GPU reservation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import activation_profile  # noqa: E402
import capacity_scan  # noqa: E402

SMEM_PRAGMA = 'asm volatile (".pragma \\"enable_smem_spilling\\";");'
PTXAS_LINE = re.compile(r"Function properties for (\S+)\s+(\d+) bytes stack frame, (\d+) bytes spill stores, "
                        r"(\d+) bytes spill loads")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--repeat", default="3x3x3", help="NXxNYxNZ")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE", help="RewriteConfig overrides")
    p.add_argument("--smem-spilling", action="store_true", help="also compile every source with the pragma")
    return p.parse_args(argv)


def with_smem_spilling(source):
    """``source`` with the pragma as the first statement of every ``__global__`` function body."""
    out, at = [], 0
    for match in re.finditer(r"__global__[^{;]*\{", source):
        out.append(source[at:match.end()])
        out.append("\n" + SMEM_PRAGMA + "\n")
        at = match.end()
    out.append(source[at:])
    return "".join(out)


def ptxas_report(source, capability, options):
    """{lowered kernel name: dict(stack, spill_stores, spill_loads, registers, smem)} from NVRTC's ptxas report."""
    from cuda.bindings import nvrtc

    major, minor = capability
    program = nvrtc.nvrtcCreateProgram(source.encode(), b"probe.cu", 0, [], [])[1]
    flags = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17", b"--ptxas-options=-v",
             *[option.encode() for option in options]]
    result = nvrtc.nvrtcCompileProgram(program, len(flags), flags)
    size = nvrtc.nvrtcGetProgramLogSize(program)[1]
    log = b" " * size
    nvrtc.nvrtcGetProgramLog(program, log)
    text = log.decode(errors="replace")
    if int(result[0]) != 0:
        raise RuntimeError("NVRTC compilation failed:\n" + text[-3000:])
    kernels = {}
    for name, stack, stores, loads in PTXAS_LINE.findall(text):
        kernels.setdefault(name, {}).update(stack=int(stack), spill_stores=int(stores), spill_loads=int(loads))
    for block in text.split("ptxas info    : Compiling entry function")[1:]:
        name = block.split("'")[1]
        used = re.search(r"Used (\d+) registers", block)
        smem = re.search(r"(\d+) bytes smem", block)
        kernels.setdefault(name, {}).update(registers=int(used.group(1)) if used else None,
                                            smem=int(smem.group(1)) if smem else 0)
    return kernels, text


def function_attributes(function):
    from cuda.bindings import driver

    names = {"local": driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
             "registers": driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
             "static_smem": driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
             "max_threads": driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK}
    return {key: int(driver.cuFuncGetAttribute(attribute, function)[1]) for key, attribute in names.items()}


def outside_allocator():
    import torch

    free, total = torch.cuda.mem_get_info()
    return int(total - free - torch.cuda.memory_reserved())


def stack_limit():
    from cuda.bindings import runtime

    error, value = runtime.cudaDeviceGetLimit(runtime.cudaLimit.cudaLimitStackSize)
    return int(value) if int(error) == 0 else None


def set_stack_limit(value):
    from cuda.bindings import runtime

    return int(runtime.cudaDeviceSetLimit(runtime.cudaLimit.cudaLimitStackSize, int(value))[0])


def fastmace_kernels():
    """(owner label, object with ``.source``, {attribute name: CUfunction}) of every compiled FastMACE kernel."""
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming import fused_forward, runtime, wprime_backward

    for label, registry in (("streaming", runtime._REGISTRY), ("fused_forward", fused_forward._REGISTRY),
                            ("wprime_backward", wprime_backward._REGISTRY)):
        for key, obj in registry.items():
            functions = {name: value for name, value in vars(obj).items() if isinstance(value, driver.CUfunction)}
            yield f"{label}[{key}]", obj, functions


def oeq_kernels(model):
    """(path, source) of every OEq JIT object reachable from ``model``'s modules (attributes with ``jit_kernel``)."""
    seen = set()
    for path, module in model.named_modules():
        for name, value in list(vars(module).items()) + [("", module)]:
            source = getattr(value, "jit_kernel", None)
            if isinstance(source, str) and id(value) not in seen:
                seen.add(id(value))
                yield f"{path}.{name}".strip("."), type(value).__name__, source


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm
    from fastmace.config import RewriteConfig

    name, cell = capacity_scan.base_cell(args.structure)
    atoms = cell.repeat(tuple(int(n) for n in args.repeat.lower().split("x")))
    atoms.set_pbc(True)
    spec = importlib.util.spec_from_file_location("_ab_frontends", activation_profile.TOOLS / "ab_frontends.py")
    frontends = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontends)
    frontends.thermalize(atoms, 300.0, 1)
    rewrites = dict(item.split("=", 1) for item in args.rewrites)
    field = fm.ForceField(fm.load(args.model), atoms, precision=args.precision,
                          **({"rewrites": RewriteConfig(**rewrites)} if rewrites else {}))
    sim = fm.Simulation(atoms, field, seed=1, integrator=fm.integrators.Langevin(
        timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.002))
    sim.run(steps=2)
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    capability = (properties.major, properties.minor)
    threads = properties.max_threads_per_multi_processor * properties.multi_processor_count
    rows = []

    def add(owner, kind, source, options, attributes=None):
        report, _ = ptxas_report(source, capability, options)
        spilled = ptxas_report(with_smem_spilling(source), capability, options)[0] if args.smem_spilling else {}
        for kernel, numbers in report.items():
            row = dict(owner=owner, kind=kind, kernel=kernel, **numbers)
            if attributes:
                match = [value for key, value in attributes.items() if key in kernel] or list(attributes.values())
                row["loaded_local"] = match[0]["local"] if len(attributes) == 1 else None
            row["reservation_bytes"] = numbers.get("stack", 0) * threads
            if kernel in spilled:
                row["smem_spilling"] = spilled[kernel]
                row["smem_spilling"]["reservation_bytes"] = spilled[kernel].get("stack", 0) * threads
            rows.append(row)

    for owner, obj, functions in fastmace_kernels():
        attributes = {key: function_attributes(function) for key, function in functions.items()}
        add(owner, "fastmace-nvrtc", obj.source, (), attributes)
        for key, values in attributes.items():
            rows.append(dict(owner=owner, kind="loaded", kernel=key, **values,
                             reservation_bytes=values["local"] * threads))
    model = sim._engine.force_field.model
    for path, kind, source in oeq_kernels(model):
        add(path, f"oeq:{kind}", source, ("--split-compile=0", "--use_fast_math"))
    rows.sort(key=lambda row: -row["reservation_bytes"])
    # The conductor's stack-limit check: down to what the loaded FastMACE kernels need, after setup.
    torch.cuda.synchronize()
    loaded = max((row["local"] for row in rows if row["kind"] == "loaded"), default=0)
    stack = dict(limit_after_setup=stack_limit(), outside_before=outside_allocator(), set_to=max(loaded, 1024))
    stack["set_error"] = set_stack_limit(stack["set_to"])
    torch.cuda.synchronize()
    sim.run(steps=2)
    torch.cuda.synchronize()
    stack.update(limit_after=stack_limit(), outside_after=outside_allocator())
    record = dict(tool="kernel_resources", structure=name, natoms=len(atoms), precision=args.precision,
                  gpu=properties.name, capability=capability, threads=threads, rewrites=rewrites,
                  smem_spilling=args.smem_spilling, stack=stack, kernels=rows)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"kernels-{name}-{args.precision}-{len(atoms)}.json"
    path.write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(f"{properties.name}: {threads} resident threads; {len(rows)} kernels ({path})")
    print(f"stack limit after setup {stack['limit_after_setup']} B; set to {stack['set_to']} B (error "
          f"{stack['set_error']}), now {stack['limit_after']} B; outside the allocator "
          f"{stack['outside_before'] / 2**20:.0f} -> {stack['outside_after'] / 2**20:.0f} MiB")
    for row in rows[:40]:
        extra = row.get("smem_spilling")
        after = (f" | pragma: stack {extra.get('stack')} B, spills {extra.get('spill_stores')}/"
                 f"{extra.get('spill_loads')} B, {extra.get('registers')} regs, smem {extra.get('smem')} B") if extra else ""
        print(f"{row['reservation_bytes'] / 2**30:6.3f} GiB  {row['kind']:18s} {row['owner'][:28]:28s} "
              f"{row['kernel'][:44]:44s} stack {row.get('stack', row.get('local'))} B, spills "
              f"{row.get('spill_stores')}/{row.get('spill_loads')} B, {row.get('registers')} regs{after}")
    sim.close()
    sys.stdout.flush()  # os._exit skips the buffers of a redirected stdout
    os._exit(0)


if __name__ == "__main__":
    main()
