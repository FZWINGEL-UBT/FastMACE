"""segment_gemm's two tile orders side by side, on MH-1's ``linear_2`` shapes (the library's own kernel).

Like results/c1-falsifier-20261030/gemm_order_probe.py, which found the effect with a copy of the kernel, but through
``segment_gemm._gemm`` with ``GemmChoice.column_fastest`` off and on: l = 0..3, rows = atoms * (2l + 1),
K = N = 512, float32 IEEE, alternating blocks in one process. Prints the L2 rule's choice for each product. Run
inside a GPU reservation:  python gemm_order_probe.py ATOMS
"""

import dataclasses
import sys

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
import torch

from fastmace.kernels.triton import segment_gemm as sg
from fastmace.kernels.triton import tile_rule


def main():
    atoms = int(sys.argv[1]) if len(sys.argv) > 1 else 9216
    reps, blocks = 20, 6
    device = torch.device("cuda")
    sms, l2 = tile_rule.device_sms(device), tile_rule.device_l2_bytes(device)
    print(f"{torch.cuda.get_device_name()}: {sms} SMs, L2 {l2 / 2**20:.1f} MiB")
    g = torch.Generator(device="cuda").manual_seed(1)
    total = {False: 0.0, True: 0.0}
    for ell in range(4):
        dim = 2 * ell + 1
        rows = atoms * dim
        a = torch.randn((atoms * dim, 512), device="cuda", generator=g)
        b = torch.randn((512, 512), device="cuda", generator=g) / 23
        choice = tile_rule.segment_gemm_choice(rows, 512, 512, sms)
        if choice.backend != "triton":
            print(f"l={ell} rows={rows}: {choice.backend}")
            continue
        rule = tile_rule.columns_together(rows, 512, 4, l2, tile_rule.device_name(device))
        outs = {order: torch.empty((atoms, dim * 512), device="cuda") for order in total}

        def run(order):
            sg._gemm(a, 0, dim * 512, 512, b, outs[order], 0, dim * 512, 512, rows, dim,
                     dataclasses.replace(choice, column_fastest=order))

        for order in total:
            run(order)
        torch.cuda.synchronize()
        equal = torch.equal(outs[False], outs[True])
        times = {order: [] for order in total}
        for block in range(blocks):
            for order in ((False, True) if block % 2 == 0 else (True, False)):
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(reps):
                    run(order)
                stop.record()
                torch.cuda.synchronize()
                times[order].append(start.elapsed_time(stop) / reps * 1e3)
        med = {order: sorted(v)[len(v) // 2] for order, v in times.items()}
        for order in total:
            total[order] += med[order]
        print(f"l={ell} rows={rows} A {rows * 2048 / 2**20:.0f} MiB rule={'columns' if rule else 'rows'}: "
              f"rows-first {med[False]:.1f} us, columns-together {med[True]:.1f} us, "
              f"ratio {med[True] / med[False]:.3f}, bitwise equal {equal}")
    print(f"sum rows-first {total[False]:.1f} us, columns-together {total[True]:.1f} us, "
          f"ratio {total[True] / max(total[False], 1e-9):.3f}")


if __name__ == "__main__":
    main()
