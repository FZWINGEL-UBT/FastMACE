"""End-to-end check of the owner-pass forward (ideas plan idea 14(A), ``fused_forward.OWNER_PASS``).

One process per side (the compiled model is cached per process). Builds the plan's engine (OMAT-0 medium, SrTiO3
10x10x10 by default, the hybrid backend, NVE or Langevin at 300 K, 1 fs), and:

- at the start and every ``--check-every`` steps, evaluates the force field at the current positions. With
  ``--owner on`` it captures the owner op's inputs and requires the two passes to give the one-pass block kernel's
  message bit for bit on them, and every partner edge's folded input a to equal its owner's; it also evaluates the
  field again and with the owner pairs unbound (the one-pass block forward on the same owned-first edge order) and
  reports the force differences beside the repeat's (the backward's sender sums are atomic, so forces are not
  bitwise reproducible);
- runs ``--steps`` MD steps through the engine's graphs, counting the neighbour-list rebuilds;
- times ``--time-steps`` further steps (CUDA events, after a flush);
- writes the start's energy and forces, the energies along the run and the timings to ``<output>/<side>.npz`` and
  ``<side>.json``.

``--compare DIR`` then compares the sides' start forces (rounding: the owned-first order changes the message sums).

Usage (hybrid environment, FastMACE on PYTHONPATH):
  python tools/owner_pass_check.py --checkpoint mace-omat-0-medium.model --output DIR --owner on
  python tools/owner_pass_check.py --checkpoint mace-omat-0-medium.model --output DIR --owner off
  python tools/owner_pass_check.py --compare DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402


def evaluate(field, positions, owner_bound):
    """(energy, forces) of the field at ``positions``; ``owner_bound=False``: the owner pairs unbound for this call."""
    neighbours = field.neighbours
    saved = getattr(neighbours, "owner_pairs", False)
    if not owner_bound:
        neighbours.owner_pairs = False
    try:
        out = field(positions)
        return out.energy.detach().double().cpu().clone(), out.forces.detach().double().cpu().clone()
    finally:
        neighbours.owner_pairs = saved


def run(args):
    ab_step_time.prepare_fastmace_hybrid()
    import numpy as np
    import torch

    from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
    from fastmace.kernels.oeq_streaming import fused_forward
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    fused_forward.OWNER_PASS = args.owner == "on"
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"), neighbours=NeighbourConfig(skin=args.skin))
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    name, base = ab_step_time.read_structure(args.structure, None)
    field = ForceField(loaded, base, precision="float32", rewrites="auto")
    integrator = IntegratorConfig(ensemble=args.ensemble, temperature_K=300.0, timestep_fs=1.0, seed=7)
    plan = planner.plan(field, base, integrator, Workload(max_temperature_K=300.0), planner.hardware())
    engine = planner.build(plan, field, base.copy(), temperature_K=300.0)
    engine.prepare()
    field = engine.force_field
    lists = field.neighbours
    owner_kernels = sum(getattr(m._modules.get("wprime_backward"), "owner", None) is not None for m in field._streaming)
    report = dict(side=args.owner, structure=name, natoms=len(base), gpu=torch.cuda.get_device_name(),
                  owner_pairs=bool(getattr(lists, "owner_pairs", False)), owner_interactions=int(owner_kernels),
                  checks=[], fastmace_commit=ab_step_time.fastmace_commit())
    if args.owner == "on" and not (report["owner_pairs"] and owner_kernels):
        raise SystemExit(f"OWNER_PASS on, but owner_pairs={report['owner_pairs']} owner interactions={owner_kernels}"
                         f" sort_receivers={lists.sort_receivers} sender_groups={lists.sender_groups is not None}"
                         " (is the plan's block message with TABLE_W and symmetric radial rows on?)")

    def engine_rebuilds():  # the engine's count (graph rebuilds included) beside the overlapped ones
        return int(getattr(engine, "rebuilds", 0)) + int(getattr(engine, "rebuilds_beside", 0))

    report["skin_A"] = float(lists.skin)
    captured = {}
    original = fused_forward.owner_message

    def hook(pass1, pass2, *tensors):  # the owner op's inputs of the checked call
        if captured.get("armed"):
            captured.update(armed=False, pass1=pass1, pass2=pass2, args=[t.clone() for t in tensors[:-1]])
        return original(pass1, pass2, *tensors)

    fused_forward.owner_message = hook

    def check(step):
        engine.flush()
        torch.cuda.synchronize()
        positions = engine.state.positions
        captured["armed"] = True
        energy, forces = evaluate(field, positions, True)
        row = dict(step=step, rebuilds=engine_rebuilds(), list_builds=int(lists.rebuilds),
                   energy=float(energy.reshape(-1)[0]), edges=int(lists.count))
        if report["owner_pairs"]:
            count = int(lists.count)
            owned, mirror = lists.owned[:count], lists.mirror_edge[:count].long()
            row.update(owned=int(owned.sum()), without_mirror=int((mirror < 0).sum()))
            # The message: the two passes against the one-pass block kernel on the captured inputs, bit for bit;
            # and every partner edge's folded input a equal to its owner's.
            a, weight, x, sh, senders, receivers, row_ptr, mid, mirror32 = captured["args"]
            partner = torch.nonzero(~owned).flatten()
            row["partner_a_equal_owner"] = bool(torch.equal(a[partner], a[mirror[partner]]))
            one = next(m._modules["wprime_backward"].block[2] for m in field._streaming
                       if getattr(m._modules.get("wprime_backward"), "owner", None) is not None)
            single = fused_forward.fused_folded_message_blocks(a, weight, x, sh, senders, receivers, row_ptr, one.id)
            passes = fused_forward.fused_folded_message_blocks_owner(a, weight, x, sh, senders, receivers, row_ptr,
                                                                     mid, mirror32, captured["pass1"].id,
                                                                     captured["pass2"].id)
            row["message_bitwise"] = all(torch.equal(u, v) for u, v in zip(single, passes))
            # The forces: against the one-pass forward on the same edge order (unbound owner pairs), beside the
            # noise floor of two owner-pass calls (the backward's sender sums are atomic).
            again_energy, again_forces = evaluate(field, positions, True)
            one_energy, one_forces = evaluate(field, positions, False)
            row.update(energy_bitwise_one_pass=bool(torch.equal(energy, one_energy)),
                       max_force_diff_one_pass=float((forces - one_forces).abs().max()),
                       max_force_diff_repeat=float((forces - again_forces).abs().max()),
                       energy_diff_one_pass=float((energy - one_energy).abs().max()))
        report["checks"].append(row)
        print(json.dumps(row), flush=True)
        engine.invalidate_graphs("owner_pass_check: eager force calls")
        return energy, forces

    energy0, forces0 = check(0)
    energies = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        engine.step()
        if step % args.check_every == 0:
            check(step)
        if step % 10 == 0:
            energies.append((step, engine.potential_energy()))
    engine.flush()
    torch.cuda.synchronize()
    report["md_seconds"] = time.perf_counter() - started
    report["rebuilds"] = engine_rebuilds()
    if args.time_steps:
        for _ in range(20):
            engine.step()
        engine.flush()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.time_steps):
            engine.step()
        end.record()
        end.synchronize()
        report["ms_per_step"] = start.elapsed_time(end) / args.time_steps
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / f"{args.owner}.npz", energy0=energy0.numpy(), forces0=forces0.numpy(),
             positions0=base.get_positions(), energies=np.array(energies, dtype=float))
    (args.output / f"{args.owner}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}), flush=True)
    if report["owner_pairs"] and not all(row["message_bitwise"] and row["partner_a_equal_owner"]
                                         for row in report["checks"]):
        raise SystemExit("the owner-pass message differs from the one-pass forward on the same edge order")


def compare(directory):
    import numpy as np
    on, off = np.load(directory / "on.npz"), np.load(directory / "off.npz")
    df = np.abs(on["forces0"] - off["forces0"])
    scale = np.abs(off["forces0"]).max()
    out = dict(max_force_diff=float(df.max()), rms_force_diff=float(np.sqrt((df ** 2).mean())),
               max_force=float(scale), relative=float(df.max() / scale),
               energy_diff=float(abs(on["energy0"].reshape(-1)[0] - off["energy0"].reshape(-1)[0])))
    steps = min(len(on["energies"]), len(off["energies"]))
    if steps:
        out["energy_diff_along_run"] = [(int(on["energies"][i, 0]), float(on["energies"][i, 1] - off["energies"][i, 1]))
                                        for i in range(0, steps, max(1, steps // 10))]
    print(json.dumps(out, indent=1))
    (directory / "compare.json").write_text(json.dumps(out, indent=1) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--owner", choices=("on", "off"), default="on")
    p.add_argument("--ensemble", default="nvt")
    p.add_argument("--skin", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--check-every", type=int, default=100)
    p.add_argument("--time-steps", type=int, default=200)
    p.add_argument("--output", type=Path)
    p.add_argument("--compare", type=Path)
    args = p.parse_args(argv)
    if args.compare:
        return compare(args.compare)
    if args.checkpoint is None or args.output is None:
        p.error("--checkpoint and --output are needed for a run")
    run(args)


if __name__ == "__main__":
    main()
