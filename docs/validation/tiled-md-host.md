# Tiled path: MD at 70.96M with the integrator's state on the host

## The problem

Until now, `tools/bench/tiled_md.py` kept the integrator's state on the device: the positions, velocities, forces
and masses, as float64. That is 72+ bytes per atom, plus the update temporaries. At 40-70M atoms it does not fit
beside the tiled call, so the largest MD sizes were measured only as repeated force calls
([tiled-leftovers.md](tiled-leftovers.md)).

## The change (tool only; the library is unchanged)

`tiled_md.py --integrator`:

- **`host` (the default):** the state lives on the host as float64 CPU tensors.
  - Each call gets a float64 device copy of the positions, freed after the call. That is what
    `tiled_capacity.py` already keeps on the device.
  - The tiled call bins those float64 positions. It cannot bin the force field's own float32 positions, which
    would change the tiles.
- **`device`:** the state stays on the device, as before.
- **`both`:** the host integrator runs with a device integrator beside it. The device integrator gets the same
  forces each step, and the two states are compared bitwise every step.
  - Why not two runs: the device's forces change in their last bits from run to run (about 1e-6 eV/Å between two
    arms in one process), so two separate runs cannot be compared bitwise.
  - Why the integrators should match: they apply the same float64 element-wise operations in the same order.
    Only the kinetic-energy sum is reduced in another order.

`--no-list-cache` turns off the tile-list cache, which saves about 270 bytes per atom on the host at the largest
sizes.

## Gate 1: bitwise at 884,736 atoms (murphy)

Run `results/tiled-md-host-20261101/runs/murphy-mdh-both-NbCl5-16x16x48`, at 15e21f7c:
- compiled, 20 steps of 2 fs at 600 K, `--integrator both`;
- **the host and device positions and velocities are bitwise equal at all 20 steps**;
- drift 0.050 meV/atom.

## Gate 2: NVE at 70.96M atoms (murphy)

Run `runs/murphy-nve-host-NbCl5-69x69x207`. It used 3baf655f, a local merge of this branch with
tiled-slot-growth; the merge is not pushed.

**Setup:**
- compiled, with the default budget (R = 16,384 after the first call);
- tile lists, the host state aliased, the species index;
- no list cache;
- 3 steps of 2 fs at 600 K.

**Host guard:** 3,330 bytes per atom, keeping 20 GB free. 265 GB were available and the guard required 236 GB.

| Step | Wall time | E_pot (eV) | E_kin (eV) | Edges | Slot changes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | | -753,800,986.8 | 5,504,129.5 | 1,864,617,084 | 2 |
| 1 | 1,257 s | -753,764,939.1 | 5,468,213.8 | 1,891,643,690 | 4 (the budget's switch) |
| 2 | 1,194 s | -753,658,936.6 | 5,362,603.0 | 1,908,418,414 | 4 |
| 3 | 1,279 s | -753,489,317.7 | 5,193,618.6 | 1,911,718,922 | 5 (the small slot, once) |

- **Drift over the 3 steps: 0.016 meV/atom.**
- **MD at 70.96M NbCl5 atoms runs on 20 GB.**
- A step takes about 1,200-1,280 s. The repeated call took 1,034 s with the list cache. The difference is the
  cache, which is off here, and the variation from slot changes on a 3-step run. The host integrator's own time
  is not measured separately.

## Left over

- The library's integrators (`fastmace.integrators`) do not use the tiled force field yet; the planner does not
  pick it. This is the benchmark tool's integrator.
