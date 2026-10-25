"""Writes the mkdocstrings stub pages under docs/api/ (6-T06). Run from the repository root."""

from pathlib import Path

PAGES = {
    "quick.md": ("Quick runs", ["fastmace.quick.md", "fastmace.quick.RunResult"]),
    "model.md": ("Models and force fields", ["fastmace.model.loading.load_pristine",
                                             "fastmace.model.forcefield.ForceField",
                                             "fastmace.planner.workload.Workload"]),
    "simulation.md": ("Simulation", ["fastmace.simulation.Simulation", "fastmace.simulation.Committed"]),
    "integrators.md": ("Integrators", [("fastmace.integrators", ["Langevin", "LangevinBAOAB", "NVE", "CSVR",
                                                                 "NoseHooverChain", "MTK", "Berendsen"])]),
    "optimize.md": ("Optimisers", [("fastmace.optimize", ["LBFGS", "FIRE", "FrechetCell"])]),
    "mc.md": ("Monte Carlo", [("fastmace.mc.moves", ["Swap", "Transmute", "Displace", "configurational_mu"])]),
    "protocols.md": ("Protocols and schedules", [("fastmace.protocols.schedule", ["ramp", "hold", "Schedule",
                                                                                 "Segment"]),
                                                 ("fastmace.protocols.profiles", ["PROFILES", "Profile",
                                                                                 "load_profile"])]),
    "output.md": ("Output", [("fastmace.io.output", ["Output", "read_thermo", "read_frames"])]),
    "d3.md": ("Dispersion (D3)", ["fastmace.longrange.d3.D3"]),
    "ase.md": ("ASE front end", [("fastmace.ase.calculator", ["calc", "FastMACECalculator"]),
                                 ("fastmace.ase.md", ["Langevin", "LangevinBAOAB", "VelocityVerlet", "Bussi",
                                                      "NoseHooverChainNVT", "IsotropicMTKNPT", "MTKNPT",
                                                      "NVTBerendsen", "NPTBerendsen",
                                                      "Inhomogeneous_NPTBerendsen"]),
                                 ("fastmace.ase.optimize", ["LBFGS", "FIRE"])]),
    "errors.md": ("Errors", ["fastmace.errors"]),
}

NOTES = {
    "model.md": "`fm.load` is `fastmace.model.loading.load_pristine`.\n\n",
    "errors.md": "Every FastMACE error derives from `FastMACEError`.\n\n",
}


def main():
    root = Path("docs/api")
    for name, (title, entries) in PAGES.items():
        text = f"# {title}\n\n" + NOTES.get(name, "")
        for entry in entries:
            if isinstance(entry, tuple):
                path, members = entry
                text += f"::: {path}\n    options:\n      members: [{', '.join(members)}]\n\n"
            else:
                text += f"::: {entry}\n\n"
        (root / name).write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
