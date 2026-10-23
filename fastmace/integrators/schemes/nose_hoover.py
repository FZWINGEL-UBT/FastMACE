"""Nose-Hoover chains (ASE 3.27 ``NoseHooverChainNVT``) and MTK NPT (``IsotropicMTKNPT``, ``MTKNPT``), 4-T03."""

from fastmace.integrators.registry import IntegratorRecord

# ASE 3.27 raises "Current implementation does not support constraints" in IsotropicMTKNPT and MTKNPT, and
# NoseHooverChainNVT counts 3N regardless of them (owner decision D1, recommended (a)).
_NO_FIX_ATOMS = ("ASE's {} does not support constraints, so there is no reference for fixed atoms "
                 "(owner decision D1)")
_CHAIN = (("thermostat_eta", "save", None), ("thermostat_p_eta", "save", None),
          ("kT", "derived", "Integrator.set_temperature"), ("kT_step", "derived", "Integrator.load_state_dict"))
_BAROSTAT = (("barostat_xi", "save", None), ("barostat_p_xi", "save", None), ("p_cell", "save", None),
             ("eps", "save", None), ("cell0", "save", None), ("volume0", "derived", "MTK.load_state_dict"),
             ("pressure", "save", None), ("pressure_step", "derived", "MTK.load_state_dict"))

RECORDS = (
    IntegratorRecord(scheme="nhc", implementation="fastmace.integrators.nose_hoover:NoseHooverChainNVT",
                     ensembles=("nvt",), cell_modes=("fixed",), constraints=("fixcm",),
                     refusals={"fix_atoms": _NO_FIX_ATOMS.format("NoseHooverChainNVT")},
                     ase_class="ase.md.nose_hoover_chain.NoseHooverChainNVT", ase_dof="3n",
                     state_keys=("eta", "p_eta", "mass_temperature_K"), buffers=_CHAIN,
                     conserved="E_pot + E_kin + the thermostat energy (ASE's get_conserved_energy)"),
    # One scheme, two ASE classes: cell_mode "hydrostatic" is IsotropicMTKNPT (state keys eps and cell0 too),
    # "full" MTKNPT; state_keys is their union, and state_dict() holds the keys of the run's cell mode.
    IntegratorRecord(scheme="mtk", implementation="fastmace.integrators.mtk:MTK",
                     ensembles=("npt",), cell_modes=("hydrostatic", "full"), constraints=("fixcm",),
                     refusals={"fix_atoms": _NO_FIX_ATOMS.format("IsotropicMTKNPT and MTKNPT")},
                     ase_class="ase.md.nose_hoover_chain.IsotropicMTKNPT, ase.md.nose_hoover_chain.MTKNPT",
                     ase_dof="3n",
                     state_keys=("eta", "p_eta", "xi", "p_xi", "p_cell", "eps", "cell0", "pressure_au",
                                 "mass_temperature_K"),
                     buffers=_CHAIN + _BAROSTAT,
                     conserved="E_pot + E_kin + thermostat and barostat energies + cell kinetic energy + P V "
                               "(ASE's get_conserved_energy)",
                     report="experimental: not yet qualified"),
)
