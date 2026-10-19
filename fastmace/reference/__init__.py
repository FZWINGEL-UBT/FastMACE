"""Versioned reference records and corrected reference calculators (SPEC §12.1).

- ``records``: the reference record per (MACE version, model family) and its stable id (1b-T13);
- ``calculators``: MACE's calculator with the record's corrections, float64 reference and float32
  yardstick (1b-T13);
- ``integrators``: the corrected BAOAB reference and its qualification (1b-T09).

Nothing is imported here, so ``import fastmace.reference`` stays cheap.
"""
