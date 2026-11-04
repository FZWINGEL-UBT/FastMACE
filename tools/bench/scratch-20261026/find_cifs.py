import torch
import fastmace as fm
import ase.io
from pathlib import Path

pristine = fm.load("/home/fzwin/models/mace-mh-1.model", head="matpes_r2scan", trusted=True)

# Find CIF files in FastMACE or local copies
cifs = list(Path("/mnt/d/Margraf").rglob("NaCl.cif"))
if cifs:
    struct_dir = cifs[0].parent
    print("Found structures dir:", struct_dir)
else:
    struct_dir = None
    print("No CIF dir found under /mnt/d/Margraf")
