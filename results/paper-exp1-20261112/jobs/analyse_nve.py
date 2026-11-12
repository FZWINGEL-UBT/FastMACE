"""NVE table: analyse_nve.py DIR... (nve-*.json of tools/nve_energy.py) -> markdown on stdout.
Drift = least-squares slope of (E_pot + E_kin)/N, ± the standard error inflated for the residuals' lag-1
autocorrelation; RMS and range about / of the series; temperatures over the whole run, first and last tenth."""
import json
import sys
from pathlib import Path

print("| GPU | run | drift (meV/atom/ps) | RMS (meV/atom) | range (meV/atom) | mean T (K) | T first / last tenth (K) | "
      "rebuilds | wall ms/step | start state |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for directory in sys.argv[1:]:
    for path in sorted(Path(directory).glob("nve-*.json")):
        r = json.loads(path.read_text())
        res, md = r["result"], r["md"]
        start = r["start_state"].split(" sha256 ")[-1][:8] if "sha256" in r["start_state"] else r["start_state"]
        print(f"| {r['gpu'].replace('NVIDIA ', '')} | {path.stem[4:]} | {res['drift_meV_per_atom_per_ps']:+.2e} ± "
              f"{res['drift_se_ar1']:.1e} | {res['rms_meV_per_atom']:.2e} | {res['range_meV_per_atom']:.2e} | "
              f"{md['mean_temperature_K']:.1f} | {md['temperature_first_tenth_K']:.1f} / {md['temperature_last_tenth_K']:.1f} | "
              f"{md['neighbour_rebuilds']} | {md['ms_per_step']:.2f} | {start} |")
