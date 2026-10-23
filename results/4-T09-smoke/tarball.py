"""The source tarball of the 4-T09 Modal smoke: ``git archive`` of a commit, ``.source_commit`` and ``inputs/``.

    python3 results/4-T09-smoke/tarball.py REPO COMMIT SHORT_RUN OUT.tar

``inputs/npt_anneal_NbCl5_short_gpu.yaml``: the shipped short NPT protocol with gpu-LBFGS for both stages,
as YAML (JSON) for the image's Python 3.10; ``inputs/3b-T07-short/``: 3b-T07's short run without its quench.
Needs Python 3.11 or later (tomllib).
"""

import io
import json
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path


def main(repo, commit, short_run, out):
    sha = subprocess.run(["git", "-C", repo, "rev-parse", commit], capture_output=True, text=True,
                         check=True).stdout.strip()
    data = subprocess.run(["git", "-C", repo, "-c", "core.autocrlf=false", "archive", "--format=tar", sha],
                          capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as source:
        protocol = tomllib.loads(source.extractfile("fastmace/protocols/data/npt_anneal_NbCl5_short.toml")
                                 .read().decode())
    protocol["structure"] = f"fastmace/protocols/data/{protocol['structure']}"
    protocol["relax"]["optimizer"] = protocol["quench"]["optimizer"] = "gpu-LBFGS"
    with tarfile.open(fileobj=io.BytesIO(data)) as source, tarfile.open(out, "w") as dest:
        for member in source.getmembers():
            dest.addfile(member, source.extractfile(member) if member.isfile() else None)
        for name, payload in ((".source_commit", (sha + "\n").encode()),
                              ("inputs/npt_anneal_NbCl5_short_gpu.yaml", json.dumps(protocol, indent=1).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            dest.addfile(info, io.BytesIO(payload))
        run = Path(short_run)
        for path in sorted(run.iterdir()):
            if path.is_file():
                dest.add(path, arcname=f"inputs/3b-T07-short/{path.name}")
    print(sha, out)


if __name__ == "__main__":
    main(*sys.argv[1:])
