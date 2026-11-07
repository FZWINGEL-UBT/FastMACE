"""Slim source tarball of a commit for Modal jobs (AGENTS.md "Using resources": never ``--commit`` clients, which
hold about 9 GB each in WSL). Drops ``docs/``, ``runs/`` and ``results/`` (transcript zips, profiles) except the
small inputs the job templates read (``KEEP_RESULTS``, e.g. SrTiO3.cif); keeps code, tests and tools, and adds
``.source_commit``. About 13 MB instead of 635 MB.

Usage: python3 tools/jobs/slim_tar.py COMMIT OUT.tar [REPO]
"""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DROP = ("docs/", "runs/", "results/")
KEEP_RESULTS = ("results/loup-a100-20261103/", "results/ideas-queue2-20261106/")


def keep(name: str) -> bool:
    """Whether a file of the commit goes into the slim tarball."""
    return name.startswith(KEEP_RESULTS) or not name.startswith(DROP)


def resolve(commit: str, repo=REPO) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", f"{commit}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()


def slim_tar(commit: str, out, repo=REPO):
    """Write the slim tarball of ``commit`` to ``out``; returns (sha, kept bytes, dropped bytes)."""
    sha = resolve(commit, repo)
    proc = subprocess.Popen(["git", "-C", str(repo), "-c", "core.autocrlf=false", "archive", "--format=tar", sha],
                            stdout=subprocess.PIPE)
    kept = dropped = 0
    with tarfile.open(fileobj=proc.stdout, mode="r|") as src, tarfile.open(out, "w") as dst:
        for member in src:
            if member.isfile() and not keep(member.name):
                dropped += member.size
                continue
            dst.addfile(member, src.extractfile(member) if member.isfile() else None)
            kept += member.size
        info = tarfile.TarInfo(".source_commit")
        payload = (sha + "\n").encode()
        info.size = len(payload)
        dst.addfile(info, io.BytesIO(payload))
    if proc.wait() != 0:
        raise RuntimeError(f"git archive {sha} failed")
    return sha, kept, dropped


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    sha, kept, dropped = slim_tar(sys.argv[1], sys.argv[2], *(sys.argv[3:] or [REPO]))
    print(f"{sha} kept {kept / 1e6:.1f} MB dropped {dropped / 1e6:.1f} MB -> {sys.argv[2]}")
