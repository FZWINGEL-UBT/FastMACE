#!/bin/bash
# Build the fastmace-oeq-ext wheels (CPython 3.10-3.13, torch 2.14.1+cu130) in a manylinux_2_28 container.
#
#   tools/release/oeq_wheels/build.sh [OUTPUT_DIR]      (default ~/tmp/6-T04/wheels)
#
# Run from WSL or Linux. DOCKER selects the docker client (default: docker, else docker.exe, the Windows
# client of Docker Desktop when its WSL integration is off). The scratch tree (pip cache, torch per Python,
# build trees, ~10 GB) is the docker volume fastmace-oeq-ext-work; the wheels are streamed out with tar,
# so no bind mount of a WSL path is needed. Nothing is uploaded: publishing to PyPI is a separate,
# owner-run `twine upload` (plans/phase-6.md D4).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$HOME/tmp/6-T04/wheels}
image=${MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}
docker=${DOCKER:-$(command -v docker >/dev/null && docker info >/dev/null 2>&1 && echo docker || echo docker.exe)}
mkdir -p "$out"

# The source directory goes in as a tar stream as well (works with either client).
"$docker" volume create fastmace-oeq-ext-work >/dev/null
"$docker" volume rm -f fastmace-oeq-ext-out >/dev/null 2>&1 || true
"$docker" volume create fastmace-oeq-ext-out >/dev/null
tar -C "$here" -cf - build_in_container.sh package | "$docker" run --rm -i \
  -v fastmace-oeq-ext-work:/work -v fastmace-oeq-ext-out:/out \
  -e TORCH_VERSION -e OEQ_VERSION -e EXT_VERSION -e PYTHONS \
  "$image" bash -c 'mkdir -p /src && tar -C /src -xf - && bash /src/build_in_container.sh' 
"$docker" run --rm -v fastmace-oeq-ext-out:/out "$image" tar -C /out -cf - . | tar -C "$out" -xf -
ls -l "$out"
