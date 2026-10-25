#!/bin/bash
# 6-T13 item 3a: clean installs from the built 1.0.0 artefacts, Python 3.10/3.11/3.12 (wheel) and 3.10 (sdist).
set -u
T=~/tmp/6-T13
repo=$T/freeze2
wheels=$T/oeq-wheels
mkdir -p "$wheels" && cp ~/tmp/6-T04/wheels/*.whl "$wheels/"
whl=$(ls $T/dist/*.whl); sdist=$(ls $T/dist/*.tar.gz)
cd /mnt/d/Margraf/fm-6-T13
run() {  # name python artefact
  local name=$1 py=$2 art=$3 root=$T/clean-$1
  echo "=== $name install $(date -Is)"
  CLEAN_INSTALL_ROOT=$root CLEAN_INSTALL_FASTMACE=$art CLEAN_INSTALL_STAGE=install \
    bash $repo/tools/release/clean_install.sh "$wheels" "$py" > $T/clean-$name-install.log 2>&1
  echo "install exit $?"
  echo "=== $name check $(date -Is)"
  CLEAN_INSTALL_ROOT=$root CLEAN_INSTALL_FASTMACE=$art CLEAN_INSTALL_STAGE=check \
    python3 tools/ci/local_gpu.py run --purpose "6-T13 clean install $name" -- \
    bash $repo/tools/release/clean_install.sh "$wheels" "$py" > $T/clean-$name-check.log 2>&1
  echo "check exit $?"
}
run whl-py3.10 3.10 "$whl"
run whl-py3.11 3.11 "$whl"
run whl-py3.12 3.12 "$whl"
run sdist-py3.10 3.10 "$sdist"
echo ALLDONE
