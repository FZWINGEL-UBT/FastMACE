#!/usr/bin/env bash
# Build and check FastMACE's sdist and wheel from a git tag (docs/development.md, "Releasing"; plan task 6-T12).
#
#   tools/release/make_release.sh <tag> [python]
#
# git archive of <tag> -> python -m build -> twine check, in a fresh directory under ${TMPDIR:-~/tmp}.
# It prints the upload command and NEVER uploads: the owner runs `twine upload` with their own token.
set -euo pipefail

tag="${1:?usage: make_release.sh <tag> [python]}"
python="${2:-$HOME/environments/fastmace_cpu/bin/python}"
repo="$(git rev-parse --show-toplevel)"

commit="$(git -C "$repo" rev-parse --verify "${tag}^{commit}")"
base="${TMPDIR:-$HOME/tmp}"
mkdir -p "$base"
work="$(mktemp -d "$base/fastmace-release-${tag}-XXXXXX")"
echo "tag ${tag} = ${commit}; working in ${work}"

# A clean tree of the tag only (no untracked files, LF endings from the repository).
mkdir -p "$work/src"
git -C "$repo" -c core.autocrlf=false archive --format=tar "$commit" | tar -x -C "$work/src"

version="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$work/src/fastmace/__init__.py")"
echo "fastmace.__version__ at ${tag}: ${version}"

PYTHONDONTWRITEBYTECODE=1 "$python" -m build --sdist --wheel --outdir "$work/dist" "$work/src"
"$python" -m twine check --strict "$work/dist"/*

for file in "$work/dist"/*; do
  case "$file" in
    *"-${version}"[.-]*) ;;
    *) echo "error: $file does not carry version ${version}" >&2; exit 1 ;;
  esac
done
( cd "$work/dist" && sha256sum ./* )

cat <<EOF

Built and checked; nothing was uploaded. To publish (the owner, with their own PyPI API token):

  TWINE_USERNAME=__token__ TWINE_PASSWORD=<pypi-token> \\
    $python -m twine upload $work/dist/*

EOF
