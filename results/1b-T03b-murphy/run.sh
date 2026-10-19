cd /mnt/homes/zwingel/fastmace-1b-T03b
. results/1b-T03b-murphy/env.sh
export FASTMACE_RESULTS=/mnt/homes/zwingel/fastmace-1b-T03b/results/1b-T03b-murphy/reports
echo "== repro on, fixed $(date +%T)"; $PY results/1b-T03b-murphy/repro.py on
echo "== kernels $(date +%T)"; $PY results/1b-T03b-murphy/kernels.py
echo "== tests $(date +%T)"
$PY -m pytest -p no:cacheprovider -rs -v -s --basetemp=/mnt/homes/zwingel/fastmace-cache/pytest-1b-T03b \
  tests/gpu/test_engine_gpu.py -k "against_the_research_engine" -m slow 2>&1 | grep -vE "^\s*$" | grep -E "PASSED|FAILED|SKIPPED|passed|failed|^E |Error" 
echo "== done $(date +%T)"
