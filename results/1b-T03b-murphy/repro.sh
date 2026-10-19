cd /mnt/homes/zwingel/fastmace-1b-T03b
. results/1b-T03b-murphy/env.sh
for args in "on 200" "off 200" "on"; do echo "== $args $(date +%T)"; $PY results/1b-T03b-murphy/repro.py $args; done
echo "== done $(date +%T)"
