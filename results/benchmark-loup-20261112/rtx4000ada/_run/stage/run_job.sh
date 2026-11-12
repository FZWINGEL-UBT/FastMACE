# usage: run_job.sh <name> <item>...   item = srtio3 | cantor-inputs | cu111_madsen | omat0:<NXxNY>
# Runs items one after another; before EACH item waits until idle.sh says the host is idle. Status: $BASE/status/<name>.status
source /mnt/homes/zwingel/benchmark-loup-20261112/stage/env.sh
NAME=$1; shift
ST=$BASE/status/$NAME.status; OUT=$BASE/results; R=$BASE/Benchmark_loup/reproduce
mkdir -p $BASE/status $BASE/logs $OUT
s() { echo "$(date '+%F %T %Z') $(hostname) $*" >> $ST; }
trap 'rm -rf "$JITROOT"; s "JIT cache $JITROOT removed"' EXIT
s "START items: $*"
for it in "$@"; do
  while ! bash $BASE/stage/idle.sh > $BASE/logs/$NAME.idle.txt 2>&1; do s "WAIT busy: $(grep BUSY $BASE/logs/$NAME.idle.txt | head -2 | tr '\n' ' ')"; sleep 300; done
  s "BEGIN $it"
  L=$BASE/logs/$NAME.$(echo $it | tr ':' '_').log
  case $it in
    srtio3)        $R/run_all.sh --out $OUT > $L 2>&1 ;;
    cantor-inputs) $R/run_all.sh --system cantor --out $OUT --inputs-only > $L 2>&1 ;;
    cu111_madsen)  $R/run_all.sh --system cu111_h2o --size 4x5 --out $OUT > $L 2>&1 ;;
    omat0:*)       $R/run_all.sh --system cu111_h2o_omat0 --size ${it#omat0:} --out $OUT > $L 2>&1 ;;
    *) echo "unknown item $it" > $L; false ;;
  esac
  rc=$?
  s "END $it rc=$rc"
done
s "ALLDONE"
