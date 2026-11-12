# background install job (leeloo). status in $BASE/status/install.status
source /mnt/homes/zwingel/benchmark-loup-20261112/stage/env.sh
mkdir -p $BASE/status $BASE/logs
ST=$BASE/status/install.status
echo "RUNNING $(hostname) $(date '+%F %T %Z') pid $$" > $ST
cd $BASE/Benchmark_loup/reproduce
./install.sh --prefix $PREFIX --jobs $JOBS > $BASE/logs/install.log 2>&1
rc=$?
echo "DONE rc=$rc $(hostname) $(date '+%F %T %Z')" > $ST
