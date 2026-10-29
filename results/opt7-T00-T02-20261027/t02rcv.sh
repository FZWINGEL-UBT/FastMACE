#!/usr/bin/env bash
cd ~/tmp/opt7
export PYTHONPATH=/mnt/d/Margraf/fm-f64:. PYTHONDONTWRITEBYTECODE=1
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,lts__t_requests_srcunit_tex_op_red.sum,lts__t_requests_srcunit_tex_op_atom.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
rm -f t02rcv.txt
for f in ops-mh1-r2scan-NbCl5.pt ops-mh1-r2scan-NaNbO3.pt; do
  echo "== $f" >> t02rcv.txt
  /usr/local/cuda-13.3/bin/ncu --profile-from-start off --cache-control all --clock-control none --metrics $M --csv \
    ~/environments/mace_cueq_oeq/bin/python ncu_probe_rcv.py $f >> t02rcv.txt 2> t02rcv-$f.err
done
echo ALLDONE >> t02rcv.txt
