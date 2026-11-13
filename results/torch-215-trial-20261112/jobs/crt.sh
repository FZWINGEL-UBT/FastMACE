B=/mnt/homes/zwingel/t215; export UV_CACHE_DIR=$B/uv-cache TMPDIR=$B/tmp
for E in /mnt/homes/zwingel/environments/mace_cueq_oeq_torch215 /mnt/homes/zwingel/environments/mace_cueq_latest_torch215; do
  $B/uvpkg/bin/uv pip install -q --python $E/bin/python nvidia-cuda-crt==13.2.86 && ls $E/lib/python3.11/site-packages/nvidia/cu13/include/crt/host_defines.h
  $B/uvpkg/bin/uv pip freeze --python $E/bin/python > $E.requirements.txt
done
rm -rf $B/runs/leeloo/cache/b/oeq $B/runs/leeloo/out/diag
