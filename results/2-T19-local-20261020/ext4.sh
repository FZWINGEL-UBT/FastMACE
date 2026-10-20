set -u
i=0
for s in fill 1a 1b 1b 1a 1a 1b; do
  side=$s; [ $s = fill ] && side=1b
  name=ext4-$i-$s; [ $s = fill ] && name=ext4-fill-1b
  { date -u +%FT%TZ; uptime; nvidia-smi --query-compute-apps=pid --format=csv,noheader; } > $name.idle.txt
  /home/fzwin/environments/mace_cueq_oeq/bin/python /home/fzwin/fastmace-t19/checkout/results/2-T19-probes/setup_stages.py --output /home/fzwin/fastmace-t19/2-T19-local-20261020/$name --side $side --checkpoint /home/fzwin/models/mace-mh-1.model --renamed /home/fzwin/fastmace-t19/versions > $name.log 2>&1 || echo FAILED $name
  i=$((i+1))
done
