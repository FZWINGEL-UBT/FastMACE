# Exits (printing it) when a watched run logs a new DONE / STOP / exit=[1-9] line; seen lines go to seen.txt.
touch seen.txt
while true; do
  for spec in "leto leto-A134" "murphy murphy-polar" "murphy murphy-off" "murphy murphy-A2"; do
    set -- $spec
    lines=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 $1 "grep -hE 'DONE|STOP|exit=[1-9]' /mnt/homes/zwingel/fm-ada5060/runs/$2/run.log 2>/dev/null" < /dev/null)
    while IFS= read -r l; do [ -n "$l" ] && ! grep -qxF "$2: $l" seen.txt && { echo "$2: $l" | tee -a seen.txt; new=1; }; done <<< "$lines"
  done
  lines=$(wsl -e bash -lc "grep -hE 'DONE|STOP|exit=[1-9]|group' ~/fm-ada5060/runs/local/run.log ~/fm-ada5060/runs/local/chain.log ~/fm-ada5060/runs/local-polar/run.log 2>/dev/null" < /dev/null)
  while IFS= read -r l; do [ -n "$l" ] && ! grep -qxF "local: $l" seen.txt && { echo "local: $l" | tee -a seen.txt; new=1; }; done <<< "$lines"
  [ -n "${new:-}" ] && exit 0
  sleep 300
done
