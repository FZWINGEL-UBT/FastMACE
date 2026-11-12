# Idle = no process on the GPU and no other user's (or our other sessions') process using real CPU (> 50 %).
# Logins, sftp transfers and idle editor daemons do not count (owner, 12 Nov: "we just need a free gpu").
# Processes of this run (path benchmark-loup-20261112 in their command line) are ignored.
busy=0
apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
[ -n "$apps" ] && { echo "BUSY gpu: $apps"; busy=1; }
while read -r u pid pcpu comm args; do
  [[ $args == *benchmark-loup-20261112* ]] && continue
  case $u in root) continue;; esac
  if [ "${pcpu%.*}" -ge 50 ]; then echo "BUSY cpu $u $pid $pcpu% $comm ${args:0:120}"; busy=1; fi
done < <(ps -eo user:20,pid,pcpu,comm,args --no-headers)
[ $busy = 0 ] && echo "IDLE $(hostname) $(date '+%F %T %Z')"
exit $busy
