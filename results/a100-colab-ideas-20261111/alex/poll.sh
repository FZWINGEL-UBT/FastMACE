#!/bin/bash
# Poll the two Alex jobs every 15 min (one ssh call each); exit when both have left the queue.
S=/c/Windows/System32/OpenSSH
while true; do
  q=$(timeout 120 $S/ssh.exe alex "squeue -j 4981264,4981265 -h -o '%i %T %M %R'" 2>&1)
  echo "$(date -u +%H:%MZ) ${q:-left the queue}"
  [ -z "$q" ] && break
  sleep 900
done
