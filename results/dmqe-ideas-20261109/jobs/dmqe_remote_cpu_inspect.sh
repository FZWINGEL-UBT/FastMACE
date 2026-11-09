#!/usr/bin/env bash
set -eu
ps -u sovic -o pid,ppid,stat,time,etime,args
exit
