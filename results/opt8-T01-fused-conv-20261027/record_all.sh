#!/usr/bin/env bash
cd ~/tmp/opt8
source ./env.sh
export TORCHDYNAMO_DISABLE=1
rm -f record.txt
for s in NaCl NbCl5 NaNbO3; do
  [ -f ops-mh1-r2scan-$s.pt ] || $P /mnt/d/Margraf/fm-opt8/tools/bench/opt8_record_operands.py mh1-r2scan $s . 2>&1 | grep -E "calls|Error|error" >> record.txt
done
echo ALLDONE >> record.txt
