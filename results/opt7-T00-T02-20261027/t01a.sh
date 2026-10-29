#!/usr/bin/env bash
cd ~/tmp/opt7
PYTHONPATH=/mnt/d/Margraf/fm-f64:. PYTHONDONTWRITEBYTECODE=1 ~/environments/mace_cueq_oeq/bin/python t01a.py $(ls ops-*.pt) > t01a.jsonl 2> t01a.err
echo ALLDONE >> t01a.jsonl
