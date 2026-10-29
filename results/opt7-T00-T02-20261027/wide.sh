#!/usr/bin/env bash
cd ~/tmp/opt7
PYTHONPATH=/mnt/d/Margraf/fm-f64:. PYTHONDONTWRITEBYTECODE=1 ~/environments/mace_cueq_oeq/bin/python wide.py $(ls ops-*.pt) > wide-local.jsonl 2> wide-local.err
echo ALLDONE >> wide-local.jsonl
