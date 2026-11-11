#!/bin/bash
O=$HOME/fm-runs/binned-5060-20261111/out
ls -la $O | head -30
python3 /mnt/d/Margraf/FastMACE/results/binned-screen-5060ti-20261111/jobs/show_ab.py $O | cut -c1-1500
tail -6 $O/ab-16x16x16-ab.log | cut -c1-600
