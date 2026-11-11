#!/usr/bin/env bash
# src-fix = src-7d18028d (the timing source) plus the registry fix files from 2a62a9e0 (staged in $B/fixfiles).
set -e
B=/mnt/homes/zwingel/fm-ada5060
rm -rf $B/src-fix && cp -a $B/src-7d18028d $B/src-fix
cp $B/fixfiles/spline_forward.py $B/fixfiles/wprime_backward.py $B/src-fix/fastmace/kernels/oeq_streaming/
mkdir -p $B/src-fix/tests/cpu && cp $B/fixfiles/test_spline_forward_registry.py $B/src-fix/tests/cpu/
echo "2a62a9e0 (7d18028d + SplineForward registry fix)" > $B/src-fix/.source_commit
cd $B/src-fix && grep -c "registered_forward(SplineForward" fastmace/kernels/oeq_streaming/wprime_backward.py
