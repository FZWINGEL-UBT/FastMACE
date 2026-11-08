#!/usr/bin/env bash
# leto (jobs2/: a copy, so the running stage-1 ab.sh stayed unchanged): stage 2, leave one out on A (spline group + tail + tf32x3_gemms and the compose group), both build orders.
J=/mnt/homes/zwingel/fm-ada-capacity
ROOT=$J/runs/ab-leto bash $J/jobs2/ab.sh $J/src-73e421ea a2a a2b a2c
