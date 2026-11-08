#!/usr/bin/env bash
# leto: stage-1 A/B packs with the Ada's prerequisites (Benchmark_loup, both build orders).
J=/mnt/homes/zwingel/fm-ada-capacity
ROOT=$J/runs/ab-leto bash $J/jobs/ab.sh $J/src-73e421ea s1a s1b s1c
