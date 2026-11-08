#!/usr/bin/env bash
# murphy: the four A/B packs (Benchmark_loup, both build orders).
J=/mnt/homes/zwingel/fm-ada-capacity
ROOT=$J/runs/ab-murphy bash $J/jobs/ab.sh $J/src-73e421ea pack1a pack1b pack2a pack2b pack2c
