#!/usr/bin/env bash
# OPT4-T02: NaCl-only timed A/B rerun, both orders, warm-up 3000 steps (one quiet reservation).
export STRUCTS=NaCl WARMUP=3000 TAG=-nacl-w3000
bash ~/fastmace-opt4/t02_ab.sh ab abab
bash ~/fastmace-opt4/t02_ab.sh ba abba
