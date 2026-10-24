#!/usr/bin/env bash
# OPT4-T02 experiment 2: module profiles base / fx_e3nn_linear / polar_graph_sums / both, NaCl and NbCl5 (untimed).
export OUT=$HOME/fastmace-opt4/T02/exp3
TAG=-base EXTRA= bash ~/fastmace-opt4/t02_profile.sh
TAG=-fx EXTRA="--rewrites fx_e3nn_linear=on" bash ~/fastmace-opt4/t02_profile.sh
TAG=-gs EXTRA="--rewrites polar_graph_sums=on" bash ~/fastmace-opt4/t02_profile.sh
TAG=-both EXTRA="--rewrites fx_e3nn_linear=on polar_graph_sums=on" bash ~/fastmace-opt4/t02_profile.sh
