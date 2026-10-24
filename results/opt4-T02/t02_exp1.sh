#!/usr/bin/env bash
# OPT4-T02 experiment 1: baseline and fx_linear module profiles, NaCl and NbCl5 (untimed).
export OUT=$HOME/fastmace-opt4/T02/exp1
TAG=-base EXTRA= bash ~/fastmace-opt4/t02_profile.sh
TAG=-fxlin EXTRA="--experiment fx_linear" bash ~/fastmace-opt4/t02_profile.sh
