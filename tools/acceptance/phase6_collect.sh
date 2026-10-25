#!/bin/bash
# 6-T13: copy the small acceptance records (<1 MB, no trajectories/.npz/models) into results/6-T13-20261025/.
T=$HOME/tmp/6-T13
R=${1:?results dir}
mkdir -p "$R"/{clean-install,colab-l4-py313,colab-l4-py312,traj,examples,corpus}
cp "$T"/{ruff.log,twine.log,sha256.txt,contents.txt,mkdocs.log,clean-installs.log} "$R/" 2>/dev/null
cp "$T"/cpu-suite-2-ruff.log "$R/"
tail -5 "$T/cpu-suite.log" > "$R/cpu-suite-1-tail.log"
grep -E '^(FAILED|ERROR)' "$T/cpu-suite.log" >> "$R/cpu-suite-1-tail.log"
tail -3 "$T/cpu-suite-2.log" > "$R/cpu-suite-2-tail.log"
cp "$T"/clean-*-check.log "$T"/clean-*-install.log "$R/clean-install/"
for d in colab-l4:colab-l4-py313 colab-l4-py312:colab-l4-py312; do
  src=${d%%:*}; dst=${d##*:}
  cp "$T/$src/colab.json" "$R/$dst/" 2>/dev/null
  cp "$T/$src.log" "$R/$dst/driver.log" 2>/dev/null
  for f in "$T/$src"/out/cmd*/*.log "$T/$src"/out/cmd*/gpu.csv "$T/$src"/out/probe.json "$T/$src"/out/setup.json; do
    [ -f "$f" ] && cp "$f" "$R/$dst/$(basename "$(dirname "$f")")-$(basename "$f")"
  done
done
cp "$T"/traj/*-compare.log "$R/traj/"
for f in "$T"/traj/traj-*/trajectory.json; do
  n=$(basename "$(dirname "$f")"); [ "$(stat -c %s "$f")" -lt 1000000 ] && cp "$f" "$R/traj/$n.json"
done
cp "$T"/traj-cpu.log "$T"/traj-repro.log "$R/traj/" 2>/dev/null
grep -v Warning "$R/traj/traj-cpu.log" | grep -E 'start|end' > "$R/traj/traj-cpu-jobs.log"; rm "$R/traj/traj-cpu.log"
cp "$T"/examples.log "$T"/examples/*.log "$R/examples/"
cp "$T"/corpus/compare.txt "$R/corpus/"
find "$R" -size +1M -print -delete
du -sh "$R"
