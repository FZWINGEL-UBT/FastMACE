sleep 5; while systemctl --user is-active -q b4-shuffle; do sleep 60; done
echo "unit ended $(date -u +%FT%TZ)"; tail -12 /home/fzwin/fm-b4/out/run.log
