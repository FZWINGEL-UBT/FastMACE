sleep 5; while systemctl --user is-active -q t215-trial; do sleep 60; done
echo "unit ended $(date -u +%FT%TZ)"; tail -12 /home/fzwin/fm-t215/out/run.log
