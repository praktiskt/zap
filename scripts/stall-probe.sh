#!/usr/bin/env bash
# scripts/stall-probe.sh -- diagnose the torture_repartition flake.

# Reproduces the scenario while POLLING shard health + thread states:
#   t+X  shard health snapshot
#   the first stuck shard's non-futex thread wait-channels
#
# Usage: scripts/stall-probe.sh       (runs ~60s, logs -> /tmp/zap-torture/)
# Requires: target/debug build (cargo test --no-run compiles the fixture).

set -uo pipefail
BASE=${BASE:-51012}
BIN=target/debug/zap
LOGDIR=/tmp/zap-torture
mkdir -p "$LOGDIR"
DUR=${DUR:-52}

cleanup() { pkill -x zap 2>/dev/null; }
trap cleanup EXIT

spawn() {
  local port=$1 peers=$2
  "$BIN" serve \
    --bind 127.0.0.1:$port \
    --announce 127.0.0.1:$port \
    --peers "$peers" \
    --lanes 2 \
    ${3:-} >"$LOGDIR/shard-$port.log" 2>&1 &
  echo $!
}

A=$((BASE)); B=$((BASE+1)); C=$((BASE+2))

# A singleton, load it, then let B and C join while we watch
spawn "$A" ""
sleep 1
curl -s -X POST "127.0.0.1:$A/ingest" \
  -d "$(python3 - <<'PYEOF'
import json
docs = [{"id": f"r{i}", "text": f"repart doc {i} fox storm"} for i in range(40)]
print(json.dumps(docs))
PYEOF
)" >/dev/null
echo "loaded 40 docs on $A"

spawn "$B" "127.0.0.1:$A"
sleep 1
spawn "$C" "127.0.0.1:$A,127.0.0.1:$B"

end=$((SECONDS + DUR))
snapped=0
while [ $SECONDS -lt $end ]; do
  line=""
  for p in "$A" "$B" "$C"; do
    code=$(curl -s -m1 -o /dev/null -w '%{http_code}' "127.0.0.1:$p/health" || true)
    line="$line $p=$code"
  done
  echo "t=$SECONDS$line"
  # the FIRST stuck shard (000) gets a thread-state dump, once
  if [ "$snapped" = 0 ] && echo "$line" | grep -q '=000'; then
    stuck=$(echo "$line" | tr ' ' '\n' | grep '=000' | head -1 | cut -d= -f1)
    pid=$(pgrep -f "bind 127.0.0.1:$stuck" | head -1 || true)
    if [ -n "$pid" ]; then
      echo "-- thread snapshot for stuck shard $stuck (pid $pid):"
      for t in /proc/$pid/task/*; do
        w=$(cat "$t/wchan" 2>/dev/null || echo gone)
        printf '  %s %s\n' "$(basename $t)" "$w"
      done | grep -v futex_do_wait | head -8
      snapped=1
    fi
  fi
  sleep 1.5
done
cleanup
echo "done — shard logs in $LOGDIR"
