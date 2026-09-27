#!/usr/bin/env bash
# scripts/handoff-probe.sh -- manual live probe of shard movement / resync
#
# Scripts the scenario the tests/handoff.rs integration test covers:
#   1. node A (singleton, --peers empty) ingests 3 id-carrying docs
#   2. node B joins with --resync: the new ring [A,B] re-assigns doc slots
#      that hash to B; B pulls them from A and A ack-deletes them
#   3. /__zap/peers repartitions A's static ring view so /doc/{id} routes right
#   4. every doc answers 200 from BOTH nodes; totals stay 3
#
# Run AFTER `cargo build --release`. Leaves nothing running (trap on exit).
#   ./scripts/handoff-probe.sh            # 48681/48682 default ports
#   PORT_A=48691 PORT_B=48692 ./scripts/handoff-probe.sh

set -euo pipefail
A=${PORT_A:-48681}
B=${PORT_B:-48682}
BIN=${ZAP_BIN:-target/release/zap}
LOG="/tmp/zap-handoff-$A.log"

cleanup() {
  pkill -f "zap serve --bind 127.0.0.1:$A" 2>/dev/null || true
  pkill -f "zap serve --bind 127.0.0.1:$B" 2>/dev/null || true
}
trap cleanup EXIT

# 1) singleton ingest
"$BIN" serve --bind "127.0.0.1:$A" --announce "127.0.0.1:$A" --peers "" --lanes 2 \
  >"$LOG" 2>&1 &
sleep 1.5
for i in 1 2 3; do
  curl -s -X POST "127.0.0.1:$A/ingest" -d "{\"id\":\"h$i\",\"text\":\"handoff doc $i fox\"}" >/dev/null
done

# 2) B joins with --resync; boot pull runs ~1.5s after serving
"$BIN" serve --bind "127.0.0.1:$B" --announce "127.0.0.1:$B" \
  --peers "127.0.0.1:$A" --lanes 2 --resync >>"$LOG" 2>&1 &
sleep 5
echo "== B log =="; tail -3 "$LOG"

# 3) repartition A's ring view (both nodes must agree on membership)
curl -s -X POST "127.0.0.1:$A/__zap/peers" -d '{"peers":"127.0.0.1:'"$B"'"}'; echo

# 4) verify
echo "== doc routing =="
for p in "$A" "$B"; do
  for i in 1 2 3; do
    printf '%s h%s: ' "$p" "$i"
    curl -s -m 3 -o /dev/null -w '%{http_code}\n' "127.0.0.1:$p/doc/h$i"
  done
done
echo "== stats (totals must be 3) =="
curl -s "127.0.0.1:$B/stats"
echo
