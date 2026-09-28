#!/usr/bin/env bash
# scripts/torture.sh -- chaos loop: run the torture scenario binaries N times
# each with varied shake (kill timing, ring width, corrupt-state variants).
# Sweeps leftover zap processes before/after. Logs land in /tmp/zap-torture.
#
#   scripts/torture.sh                 # one pass over every scenario
#   REPEATS=5 scripts/torture.sh       # five passes of each scenario
#   ONLY=torture_churn scripts/torture.sh

set -uo pipefail
REPEATS=${REPEATS:-1}
ONLY=${ONLY:-}
SCENES=${SCENES:-torture_kill torture_churn torture_state torture_repartition}

cargo build --release --bin zap >/dev/null 2>&1 || {
  echo "release build failed" >&2
  exit 1
}

cleanup() { pkill -x zap 2>/dev/null; }
trap cleanup EXIT

rc=0
for s in $SCENES; do
  if [ -n "$ONLY" ] && [ "$s" != "$ONLY" ]; then
    continue
  fi
  for r in $(seq 1 "$REPEATS"); do
    echo "== $s round $r"
    if ! cargo test --release --test "$s" -- --nocapture; then
      echo "!! $s round $r FAILED - logs: /tmp/zap-torture/"
      rc=1
    fi
    pkill -x zap 2>/dev/null
    sleep 0.3
  done
done
exit $rc
