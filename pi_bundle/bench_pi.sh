#!/usr/bin/env bash
# Full on-device sweep: every models/*.pte x 4/3/2/1 threads, with a temperature/clock/throttle log.
#   bash bench_pi.sh [images_dir] [pack_prefix]      (packs are packs/<prefix>_<resolution>.npz)
set -uo pipefail
cd "$(dirname "$0")"
IMAGES=${1:-images}; PREFIX=${2:-trashnet}; OUT=results; mkdir -p "$OUT"

( while :; do echo "$(date +%s),$(vcgencmd measure_temp),$(vcgencmd measure_clock arm),$(vcgencmd get_throttled)"; sleep 1; done > "$OUT/thermal.csv" ) &
LOGGER=$!; trap 'kill $LOGGER 2>/dev/null' EXIT

for PTE in models/*.pte; do
  RES=$(basename "$PTE" | grep -oE '_[0-9]{3}_' | tr -d '_')
  STEM=$(basename "$PTE" .pte); STEM=${STEM%_fp32}; STEM=${STEM%_int8}; STEM=${STEM%_portable}
  PACK="packs/${STEM}.npz"; [ -f "$PACK" ] || PACK="packs/${PREFIX}_${RES}.npz"
  [ -f "$PACK" ] || { echo "skip $PTE (no $PACK)"; continue; }
  for T in 4 3 2 1; do
    TAG="$(basename "$PTE" .pte)_t$T"
    echo "=== $TAG ==="
    python run_eval.py --pte "$PTE" --pack "$PACK" --images "$IMAGES" --threads "$T" --repeat 3 --out "$OUT" --tag "$TAG" | tail -16
  done
done
python summarize.py "$OUT"
