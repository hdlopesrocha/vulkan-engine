#!/usr/bin/env bash
# Phase 3 A/B capture for the octree screen-space hit tolerance.
#
# Runs the app once per epsilon value with the deterministic benchmark camera,
# lets it exit itself after a fixed number of post-load frames (the normal exit
# screenshot path writes bin/screenshot.png), then compares each image against
# the world-space-epsilon baseline with ImageMagick. Counter lines from the
# Phase 0 profiler are retained alongside the images.
#
# Usage:
#   scripts/compare_raymarch_eps.sh
#   FRAMES=300 CAMERA_PATH=pan5 scripts/compare_raymarch_eps.sh
#   EPS_LIST="0 0.25 0.5" VULKAN_CAMERA="..." scripts/compare_raymarch_eps.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$ROOT/bin"
OUT_DIR="${OUT_DIR:-$ROOT/logs/raymarch_eps}"
FRAMES="${FRAMES:-180}"
CAMERA_PATH="${CAMERA_PATH:-static}"
EPS_LIST="${EPS_LIST:-0 0.25 0.5}"

if [[ ! -x "$BIN_DIR/app" ]]; then
    echo "error: $BIN_DIR/app not found; build with 'make -j8 debug' first" >&2
    exit 1
fi
if ! command -v compare >/dev/null 2>&1; then
    echo "error: ImageMagick 'compare' is required" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
rm -f "$BIN_DIR/screenshot.png"

echo "capture: $FRAMES post-load frames, camera path '$CAMERA_PATH'"
echo "output:  $OUT_DIR"

for eps in $EPS_LIST; do
    log="$OUT_DIR/eps_${eps}.log"
    image="$OUT_DIR/screenshot_${eps}.png"
    rm -f "$BIN_DIR/screenshot.png" "$image"

    echo "-- epsilon $eps px"
    if ! (
        cd "$BIN_DIR"
        VULKAN_OCTREE_RAYMARCH=1 \
        VULKAN_OCTREE_PROFILE=1 \
        VULKAN_OCTREE_SCREEN_EPS="$eps" \
        VULKAN_CAMERA_PATH="$CAMERA_PATH" \
        VULKAN_BENCH_EXIT_AFTER="$FRAMES" \
        VULKAN_HIDE_UI=1 \
        ./app
    ) >"$log" 2>&1; then
        echo "error: app run for epsilon $eps failed; see $log" >&2
        tail -40 "$log" >&2 || true
        exit 1
    fi

    if [[ ! -f "$BIN_DIR/screenshot.png" ]]; then
        echo "error: no exit screenshot for epsilon $eps; see $log" >&2
        exit 1
    fi
    mv "$BIN_DIR/screenshot.png" "$image"

    counter_line="$(grep '\[octree\]' "$log" | tail -1 || true)"
    if [[ -n "$counter_line" ]]; then
        echo "$counter_line" > "$OUT_DIR/counters_${eps}.txt"
    fi
done

baseline="$OUT_DIR/screenshot_0.png"
if [[ ! -f "$baseline" ]]; then
    baseline="$(find "$OUT_DIR" -maxdepth 1 -name 'screenshot_*.png' | head -1)"
fi

printf '\n%-8s %-14s %-14s %-12s\n' "epsilon" "RMSE" "PSNR" "AE"
printf '%-8s %-14s %-14s %-12s\n' "-------" "----" "----" "--"
for eps in $EPS_LIST; do
    image="$OUT_DIR/screenshot_${eps}.png"
    [[ "$image" == "$baseline" ]] && continue
    rmse="$(compare -metric RMSE "$baseline" "$image" null: 2>&1 || true)"
    psnr="$(compare -metric PSNR "$baseline" "$image" null: 2>&1 || true)"
    ae="$(compare -metric AE "$baseline" "$image" null: 2>&1 || true)"
    printf '%-8s %-14s %-14s %-12s\n' "$eps" "$rmse" "$psnr" "$ae"
done

echo
echo "images and logs: $OUT_DIR"
echo "step-count lines: $OUT_DIR/counters_*.txt"
