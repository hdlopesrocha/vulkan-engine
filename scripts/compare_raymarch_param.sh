#!/usr/bin/env bash
# Phase 4 A/B capture for the parametric child-exit path.
#
# Runs the same deterministic camera twice (parametric off/on), captures the
# normal exit screenshot and the Phase 0 counter line, and compares the images.
# The path is designed to be bit-exact, so AE should be 0; any difference is a
# regression in the child-exit selection.
#
# Usage:
#   scripts/compare_raymarch_param.sh
#   FRAMES=300 CAMERA_PATH=pan5 scripts/compare_raymarch_param.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$ROOT/bin"
OUT_DIR="${OUT_DIR:-$ROOT/logs/raymarch_param}"
FRAMES="${FRAMES:-180}"
CAMERA_PATH="${CAMERA_PATH:-static}"

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

for mode in 0 1; do
    name="slab"
    [[ "$mode" == "1" ]] && name="parametric"
    log="$OUT_DIR/$name.log"
    image="$OUT_DIR/screenshot_$name.png"
    rm -f "$BIN_DIR/screenshot.png" "$image"

    echo "-- $name (VULKAN_OCTREE_PARAMETRIC_CHILD=$mode)"
    if ! (
        cd "$BIN_DIR"
        VULKAN_OCTREE_RAYMARCH=1 \
        VULKAN_OCTREE_PROFILE=1 \
        VULKAN_OCTREE_PARAMETRIC_CHILD="$mode" \
        VULKAN_CAMERA_PATH="$CAMERA_PATH" \
        VULKAN_BENCH_EXIT_AFTER="$FRAMES" \
        VULKAN_HIDE_UI=1 \
        ./app
    ) >"$log" 2>&1; then
        echo "error: app run for $name failed; see $log" >&2
        tail -40 "$log" >&2 || true
        exit 1
    fi
    if [[ ! -f "$BIN_DIR/screenshot.png" ]]; then
        echo "error: no exit screenshot for $name; see $log" >&2
        exit 1
    fi
    mv "$BIN_DIR/screenshot.png" "$image"
    grep '\[octree\]' "$log" | tail -1 > "$OUT_DIR/counters_$name.txt" || true
done

rmse="$(compare -metric RMSE "$OUT_DIR/screenshot_slab.png" \
    "$OUT_DIR/screenshot_parametric.png" null: 2>&1 || true)"
psnr="$(compare -metric PSNR "$OUT_DIR/screenshot_slab.png" \
    "$OUT_DIR/screenshot_parametric.png" null: 2>&1 || true)"
ae="$(compare -metric AE "$OUT_DIR/screenshot_slab.png" \
    "$OUT_DIR/screenshot_parametric.png" null: 2>&1 || true)"

echo
echo "image comparison (slab vs parametric)"
printf '  RMSE: %s\n  PSNR: %s\n  AE:   %s\n' "$rmse" "$psnr" "$ae"
echo
echo "counters:"
paste "$OUT_DIR/counters_slab.txt" "$OUT_DIR/counters_parametric.txt" 2>/dev/null || true
echo
echo "images and logs: $OUT_DIR"
