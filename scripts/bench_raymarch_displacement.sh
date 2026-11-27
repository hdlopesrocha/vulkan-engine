#!/usr/bin/env bash
# Deterministic A/B timing for the true raymarched displacement path.
#
# The app prints the GPU pass profile periodically as:
#   [GPU] ... octree=<ms> ... fps=<fps>
# This script runs the same ready-frame window for each configuration and
# reports the last sampled octree timings. It is a smoke benchmark, not a
# replacement for a full profiling capture.
#
# The deterministic displacement test scene keeps the scene build negligible
# so the measured window reflects rendering, not procedural generation.
set -euo pipefail
cd "$(dirname "$0")/.."

FRAMES="${VULKAN_BENCH_FRAMES:-120}"
mkdir -p logs

run_case() {
    local name="$1"; shift
    echo "=== bench: ${name} ==="
    env VULKAN_BENCH_EXIT_AFTER="${FRAMES}" VULKAN_CAMERA_PATH=static "$@" make run-debug
    cp logs/run.log "logs/bench-displacement-${name}.log"
    local line
    line=$(grep 'octree=' "logs/bench-displacement-${name}.log" | tail -1 || true)
    if [ -n "$line" ]; then
        echo "  $line"
    fi
    # The app only prints [frame] when a frame exceeds 100 ms, so the baseline
    # may be silent. The summary states the measured window explicitly.
    awk '/^\[frame\] [0-9.]+ ms/ { if (seen++) { s+=$2; n++; if (n==1 || $2<mn) mn=$2; if ($2>mx) mx=$2 } } END { if(n) printf "  frame avg=%.1f ms (min %.1f, max %.1f, n=%d)\n", s/n, mn, mx, n; else print "  frame <100 ms (below the app print threshold)" }' \
        "logs/bench-displacement-${name}.log"
}

run_case off VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=0
run_case on VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=1 \
    VULKAN_DISPLACEMENT_AMP=1 VULKAN_DISPLACEMENT_FREQ=1
run_case highfreq VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=1 \
    VULKAN_DISPLACEMENT_AMP=0.4 VULKAN_DISPLACEMENT_FREQ=16
run_case hard_shadow VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=1 \
    VULKAN_SHADOW_SOFTNESS=0 VULKAN_DISPLACEMENT_AMP=1 VULKAN_DISPLACEMENT_FREQ=1
run_case ao_off VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=1 \
    VULKAN_DISPLACEMENT_AO=0 VULKAN_DISPLACEMENT_AMP=1 VULKAN_DISPLACEMENT_FREQ=1

echo
echo "Timings above are the last periodic [GPU] lines (octree=...)."
echo "Logs: logs/bench-displacement-*.log"
