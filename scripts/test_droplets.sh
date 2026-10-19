#!/bin/bash
# Headless droplet-physics test (no GPU required: runs the debug build on the
# lavapipe software rasterizer under Xvfb).
#
# Verifies the qualitative behaviour of the GPU droplet simulation from the
# app log:
#   1. the SDF region table is built from the octree (thousands of nodes),
#   2. the particles fall (mean Y decreases over time),
#   3. terrain SDF contacts appear once they reach the ground,
#   4. the particles slow down near the ground (settling / friction),
#   5. the automatic re-seed lifts them back to cloud level,
#   6. no validation errors and no hash-bucket overflow.
#
# Usage: scripts/test_droplets.sh [seconds]   (default 300)
set -u
SECONDS_TO_RUN="${1:-300}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d /tmp/droplet_test.XXXXXX)"
DISP=":77"

cleanup() {
    [ -n "${APP:-}" ] && kill -TERM "$APP" 2>/dev/null
    [ -n "${XVFB:-}" ] && kill -TERM "$XVFB" 2>/dev/null
}
trap cleanup EXIT

Xvfb "$DISP" -screen 0 1280x720x24 >"$OUT/xvfb.log" 2>&1 &
XVFB=$!
sleep 2

cd "$REPO/bin"
DISPLAY="$DISP" VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    VULKAN_DROPLETS=1 VULKAN_HIDE_UI=1 VULKAN_DROPLET_DT=0.25 \
    timeout -s KILL "$SECONDS_TO_RUN" ./app --local >"$OUT/app.log" 2>&1
LOG="$OUT/app.log"

pass=0; fail=0
check() { # check <description> <condition-result>
    if [ "$2" = "1" ]; then echo "  PASS  $1"; pass=$((pass+1));
    else echo "  FAIL  $1"; fail=$((fail+1)); fi
}

echo "droplet physics test (log: $LOG)"

nodes=$(grep -c "SDF region table: [0-9]* nodes" "$LOG")
ncount=$(grep -o "SDF region table: [0-9]* nodes" "$LOG" | tail -1 | grep -o "[0-9]*")
check "SDF region table built from the octree (>= 1000 nodes)" \
      "$([ "${ncount:-0}" -ge 1000 ] && echo 1 || echo 0)"

firstY=$(grep "\[droplets\]" "$LOG" | head -1 | sed -n 's/.*mean=\([-0-9.]*\).*/\1/p')
lastY=$(grep "\[droplets\]" "$LOG" | tail -1 | sed -n 's/.*mean=\([-0-9.]*\).*/\1/p')
maxY=$(grep "\[droplets\]" "$LOG" | tail -1 | sed -n 's/.*max=\([-0-9.]*\).*/\1/p')
check "particles fall from cloud level (first mean Y > 1500 m)" \
      "$(awk -v y="${firstY:-0}" 'BEGIN { print (y > 1500) ? 1 : 0 }')"
check "mean Y drops over the run" \
      "$(awk -v a="${firstY:-0}" -v b="${lastY:-0}" 'BEGIN { print (b < a) ? 1 : 0 }')"

terrain=$(grep "\[droplets\]" "$LOG" | sed -n 's/.*terrain+escaped=\([0-9]*\).*/\1/p' | sort -n | tail -1)
check "terrain SDF contacts occur (terrain+escaped > 0)" \
      "$([ "${terrain:-0}" -gt 0 ] && echo 1 || echo 0)"

lastV=$(grep "\[droplets\]" "$LOG" | sed -n 's/.*|v0|=\([0-9.]*\).*/\1/p' | tail -1)
check "particles settle after impact (final |v| < 100 m/s)" \
      "$(awk -v v="${lastV:-1e9}" 'BEGIN { print (v < 100) ? 1 : 0 }')"

# Re-seed: the mean altitude climbs back toward the spawn height after having
# dropped (a > 300 m jump between consecutive samples).
reset=$(grep -o "mean=[0-9.]*" "$LOG" | sed 's/mean=//' | \
        awk 'NR>1 && $1 > prev + 300 { found=1 } { prev=$1 } END { print found+0 }')
check "automatic re-seed lifts the particles back to cloud level" "$reset"

overflow=$(grep "\[droplets\]" "$LOG" | sed -n 's/.*gridOverflow=\([0-9]*\).*/\1/p' | sort -n | tail -1)
check "spatial hash has no bucket overflow" "$([ "${overflow:-1}" -eq 0 ] && echo 1 || echo 0)"

errors=$(grep -c "validation(" "$LOG")
check "no Vulkan validation errors" "$([ "$errors" -eq 0 ] && echo 1 || echo 0)"

echo "result: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
