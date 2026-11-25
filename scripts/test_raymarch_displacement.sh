#!/usr/bin/env bash
# Headless acceptance / validation sweep for true raymarched displacement.
#
# Each case runs the regular debug build + app through `make run-debug`, which
# tees all output to logs/run.log. VULKAN_BENCH_EXIT_AFTER makes the app exit
# cleanly after N ready frames, so no timeout / kill is used.
#
# The deterministic VULKAN_DISPLACEMENT_TEST scene (a 512 m slab + a sphere)
# keeps the sweep quick; the full procedural map is covered by the
# `large_scene` case, which deliberately uses a small amplitude.
set -euo pipefail
cd "$(dirname "$0")/.."

FRAMES="${VULKAN_TEST_FRAMES:-24}"
mkdir -p logs

run_case() {
    local name="$1"; shift
    echo
    echo "=== displacement test: ${name} ==="
    env VULKAN_BENCH_EXIT_AFTER="${FRAMES}" VULKAN_CAMERA_PATH=static "$@" make run-debug
    cp logs/run.log "logs/run-displacement-${name}.log"
    if grep -nE 'Validation Error|VUID-|ERROR: ' "logs/run-displacement-${name}.log" >/dev/null; then
        echo "FAIL: validation error in ${name} (logs/run-displacement-${name}.log)" >&2
        grep -nE 'Validation Error|VUID-|ERROR: ' "logs/run-displacement-${name}.log" | head -20 >&2
        exit 1
    fi
    if grep -nE 'DEVICE_LOST|device lost|context is lost' "logs/run-displacement-${name}.log" >/dev/null; then
        echo "FAIL: device loss in ${name} — lower the amplitude/frequency or the frame count" >&2
        exit 1
    fi
    echo "ok: ${name}"
}

# 1. Baseline: the new path must be a no-op when disabled.
run_case baseline VULKAN_DISPLACEMENT=0 VULKAN_DISPLACEMENT_TEST=1

# 2. Positive displacement on the deterministic slab + sphere.
run_case positive VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=1 \
    VULKAN_DISPLACEMENT_AMP=1 VULKAN_DISPLACEMENT_FREQ=1 \
    VULKAN_DISPLACEMENT_STRENGTH=1 VULKAN_DISPLACEMENT_OFFSET=0

# 3. Negative displacement (peaks become pits), same magnitude.
run_case negative VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=1 \
    VULKAN_DISPLACEMENT_AMP=-1 VULKAN_DISPLACEMENT_FREQ=1

# 4. High-frequency displacement: exercises the in-band conservative step.
run_case highfreq VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=1 \
    VULKAN_DISPLACEMENT_AMP=0.4 VULKAN_DISPLACEMENT_FREQ=16

# 5. Offset / strength / bounds.
run_case offset_bounds VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=1 \
    VULKAN_DISPLACEMENT_AMP=2 VULKAN_DISPLACEMENT_FREQ=2 \
    VULKAN_DISPLACEMENT_STRENGTH=0.7 VULKAN_DISPLACEMENT_OFFSET=0.25

# 5b. Procedural Perlin displacement source (no height texture).
run_case procedural VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=1 \
    VULKAN_DISPLACEMENT_AMP=1 VULKAN_DISPLACEMENT_FREQ=2 \
    VULKAN_DISPLACEMENT_TEXTURE=-2

# 6. Same displacement with the CSM terrain caster off: exercises the
#    "displaced geometry self-shadows but does not cast into CSM" path.
run_case caster_off VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=1 \
    VULKAN_DISPLACEMENT_CASTER=0 VULKAN_DISPLACEMENT_AMP=1 VULKAN_DISPLACEMENT_FREQ=2

# 7. Displacement AO off: verifies the AO term is optional and does not leak.
run_case ao_off VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=1 \
    VULKAN_DISPLACEMENT_AO=0 VULKAN_DISPLACEMENT_AMP=1 VULKAN_DISPLACEMENT_FREQ=2

# 7b. Displaced-normal debug view (54): exercises the debug branch and the
# true-gradient normal path.
run_case debug_normals VULKAN_DISPLACEMENT_TEST=1 VULKAN_DISPLACEMENT=1 \
    VULKAN_DISPLACEMENT_AMP=1 VULKAN_DISPLACEMENT_FREQ=2 \
    VULKAN_DEBUG_MODE=54 VULKAN_HIDE_UI=1

# 8. Full procedural / catalogue scene (complex octree geometry) with a small
#    amplitude. This is the heavy case; 8 frames is enough for a smoke test.
FRAMES=8 run_case large_scene VULKAN_DISPLACEMENT=1 \
    VULKAN_DISPLACEMENT_AMP=0.5 VULKAN_DISPLACEMENT_FREQ=1 \
    VULKAN_DISPLACEMENT_CASTER=0

# The enable log is emitted by setupTexturesFinish for every non-baseline run.
for n in positive negative highfreq offset_bounds procedural caster_off ao_off \
         debug_normals large_scene; do
    if ! grep -q '\[MyApp\] true displacement:' "logs/run-displacement-${n}.log"; then
        echo "FAIL: displacement material setup missing in ${n}" >&2
        exit 1
    fi
done

echo
echo "All displacement cases passed (validation-clean)."
echo "Logs: logs/run-displacement-*.log"
