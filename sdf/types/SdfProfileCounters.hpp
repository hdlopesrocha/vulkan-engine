#pragma once

// GPU-side SDF march counters (set=1 binding 10 storage buffer, fragment
// atomics). The layout is shared with shaders/renderer/sdf/SdfRenderer.frag
// (single source; the surface pipeline variant SdfRendererSurface.frag.spv is
// compiled from the same file with -DSDF_VARIANT=1 and declares the same
// six-uint array, sdfCounters[0..5]).
//
// The shader only atomically increments when `enabled` (written by the CPU
// each frame from SdfRenderer::setProfileCounters), so the default path costs
// one cached load plus a uniform branch. Requires fragmentStoresAndAtomics;
// when the device does not advertise it the counters stay disabled and the UI
// toggle is hidden.
// L14 TODO (needs an app run to validate): move the gate to a push constant
// or a stripped specialization variant; the six-uint layout above stays.
#include <cstdint>

struct SdfProfileCounters {
    uint32_t rays = 0;       // fragment shader invocations (proxy raster pixels)
    uint32_t steps = 0;      // march loop iterations
    uint32_t cellVisits = 0; // iterations over a non-empty DDA cell
    uint32_t candidates = 0; // sdfEvalInstance evaluations
    uint32_t hits = 0;       // shaded surface hits
    uint32_t enabled = 0;    // CPU-written gate: 1 = shader counters on
};
static_assert(sizeof(SdfProfileCounters) == 24, "SdfProfileCounters is 6 x uint32");
