#pragma once

// Lava-anchored flame tuning (volumetric fire from brush-4 chunks).
// CPU-only (no direct GPU twin): density/scale shape ingested anchors, the
// flame-shape fields pack into the flame Definition and the Fire shape.
struct SdfLavaConfig {
    float density = 1.0f;      // flames per m^2 of lava surface at ingest
    float scale = 32.0f;       // anchor scale multiplier
    float spikiness = 0.35f;   // spike amplitude (0 = smooth rounded capsule)
    float tipRadius = 0.01f;
    float baseRadius = 1.0f;
    float height = 1.0f;
    float spikeFreq = 2.0f;
    float flameDensity = 0.025f;// volumetric density (lower = glassier)
};
