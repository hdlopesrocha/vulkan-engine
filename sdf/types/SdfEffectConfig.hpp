#pragma once

// Whole-module tuning: one instance owned by SdfRenderer, shared with the
// ray marching widget. CPU-only aggregate (no direct GPU twin): fields are
// packed into the GPU mirrors Smoke/Bullet by the renderer setters,
// which keep clamping + dirty-flag routing.
//
// This is plain tuning data, NOT GPU state: the SSBO mirrors (SmokeFragBullet
// etc.) and scene rebuilds are derived from it by the renderer.

#include "SdfLavaConfig.hpp"
#include "SdfSmokeConfig.hpp"
#include "SdfBulletConfig.hpp"

struct SdfEffectConfig {
    SdfLavaConfig lava;
    SdfSmokeConfig smoke;
    SdfBulletConfig bullet;
};
