#ifndef CLOUDS_GLSL
#define CLOUDS_GLSL

// SDF-based procedural volumetric clouds: three altitude tiers sharing one
// noise field, rendered in the fullscreen sky passes.
//   Low  ~2 km      stratus / stratocumulus / cumulus (dense water droplets)
//   Mid  2..7 km    altocumulus / altostratus (water/ice mix)
//   High 5..13+ km  cirrus / cirrocumulus / cirrostratus (ice crystals, streaky)
//
// Architecture (generic SDF framework, specialized for sky volumes):
//   Cloud SDF (smooth-unioned ellipsoid puffs per tier, §4/§11)
//       -> Vertical Profile (slab height fraction, §10)
//       -> Coverage remap (existing threshold, §8)
//       -> Domain Warp (animated, §7/§13)
//       -> Base / Detail / Fine noise bands (§6)
//       -> Edge Erosion (§9)
//       -> Final Density (§17)
//       -> Lighting: HG phase + self-shadow + silver lining + ambient (§18-21)
// Volume ray marching is bounded by the slab + SDF empty-space skipping
// (§14-16) with adaptive SDF-guided steps and early termination.
//
// The SDF evaluation reuses the shared generic framework
// (sdf_primitives.glsl / sdf_ops.glsl / sdf_noise.glsl): puff masses are
// sphere SDFs combined with polynomial smooth unions, animated in the
// noise domain by wind. The SDF is conservative (noise/erosion only remove
// density near edges), so SDF-guided skipping never jumps over cloud.
//
// Parameter mapping (CloudSettings -> SDF system, §2/§31):
//   tier enabled/coverage/density/scale/windMul/baseHeight/thickness: same
//     meaning; scale sets the puff-cell wavelength, base/thickness the slab
//     domain, coverage the density remap threshold, density the extinction
//     multiplier, windMul the per-tier shear.
//   windResponse: cloud gain on the shared wind field (set=0 binding 27,
//     shaders/includes/vegetation/WindField.glsl). The ambient base flow
//     (Wind widget direction x strength) drives the bulk domain drift; the
//     gust + tornado deviation adds a bounded noise-domain shear sampled on
//     the real shared-wind clock (windTime). 0 = clouds ignore wind.
//   timeScale: scales cloudTime (bulk drift + warp evolution).
//   detailStrength: second-octave erosion mix + warp amplitude.
//   densityScale/shadowStrength/ambientBoost/silverLining/sunForwardG/
//   exposure/raymarchSteps/lightSteps: consumed identically to before
//   (lighting formulas and loop budgets are unchanged).
//
// Requires: ubo.glsl + sky_view.glsl included first (SkyUniform `sky`,
// UniformObject `ubo`, PCG helpers). Perlin noise, the shared wind field and
// the generic SDF framework are pulled in below.
//
// Entry points (signatures unchanged):
//   raymarchClouds(camPos, viewDir, sunDir, sunColor, skyCol, dayFactor, steps)
//     Full volumetric march through every enabled slab. Used by the on-screen
//     sky (world-space origin) and the offscreen equirect (fixed origin, so the
//     cache stays view-independent — see SkyRendererEquirect.frag).
//   cloudShadowAt(worldPos)
//     Cheap SDF-backed projection along the sun direction. Used by the
//     terrain/water shaders to darken direct light (clouds cast shadows).
//   cloudApproxForReflection(reflDir)
//     Single-sample coverage for the inline procedural-sky fallback so RT-off
//     reflections still show clouds (the RT pipeline miss samples the equirect
//     which already contains the raymarched clouds).

#include "../noise/Perlin.glsl"
#include "../vegetation/WindField.glsl"
#include "../sdf/SdfPrimitives.glsl"
#include "../sdf/SdfOps.glsl"
#include "../sdf/SdfNoise.glsl"

// Local wind-shear tuning (see cloudWindShear): converts the shared field's
// gust/tornado deviation from m/s into a bounded cloud-domain displacement.
// The seconds factor is the visual response gain (world metres = m/s * secs);
// the max soft-saturates the result in domain units so funnel cores cannot
// shred the noise domain.
const float CLOUD_WIND_SHEAR_SECONDS = 90.0;
const float CLOUD_WIND_SHEAR_MAX = 3.0;

// Shared-field ambient direction, normalized with the same degenerate
// fallback as windAmbient (the CPU packs ambientA.xy normalized already).
vec2 cloudWindDir() {
    vec2 d = windField.ambientA.xy;
    float l = length(d);
    return (l > 1e-4) ? d / l : vec2(1.0, 0.0);
}

// Bulk cloud-domain drift: the shared field's ambient base flow integrated
// over the scaled cloud clock. Spatially constant, so the conservative puff
// SDF and the volume density translate by the exact same offset (the
// empty-space skip contract is unchanged). Replaces the old private
// windSpeed/windAngleDeg sliders: direction and strength now mirror the
// vegetation Wind widget, windResponse is the cloud-side gain.
vec2 cloudWindVec(float speedMul) {
    float response = clamp(sky.windResponse, 0.0, 10.0);
    float strength = max(windField.ambientA.z, 0.0);
    return cloudWindDir() * (strength * response * speedMul * sky.cloudTime);
}

// Local shared-field shear for the noise domain only: gust + tornado
// deviation from the ambient base flow, sampled at the cloud point on the
// shared real-time wind clock. World m/s -> domain units through the tier
// scale, soft-saturated at CLOUD_WIND_SHEAR_MAX. Applied exactly where
// cloudWarpOffset is applied (coverage/detail noise, never the puff SDF), so
// the conservative skip bound still holds.
vec2 cloudWindShear(vec3 p, float tierScale, float speedMul) {
    float response = clamp(sky.windResponse, 0.0, 10.0);
    vec2 vDev = windSVF(p, sky.windTime).xz
              - cloudWindDir() * max(windField.ambientA.z, 0.0);
    vec2 shear = vDev * (response * speedMul * tierScale * CLOUD_WIND_SHEAR_SECONDS);
    float m = length(shear);
    return shear * (CLOUD_WIND_SHEAR_MAX / (CLOUD_WIND_SHEAR_MAX + m));
}

// Per-tier parameter fetch (keeps the march/shadow code tier-agnostic).
// t = (coverage, density, scale, windMul), geom = (baseHeight, thickness).
void cloudTierParams(int tier, out vec4 t, out vec2 geom, out bool enabled) {
    if (tier == 0) {
        t = vec4(sky.lowCoverage, sky.lowDensity, sky.lowScale, sky.lowWindSpeedMul);
        geom = vec2(sky.lowBaseHeight, sky.lowThickness);
        enabled = (sky.lowEnabled != 0u);
    } else if (tier == 1) {
        t = vec4(sky.midCoverage, sky.midDensity, sky.midScale, sky.midWindSpeedMul);
        geom = vec2(sky.midBaseHeight, sky.midThickness);
        enabled = (sky.midEnabled != 0u);
    } else {
        t = vec4(sky.highCoverage, sky.highDensity, sky.highScale, sky.highWindSpeedMul);
        geom = vec2(sky.highBaseHeight, sky.highThickness);
        enabled = (sky.highEnabled != 0u);
    }
}

// Scaled cloud domain for one tier: horizontal coords in feature units
// (features are O(1)), wind-shifted. High cirrus is stretched along the
// wind direction for streaks (same behavior as before).
vec2 cloudTierDomain(vec2 xz, int tier, vec4 t) {
    vec2 sp = xz * max(t.z, 1e-7) + cloudWindVec(t.w);
    if (tier == 2) {
        vec2 wdir = cloudWindDir();
        vec2 wperp = vec2(-wdir.y, wdir.x);
        sp = vec2(dot(sp, wdir) * 0.35, dot(sp, wperp));
    }
    return sp;
}

// Procedural puff SDF for one tier, in world units, conservative: layered
// noise and erosion only remove density near puff edges, so this distance
// lower-bounds the smooth cloud domain and is safe for empty-space skipping.
// The domain is a hash-tiled field of ellipsoid puffs (one candidate per
// feature cell, ~18% of cells empty for broken sky), smooth-unioned over
// the 3x3 neighborhood for coherent masses with rounded transitions.
// One puff layer at the given domain frequency. Cells hold at most one
// ellipsoid puff (bounding-sphere SDF: exact and conservative), empty with
// a coverage- and region-dependent probability so spacing varies organically
// between clumps and gaps. seedOff decorrelates layers.
float cloudPuffLayer(vec3 p, vec2 sp, vec2 wind, float scaleEff, int tier,
                     vec4 t, vec2 geom, float seedOff, float rMul, float region) {
    vec2 cellId = floor(sp);
    float thickness = max(geom.y, 1.0);
    float aspect = clamp(thickness * scaleEff, 0.05, 1.0);
    float cov = clamp(t.x, 0.0, 1.0);
    float emptyFrac = (1.0 - cov) * 0.55 - region * 0.4;
    float best = 1e5;
    for (int jx = -1; jx <= 1; ++jx) {
        for (int jy = -1; jy <= 1; ++jy) {
            vec2 cid = cellId + vec2(float(jx), float(jy));
            float sb = float(tier) * 17.0 + seedOff;
            float h0 = sdfHash(vec3(cid, sb + 1.0));
            if (h0 < emptyFrac) continue; // gap between masses
            float h1 = sdfHash(vec3(cid, sb + 2.0));
            float h2 = sdfHash(vec3(cid, sb + 3.0));
            float h3 = sdfHash(vec3(cid, sb + 4.0));
            vec2 centerXZ = (cid + 0.5 + (vec2(h1, h2) - 0.5) * 0.7 - wind) / scaleEff;
            float rxz = (0.30 + 0.25 * h3) * (0.55 + 0.9 * region) * rMul / scaleEff;
            float cry = geom.x + thickness * (0.35 + 0.3 * h1);
            float rry = rxz * aspect * (0.7 + 0.6 * h2);
            vec3 c = vec3(centerXZ.x, cry, centerXZ.y);
            float r = max(rxz, rry);
            float d = length(p - c) - r;
            best = opSmoothUnion(best, d, max(rxz * 0.15, 1.0));
        }
    }
    return best;
}

float cloudPuffSDF(vec3 p, int tier) {
    vec4 t;
    vec2 geom;
    bool enabled;
    cloudTierParams(tier, t, geom, enabled);
    if (!enabled) return 1e5;
    float scale = max(t.z, 1e-7);
    float thickness = max(geom.y, 1.0);
    // Vertical gate first: outside the slab there is no cloud by definition
    // (the march loop also slab-bounds the interval, this is belt & braces
    // for direct queries such as the light march).
    if (p.y <= geom.x || p.y >= geom.x + thickness) return 1e5;
    vec2 sp = cloudTierDomain(p.xz, tier, t);
    vec2 wind = cloudWindVec(t.w);
    // Large-scale regional mask: slow fbm breaks the lattice into organic
    // clumps and gaps (variable spacing instead of uniform cells).
    float region = sdfNoise(vec3(sp * 0.11, float(tier) * 7.3 + 3.0));
    float best = cloudPuffLayer(p, sp, wind, scale, tier, t, geom, 0.0, 1.0, region);
    // Second layer at an incommensurate frequency/size: the combined field
    // has no single repetition period, so tiling reads as natural variety
    // instead of a pattern. Smooth-union keeps it conservative.
    vec2 sp2 = sp * 2.7 + vec2(5.3, 1.7);
    vec2 wind2 = wind * 2.7 + vec2(5.3, 1.7);
    best = opSmoothUnion(best,
        cloudPuffLayer(p, sp2, wind2, scale * 2.7, tier, t, geom, 11.0, 0.55, region),
        20.0);
    return best;
}

// Animated domain warp (detailStrength-scaled). Beyond wind translation the
// warp center itself drifts slowly with time so shapes evolve rather than
// merely slide (§13). Returns the warp offset in scaled-domain units.
vec2 cloudWarpOffset(vec2 sp, int tier) {
    float tSeed = float(tier) * 7.3;
    float wamp = 0.10 + clamp(sky.detailStrength, 0.0, 1.0) * 0.45;
    float drift = sky.cloudTime * 0.02;
    float wx = sdfNoise(vec3(sp * 0.5 + vec2(drift, -drift * 0.6), tSeed));
    float wy = sdfNoise(vec3(sp * 0.5 + vec2(-drift * 0.7, drift), tSeed + 4.7));
    return (vec2(wx, wy) - 0.5) * 2.0 * wamp;
}

// Coverage shaping: field in [0,1] -> density in [0,1]. coverage 0 = clear.
// (Unchanged formula.)
float cloudShapeCoverage(float n, float coverage) {
    float lo = 1.0 - coverage * 0.92;
    return smoothstep(lo, lo + 0.35, n);
}

// Vertical density profile inside a slab: smooth bottom fade-in, denser base,
// soft top fade-out. h01 = (y - base) / thickness in [0,1]. (Unchanged.)
float cloudVerticalProfile(float h01, int tier) {
    float bottom = smoothstep(0.0, tier == 0 ? 0.25 : 0.35, h01);
    float top = 1.0 - smoothstep(0.55, 1.0, h01);
    return bottom * top;
}

// Full SDF-backed density at a world-space point (0 outside every enabled
// slab). Pipeline: SDF -> vertical profile -> coverage remap -> domain warp
// -> base/detail/fine noise -> edge erosion -> extinction scaling.
float cloudSlabDensity(vec3 pos, int tier, bool cheap) {
    vec4 t;
    vec2 geom;
    bool enabled;
    cloudTierParams(tier, t, geom, enabled);
    if (!enabled) return 0.0;
    float thickness = max(geom.y, 1.0);
    float h01 = (pos.y - geom.x) / thickness;
    if (h01 <= 0.0 || h01 >= 1.0) return 0.0;
    float sdfW = cloudPuffSDF(pos, tier);
    float scale = max(t.z, 1e-7);
    float tSeed = float(tier) * 7.3;
    vec2 sp = cloudTierDomain(pos.xz, tier, t);
    // Base shape noise (per-tier octave count, as before: 5/4/4).
    int octs = (tier == 0) ? 5 : 4;
    vec2 warped = sp;
    if (!cheap) {
        // Animated shape evolution + shared-field gust/tornado shear both
        // distort only the noise domain; the puff SDF keeps cloudWindVec's
        // pure bulk translation, preserving the conservative skip bound.
        warped = sp + cloudWarpOffset(sp, tier) + cloudWindShear(pos, scale, t.w);
    }
    float base = sdfFbmOct(vec3(warped, tSeed), cheap ? 3 : octs);
    // Puff interior lifts the field so masses survive the coverage cut.
    float interiorBand = (1.0 / scale) * 0.15;
    float interior = 1.0 - smoothstep(0.0, max(interiorBand, 1.0), sdfW);
    float field = base * 0.55 + interior * 0.45;
    float cov = cloudShapeCoverage(field, clamp(t.x, 0.0, 1.0));
    if (tier == 2) cov *= cov; // wispy cirrus (as before)
    float dens = cov;
    if (!cheap) {
        // Detail erosion breaks up the edges (second high-freq sample,
        // same mix shape as before).
        if (sky.detailStrength > 0.001) {
            float d = sdfFbmOct(vec3(warped * 3.7 + 13.1, tSeed + 5.0), 3);
            dens = mix(dens, dens * (0.55 + 0.9 * d), clamp(sky.detailStrength, 0.0, 1.0));
        }
        // Fine edge erosion: irregular broken boundaries and soft wisps.
        float fine = sdfNoise(vec3(warped * 7.3, tSeed + 9.0));
        float erW = thickness * 0.06;
        float edge = smoothstep(0.0, erW, -sdfW + (fine - 0.5) * 2.0 * erW);
        dens *= edge;
    }
    return dens * cloudVerticalProfile(h01, tier) * max(t.y, 0.0);
}

// Full density at a world-space point (0 outside every enabled slab).
float cloudDensityAt(vec3 pos) {
    if (sky.cloudsEnabled == 0u) return 0.0;
    float d = cloudSlabDensity(pos, 0, false)
            + cloudSlabDensity(pos, 1, false)
            + cloudSlabDensity(pos, 2, false);
    return d * max(sky.densityScale, 0.0);
}

// Cheap density (puff SDF + reduced base noise, no warp/detail/erosion) for
// the light march and other multi-sample queries (§19/§25: cheap first).
float cloudDensityCheapAt(vec3 pos) {
    if (sky.cloudsEnabled == 0u) return 0.0;
    float d = cloudSlabDensity(pos, 0, true)
            + cloudSlabDensity(pos, 1, true)
            + cloudSlabDensity(pos, 2, true);
    return d * max(sky.densityScale, 0.0);
}

// Henyey-Greenstein phase for the sun scattering lobe. (Unchanged.)
float cloudPhaseHG(float cosTheta, float g) {
    float g2 = g * g;
    float denom = pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5);
    return (1.0 - g2) / max(denom, 1e-4) * 0.25;
}

// Slab ray intersection: parametric [t0,t1] of ray (origin, dir) with the
// horizontal slab [base, base+thickness]. Returns false when missed/behind.
// (Unchanged: the slab is the coarse bound, the SDF refines inside.)
bool cloudSlabIntersect(vec3 origin, vec3 dir, float base, float thickness,
                        out float t0, out float t1) {
    t0 = 0.0; t1 = -1.0;
    if (abs(dir.y) < 1e-5) return false;
    float tA = (base - origin.y) / dir.y;
    float tB = (base + thickness - origin.y) / dir.y;
    t0 = min(tA, tB);
    t1 = max(tA, tB);
    if (t1 < 0.0) return false;
    t0 = max(t0, 0.0);
    // Far plane of the cloud system: nothing renderable beyond the high slab.
    t1 = min(t1, 30000.0);
    return t1 > t0;
}

// Volumetric march through all enabled slabs.
// Returns vec4(rgb, transmittance): rgb = scattered cloud light, a = fraction
// of the background that survives (1 = clear sky). dayFactor dims clouds at
// night; sunColor tints the scattering. Step budget and lighting model are
// unchanged; inside the interval the SDF skips empty space and adapts the
// step (small near boundaries/dense cores, large in voids).
vec4 raymarchClouds(vec3 camPos, vec3 viewDir, vec3 sunDir, vec3 sunColor,
                    vec3 skyBackground, float dayFactor, int maxSteps) {
    if (sky.cloudsEnabled == 0u) return vec4(vec3(0.0), 1.0);
    // No horizon early-out: visibility is decided purely by slab
    // intersection, so clouds render from every side — from below (looking
    // up), from above (looking down at cloud tops), from inside the slab,
    // and at grazing horizon angles. A downward ray from below the slabs
    // simply misses every slab and returns clear sky; terrain occludes it
    // on screen anyway (sky depth test) and the equirect lower hemisphere
    // then correctly shows ground haze instead of clouds.

    int steps = int(clamp(sky.raymarchSteps, 4.0, 24.0));
    if (maxSteps > 0) steps = min(steps, maxSteps);
    int lsteps = int(clamp(sky.lightSteps, 1.0, 6.0));

    float cosTheta = dot(viewDir, sunDir);
    float phase = cloudPhaseHG(cosTheta, clamp(sky.sunForwardG, -0.9, 0.9));
    // Silver lining: bright rim when looking near the sun through thin cloud.
    float silver = pow(max(cosTheta, 0.0), 6.0) * clamp(sky.silverLining, 0.0, 2.0);

    vec3 ambient = (skyBackground * 0.7 + vec3(0.35, 0.38, 0.42) * 0.3)
                 * (0.25 + 0.75 * dayFactor);
    ambient *= (0.6 + 0.4 * clamp(sky.ambientBoost * 2.0, 0.0, 2.0));

    vec3 accum = vec3(0.0);
    float transmittance = 1.0;

    // March each enabled slab front-to-back, sharing the step budget.
    for (int tier = 0; tier < 3; ++tier) {
        vec4 tt;
        vec2 geom;
        bool onTier;
        cloudTierParams(tier, tt, geom, onTier);
        if (!onTier) continue;
        float t0, t1;
        if (!cloudSlabIntersect(camPos, viewDir, geom.x, max(geom.y, 1.0), t0, t1))
            continue;
        // Fewer steps for thin/high slabs; low clouds get the full budget.
        int tierSteps = (tier == 2) ? max(steps / 2, 2) : steps;
        float dtBig = (t1 - t0) / float(tierSteps);
        float dtSmall = dtBig * 0.3;
        // Dither the start to hide banding between steps.
        float jitter = fract(sdfHash(vec3(gl_FragCoord.xy * 0.37, 1.7)));
        float t = t0 + dtBig * jitter;
        for (int i = 0; i < 24; ++i) {
            if (i >= tierSteps) break;
            if (transmittance < 0.02) break;
            if (t > t1) break;
            vec3 p = camPos + viewDir * t;
            // SDF empty-space skipping: outside the smooth domain, advance
            // by the conservative distance without any noise evaluation.
            // The SDF lower-bounds the domain (noise/erosion only remove),
            // so the jump can never cross cloud.
            float sdfW = cloudPuffSDF(p, tier);
            if (sdfW > 0.0) {
                t += min(max(sdfW * 0.9, dtSmall), t1 - t) + 1e-3;
                continue;
            }
            // Adaptive step: small near boundaries and dense cores (sdf is
            // only just negative there), large across deep voids is handled
            // by the skip above; inside, scale by the local field depth.
            float dt = clamp(-sdfW * 0.5 + dtSmall, dtSmall, dtBig);
            float dens = cloudSlabDensity(p, tier, false)
                       * max(sky.densityScale, 0.0);
            if (dens > 0.003) {
                // Light march toward the sun (cheap density: puff field +
                // reduced base noise, no warp/detail/erosion).
                float lightTrans = 1.0;
                float ldt = max(geom.y, 1.0) / float(lsteps);
                vec3 lp = p + sunDir * ldt * 0.5;
                for (int j = 0; j < 6; ++j) {
                    if (j >= lsteps) break;
                    lightTrans *= exp(-cloudDensityCheapAt(lp) * ldt * 0.012);
                    lp += sunDir * ldt;
                }
                float beers = exp(-dens * dt * 0.012);
                float powder = 1.0 - exp(-dens * dt * 0.05);
                vec3 s = (sunColor * (phase * 2.2 + silver * powder) * lightTrans
                        + ambient * (0.5 + 0.5 * powder)) * dayFactor
                        + ambient * 0.08;
                float a = (1.0 - beers) * transmittance;
                accum += s * a;
                transmittance *= beers;
            }
            t += dt;
        }
        if (transmittance < 0.02) break;
    }

    accum *= clamp(sky.exposure, 0.1, 3.0);
    return vec4(accum, transmittance);
}

// SDF-backed coverage field for the 2D projections (shadows, reflection
// approx): puff domain + base/detail noise at the slab mid height, remapped
// by the same coverage threshold. Same weights and tone mapping as before
// apply at the call sites below. The shared-wind bulk drift is included via
// cloudTierDomain; the animated warp and the local shared-field shear are
// intentionally omitted (cheap projection approximation, matches the
// pre-existing omission of cloudWarpOffset).
float cloudShadowField(vec2 xz, int tier) {
    vec4 t;
    vec2 geom;
    bool enabled;
    cloudTierParams(tier, t, geom, enabled);
    if (!enabled) return 0.0;
    vec3 probe = vec3(xz.x, geom.x + max(geom.y, 1.0) * 0.5, xz.y);
    float sdfW = cloudPuffSDF(probe, tier);
    vec2 sp = cloudTierDomain(xz, tier, t);
    float tSeed = float(tier) * 7.3;
    int octs = (tier == 0) ? 5 : 4;
    float base = sdfFbmOct(vec3(sp, tSeed), octs);
    float scale = max(t.z, 1e-7);
    float interior = 1.0 - smoothstep(0.0, max((1.0 / scale) * 0.15, 1.0), sdfW);
    float field = base * 0.55 + interior * 0.45;
    float cov = cloudShapeCoverage(field, clamp(t.x, 0.0, 1.0));
    if (tier == 2) cov *= cov;
    if (sky.detailStrength > 0.001) {
        float d = sdfFbmOct(vec3(sp * 3.7 + 13.1, tSeed + 5.0), 3);
        cov = mix(cov, cov * (0.55 + 0.9 * d), clamp(sky.detailStrength, 0.0, 1.0));
    }
    return cov;
}

// Cheap cloud shadow for lit surfaces: project the fragment along the sun
// direction onto each enabled slab and accumulate SDF-backed coverage.
// Returns 0 (lit) .. 1 (fully shadowed); the caller scales by
// sky.shadowStrength. (Structure, weights and mapping unchanged.)
float cloudShadowAt(vec3 worldPos) {
    if (sky.cloudsEnabled == 0u) return 0.0;
    if (sky.shadowStrength <= 0.001) return 0.0;
    vec3 sunDirTo = -normalize(ubo.lightDirection);
    if (sunDirTo.y < 0.03) return 0.0;
    float shadow = 0.0;
    if (sky.lowEnabled != 0u) {
        float h = sky.lowBaseHeight - worldPos.y;
        if (h > 0.0) {
            vec2 hitXZ = worldPos.xz + sunDirTo.xz / sunDirTo.y * h;
            shadow += cloudShadowField(hitXZ, 0) * max(sky.lowDensity, 0.0) * 0.55;
        }
    }
    if (sky.midEnabled != 0u) {
        float h = sky.midBaseHeight - worldPos.y;
        if (h > 0.0) {
            vec2 hitXZ = worldPos.xz + sunDirTo.xz / sunDirTo.y * h;
            shadow += cloudShadowField(hitXZ, 1) * max(sky.midDensity, 0.0) * 0.40;
        }
    }
    if (sky.highEnabled != 0u) {
        float h = sky.highBaseHeight - worldPos.y;
        if (h > 0.0) {
            vec2 hitXZ = worldPos.xz + sunDirTo.xz / sunDirTo.y * h;
            // Cirrus barely shadows (ice crystals, optically thin).
            shadow += cloudShadowField(hitXZ, 2) * max(sky.highDensity, 0.0) * 0.18;
        }
    }
    shadow *= max(sky.densityScale, 0.0) * clamp(sky.shadowStrength, 0.0, 1.0);
    return 1.0 - exp(-max(shadow, 0.0) * 1.6);
}

// Single-sample approximation for the inline procedural-sky fallback (RT off):
// returns the cloud albedo to add over the gradient sky for a reflect ray.
// (Planar factors, weights and colors unchanged.)
vec3 cloudApproxForReflection(vec3 reflDir, vec3 sunDir, float dayFactor) {
    if (sky.cloudsEnabled == 0u) return vec3(0.0);
    if (reflDir.y < 0.02 || dayFactor <= 0.01) return vec3(0.0);
    float cov = 0.0;
    float wsum = 0.0;
    if (sky.lowEnabled != 0u) { cov += cloudShadowField(reflDir.xz / max(reflDir.y, 0.05) * 2000.0, 0) * 0.5; wsum += 0.5; }
    if (sky.midEnabled != 0u) { cov += cloudShadowField(reflDir.xz / max(reflDir.y, 0.05) * 4500.0, 1) * 0.35; wsum += 0.35; }
    if (sky.highEnabled != 0u) { cov += cloudShadowField(reflDir.xz / max(reflDir.y, 0.05) * 9000.0, 2) * 0.25; wsum += 0.25; }
    if (wsum <= 0.0) return vec3(0.0);
    cov /= wsum;
    float sunAmt = max(dot(reflDir, sunDir), 0.0);
    vec3 cloudCol = mix(vec3(0.75, 0.78, 0.82), vec3(1.05, 1.0, 0.95),
                        pow(sunAmt, 3.0) * dayFactor);
    return cloudCol * cov * smoothstep(0.02, 0.25, reflDir.y) * dayFactor
         * clamp(sky.exposure, 0.1, 3.0);
}

// ---- Development debug views (§29) ----
// mode: 1 = SDF sign, 2 = base density, 3 = coverage, 4 = domain warp,
//       5 = erosion, 6 = final density, 7 = lighting transmittance,
//       8 = cloud shadows, 9 = ray-march steps, 10 = empty-space skipping.
// Evaluated at the first enabled tier intersected by the ray; out of range
// modes return black. Called from SkyRenderer.frag only (the equirect probe always
// renders the final image so reflections stay clean).
vec3 cloudDebugView(vec3 camPos, vec3 viewDir, vec3 sunDir, int mode) {
    if (sky.cloudsEnabled == 0u) return vec3(0.0);
    int tier = -1;
    float t0 = 0.0, t1 = -1.0;
    vec4 tt;
    vec2 geom;
    for (int k = 0; k < 3; ++k) {
        vec4 ltt;
        vec2 lgeom;
        bool onTier;
        cloudTierParams(k, ltt, lgeom, onTier);
        if (!onTier) continue;
        float a0, a1;
        if (cloudSlabIntersect(camPos, viewDir, lgeom.x, max(lgeom.y, 1.0), a0, a1)) {
            tier = k;
            t0 = a0;
            t1 = a1;
            tt = ltt;
            geom = lgeom;
            break;
        }
    }
    if (tier < 0) return vec3(0.0);
    float thickness = max(geom.y, 1.0);
    vec3 mid = camPos + viewDir * (0.5 * (t0 + t1));
    // Clamp the probe inside the slab (grazing rays can place the midpoint
    // outside after clamping).
    mid.y = clamp(mid.y, geom.x + thickness * 0.02, geom.x + thickness * 0.98);
    float h01 = (mid.y - geom.x) / thickness;
    float sdfW = cloudPuffSDF(mid, tier);
    if (mode == 1) {
        // SDF sign: blue = inside domain, white = boundary, red = outside.
        float band = thickness * 0.05;
        if (abs(sdfW) < band) return vec3(1.0);
        if (sdfW < 0.0) return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 0.15), clamp(-sdfW / (band * 8.0), 0.0, 1.0));
        return mix(vec3(1.0, 0.0, 0.0), vec3(0.15, 0.0, 0.0), clamp(sdfW / (band * 8.0), 0.0, 1.0));
    }
    vec2 sp = cloudTierDomain(mid.xz, tier, tt);
    float tSeed = float(tier) * 7.3;
    int octs = (tier == 0) ? 5 : 4;
    float base = sdfFbmOct(vec3(sp, tSeed), octs);
    if (mode == 2) return vec3(base);
    float scale = max(tt.z, 1e-7);
    float interior = 1.0 - smoothstep(0.0, max((1.0 / scale) * 0.15, 1.0), sdfW);
    float cov = cloudShapeCoverage(base * 0.55 + interior * 0.45, clamp(tt.x, 0.0, 1.0));
    if (tier == 2) cov *= cov;
    if (mode == 3) return vec3(cov);
    vec2 warp = cloudWarpOffset(sp, tier);
    if (mode == 4) return vec3(clamp(length(warp) * 2.0, 0.0, 1.0));
    float dens = cov * cloudVerticalProfile(h01, tier) * max(tt.y, 0.0);
    if (sky.detailStrength > 0.001) {
        float d = sdfFbmOct(vec3(sp * 3.7 + 13.1, tSeed + 5.0), 3);
        dens = mix(dens, dens * (0.55 + 0.9 * d), clamp(sky.detailStrength, 0.0, 1.0));
    }
    float fine = sdfNoise(vec3(sp * 7.3, tSeed + 9.0));
    float erW = thickness * 0.06;
    float edge = smoothstep(0.0, erW, -sdfW + (fine - 0.5) * 2.0 * erW);
    if (mode == 5) return vec3(edge);
    dens *= edge;
    if (mode == 6) return vec3(clamp(dens * max(sky.densityScale, 0.0), 0.0, 1.0));
    if (mode == 7) {
        // Lighting transmittance toward the sun from the probe point.
        float lightTrans = 1.0;
        int lsteps = int(clamp(sky.lightSteps, 1.0, 6.0));
        float ldt = thickness / float(lsteps);
        vec3 lp = mid + sunDir * ldt * 0.5;
        for (int j = 0; j < 6; ++j) {
            if (j >= lsteps) break;
            lightTrans *= exp(-cloudDensityCheapAt(lp) * ldt * 0.012);
            lp += sunDir * ldt;
        }
        return vec3(lightTrans);
    }
    if (mode == 8) return vec3(cloudShadowAt(mid));
    // Modes 9/10: bounded mini-march over the first tier, counting density
    // evaluations (heat) vs distance advanced by SDF skips (green).
    {
        int tierSteps = (tier == 2) ? 8 : 16;
        float dtBig = (t1 - t0) / float(tierSteps);
        float dtSmall = dtBig * 0.3;
        float t = t0;
        int evals = 0;
        float skipped = 0.0;
        float span = max(t1 - t0, 1e-3);
        for (int i = 0; i < 24; ++i) {
            if (i >= tierSteps || t > t1) break;
            vec3 p = camPos + viewDir * t;
            float sd = cloudPuffSDF(p, tier);
            if (sd > 0.0) {
                float jump = min(max(sd * 0.9, dtSmall), t1 - t);
                skipped += jump;
                t += jump + 1e-3;
                continue;
            }
            evals++;
            t += dtSmall;
        }
        if (mode == 9) {
            float h = clamp(float(evals) / float(tierSteps), 0.0, 1.0);
            return mix(vec3(0.0), vec3(1.0, 0.25, 0.0), h);
        }
        float f = clamp(skipped / span, 0.0, 1.0);
        return vec3(0.0, f, f * 0.6);
    }
}

#endif // CLOUDS_GLSL
