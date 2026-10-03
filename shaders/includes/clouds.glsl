#ifndef CLOUDS_GLSL
#define CLOUDS_GLSL

// Realistic volumetric clouds: three altitude tiers sharing one noise field.
//   Low  ~2 km      stratus / stratocumulus / cumulus (dense water droplets)
//   Mid  2..7 km    altocumulus / altostratus (water/ice mix)
//   High 5..13+ km  cirrus / cirrocumulus / cirrostratus (ice crystals, streaky)
// Requires: ubo.glsl + sky_view.glsl + perlin.glsl included first (pcgHash,
// uintToUnitFloat, SkyParamsNamed `sky`, UniformObjectNamed `ubo`).
//
// Two entry points:
//   raymarchClouds(camPos, viewDir, sunDir, sunColor, skyCol, dayFactor, steps)
//     Full volumetric march through every enabled slab. Used by the on-screen
//     sky (world-space origin) and the offscreen equirect (fixed origin, so the
//     cache stays view-independent — see sky_equirect.frag).
//   cloudShadowAt(worldPos)
//     Cheap 2D coverage lookup projected along the sun direction. Used by the
//     terrain/water shaders to darken direct light (clouds cast shadows).
//   cloudApproxForReflection(reflDir)
//     Single-sample coverage for the inline procedural-sky fallback so RT-off
//     reflections still show clouds (the RT pipeline miss samples the equirect
//     which already contains the raymarched clouds).

vec2 cloudWindVec(float speedMul) {
    float a = sky.windAngleRad;
    vec2 dir = vec2(cos(a), sin(a));
    return dir * (sky.windSpeed * speedMul * sky.cloudTime);
}

// Cheap 2D value noise (single octave) built on the shared PCG hash.
float cloudValueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    uvec2 i00 = floatBitsToUint(i);
    uvec2 i10 = floatBitsToUint(i + vec2(1.0, 0.0));
    uvec2 i01 = floatBitsToUint(i + vec2(0.0, 1.0));
    uvec2 i11 = floatBitsToUint(i + vec2(1.0, 1.0));
    float h00 = uintToUnitFloat(pcgHash(pcgHash(i00.x) ^ i00.y));
    float h10 = uintToUnitFloat(pcgHash(pcgHash(i10.x) ^ i10.y));
    float h01 = uintToUnitFloat(pcgHash(pcgHash(i01.x) ^ i01.y));
    float h11 = uintToUnitFloat(pcgHash(pcgHash(i11.x) ^ i11.y));
    return mix(mix(h00, h10, u.x), mix(h01, h11, u.x), u.y);
}

float cloudFbm2(vec2 p, int octaves) {
    float total = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    float norm = 0.0;
    for (int i = 0; i < 6; ++i) {
        if (i >= octaves) break;
        total += cloudValueNoise(p * freq) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.03;
    }
    return total / max(norm, 1e-5);
}

// Coverage shaping: fbm in [0,1] -> density in [0,1]. coverage 0 = clear.
float cloudShapeCoverage(float n, float coverage) {
    float lo = 1.0 - coverage * 0.92;
    return smoothstep(lo, lo + 0.35, n);
}

// Per-tier 2D coverage (single fbm, no vertical profile). tier: 0=low,1=mid,2=high.
float cloudTierCoverage(vec2 xz, int tier) {
    vec4 t;
    vec2 geom;
    if (tier == 0) { t = sky.lowTier; geom = sky.lowGeom; }
    else if (tier == 1) { t = sky.midTier; geom = sky.midGeom; }
    else { t = sky.highTier; geom = sky.highGeom; }
    float scale = max(t.z, 1e-7);
    float windMul = t.w;
    vec2 p = xz * scale + cloudWindVec(windMul);
    // High cirrus: stretch along the wind direction for streaks.
    if (tier == 2) {
        vec2 wdir = vec2(cos(sky.windAngleRad), sin(sky.windAngleRad));
        vec2 wperp = vec2(-wdir.y, wdir.x);
        vec2 q = vec2(dot(p, wdir) * 0.35, dot(p, wperp));
        float n = cloudFbm2(q, 4);
        // Wispy: push the threshold up and soften the top.
        float c = cloudShapeCoverage(n, clamp(t.x, 0.0, 1.0));
        return c * c;
    }
    int octs = (tier == 0) ? 5 : 4;
    float n = cloudFbm2(p, octs);
    // Detail erosion breaks up the edges (second high-freq sample).
    if (sky.detailStrength > 0.001) {
        float d = cloudFbm2(p * 3.7 + 13.1, 3);
        n = mix(n, n * (0.55 + 0.9 * d), clamp(sky.detailStrength, 0.0, 1.0));
    }
    return cloudShapeCoverage(n, clamp(t.x, 0.0, 1.0));
}

// Vertical density profile inside a slab: smooth bottom fade-in, denser base,
// soft top fade-out. h01 = (y - base) / thickness in [0,1].
float cloudVerticalProfile(float h01, int tier) {
    float bottom = smoothstep(0.0, tier == 0 ? 0.25 : 0.35, h01);
    float top = 1.0 - smoothstep(0.55, 1.0, h01);
    return bottom * top;
}

// Full density at a world-space point (0 outside every enabled slab).
float cloudDensityAt(vec3 pos) {
    if (!sky.cloudsEnabled) return 0.0;
    float d = 0.0;
    if (sky.lowEnabled) {
        float h01 = (pos.y - sky.lowGeom.x) / max(sky.lowGeom.y, 1.0);
        if (h01 > 0.0 && h01 < 1.0) {
            float cov = cloudTierCoverage(pos.xz, 0);
            d += cov * cloudVerticalProfile(h01, 0) * max(sky.lowTier.y, 0.0);
        }
    }
    if (sky.midEnabled) {
        float h01 = (pos.y - sky.midGeom.x) / max(sky.midGeom.y, 1.0);
        if (h01 > 0.0 && h01 < 1.0) {
            float cov = cloudTierCoverage(pos.xz, 1);
            d += cov * cloudVerticalProfile(h01, 1) * max(sky.midTier.y, 0.0);
        }
    }
    if (sky.highEnabled) {
        float h01 = (pos.y - sky.highGeom.x) / max(sky.highGeom.y, 1.0);
        if (h01 > 0.0 && h01 < 1.0) {
            float cov = cloudTierCoverage(pos.xz, 2);
            d += cov * cloudVerticalProfile(h01, 2) * max(sky.highTier.y, 0.0);
        }
    }
    return d * max(sky.densityScale, 0.0);
}

// Henyey-Greenstein phase for the sun scattering lobe.
float cloudPhaseHG(float cosTheta, float g) {
    float g2 = g * g;
    float denom = pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5);
    return (1.0 - g2) / max(denom, 1e-4) * 0.25;
}

// Slab ray intersection: parametric [t0,t1] of ray (origin, dir) with the
// horizontal slab [base, base+thickness]. Returns false when missed/behind.
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
// night; sunColor tints the scattering.
vec4 raymarchClouds(vec3 camPos, vec3 viewDir, vec3 sunDir, vec3 sunColor,
                    vec3 skyBackground, float dayFactor, int maxSteps) {
    if (!sky.cloudsEnabled) return vec4(vec3(0.0), 1.0);
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
        bool onTier = (tier == 0) ? sky.lowEnabled
                    : (tier == 1) ? sky.midEnabled : sky.highEnabled;
        if (!onTier) continue;
        vec2 geom = (tier == 0) ? sky.lowGeom
                  : (tier == 1) ? sky.midGeom : sky.highGeom;
        vec4 tt = (tier == 0) ? sky.lowTier
                : (tier == 1) ? sky.midTier : sky.highTier;
        float t0, t1;
        if (!cloudSlabIntersect(camPos, viewDir, geom.x, max(geom.y, 1.0), t0, t1))
            continue;
        // Fewer steps for thin/high slabs; low clouds get the full budget.
        int tierSteps = (tier == 2) ? max(steps / 2, 2) : steps;
        float dt = (t1 - t0) / float(tierSteps);
        // Dither the start to hide banding between steps.
        float jitter = cloudValueNoise(gl_FragCoord.xy * 0.37);
        float t = t0 + dt * jitter;
        for (int i = 0; i < 24; ++i) {
            if (i >= tierSteps) break;
            if (transmittance < 0.02) break;
            vec3 p = camPos + viewDir * t;
            float h01 = (p.y - geom.x) / max(geom.y, 1.0);
            float dens = 0.0;
            if (h01 > 0.0 && h01 < 1.0) {
                float cov = cloudTierCoverage(p.xz, tier);
                dens = cov * cloudVerticalProfile(h01, tier)
                     * max(tt.y, 0.0) * max(sky.densityScale, 0.0);
            }
            if (dens > 0.003) {
                // Light march toward the sun (cheap: reuse density samples).
                float lightTrans = 1.0;
                float ldt = max(geom.y, 1.0) / float(lsteps);
                vec3 lp = p + sunDir * ldt * 0.5;
                for (int j = 0; j < 6; ++j) {
                    if (j >= lsteps) break;
                    lightTrans *= exp(-cloudDensityAt(lp) * ldt * 0.012);
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

// Cheap cloud shadow for lit surfaces: project the fragment along the sun
// direction onto each enabled slab and accumulate coverage. Returns 0 (lit)
// .. 1 (fully shadowed); the caller scales by sky.shadowStrength.
float cloudShadowAt(vec3 worldPos) {
    if (!sky.cloudsEnabled) return 0.0;
    if (sky.shadowStrength <= 0.001) return 0.0;
    vec3 sunDirTo = -normalize(ubo.lightDirection);
    if (sunDirTo.y < 0.03) return 0.0;
    float shadow = 0.0;
    if (sky.lowEnabled) {
        float h = sky.lowGeom.x - worldPos.y;
        if (h > 0.0) {
            vec2 hitXZ = worldPos.xz + sunDirTo.xz / sunDirTo.y * h;
            shadow += cloudTierCoverage(hitXZ, 0) * max(sky.lowTier.y, 0.0) * 0.55;
        }
    }
    if (sky.midEnabled) {
        float h = sky.midGeom.x - worldPos.y;
        if (h > 0.0) {
            vec2 hitXZ = worldPos.xz + sunDirTo.xz / sunDirTo.y * h;
            shadow += cloudTierCoverage(hitXZ, 1) * max(sky.midTier.y, 0.0) * 0.40;
        }
    }
    if (sky.highEnabled) {
        float h = sky.highGeom.x - worldPos.y;
        if (h > 0.0) {
            vec2 hitXZ = worldPos.xz + sunDirTo.xz / sunDirTo.y * h;
            // Cirrus barely shadows (ice crystals, optically thin).
            shadow += cloudTierCoverage(hitXZ, 2) * max(sky.highTier.y, 0.0) * 0.18;
        }
    }
    shadow *= max(sky.densityScale, 0.0) * clamp(sky.shadowStrength, 0.0, 1.0);
    return 1.0 - exp(-max(shadow, 0.0) * 1.6);
}

// Single-sample approximation for the inline procedural-sky fallback (RT off):
// returns the cloud albedo to add over the gradient sky for a reflect ray.
vec3 cloudApproxForReflection(vec3 reflDir, vec3 sunDir, float dayFactor) {
    if (!sky.cloudsEnabled) return vec3(0.0);
    if (reflDir.y < 0.02 || dayFactor <= 0.01) return vec3(0.0);
    float cov = 0.0;
    float wsum = 0.0;
    if (sky.lowEnabled) { cov += cloudTierCoverage(reflDir.xz / max(reflDir.y, 0.05) * 2000.0, 0) * 0.5; wsum += 0.5; }
    if (sky.midEnabled) { cov += cloudTierCoverage(reflDir.xz / max(reflDir.y, 0.05) * 4500.0, 1) * 0.35; wsum += 0.35; }
    if (sky.highEnabled) { cov += cloudTierCoverage(reflDir.xz / max(reflDir.y, 0.05) * 9000.0, 2) * 0.25; wsum += 0.25; }
    if (wsum <= 0.0) return vec3(0.0);
    cov /= wsum;
    float sunAmt = max(dot(reflDir, sunDir), 0.0);
    vec3 cloudCol = mix(vec3(0.75, 0.78, 0.82), vec3(1.05, 1.0, 0.95),
                        pow(sunAmt, 3.0) * dayFactor);
    return cloudCol * cov * smoothstep(0.02, 0.25, reflDir.y) * dayFactor
         * clamp(sky.exposure, 0.1, 3.0);
}

#endif // CLOUDS_GLSL
