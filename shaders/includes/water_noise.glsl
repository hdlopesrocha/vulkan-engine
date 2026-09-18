// Shared water noise + wave helpers.
// Requires `includes/perlin.glsl` and `includes/ubo.glsl` (WaterParamsGPU)
// to be included first.
//
// The water surface is built from ONE wave height field, waterWaveField(),
// exposed through waterWaveSample() (height + analytic gradient) and
// waterWaveDisplacement() (height only). The same field drives the TES
// displacement, the per-pixel analytic shading normal and the wave-shape
// caustics, so geometry, lighting and caustics can never disagree.

float waterFbmNoise(vec3 xyz, float spatialScale, float time, float timeScale,
                    int octaves, float persistence, float lacunarity, vec3 offset) {
    return fbm(vec4((xyz + offset) * spatialScale, time * timeScale), octaves, persistence, lacunarity);
}

// Gradient-aware FBM. Returns vec4(value, d/dx, d/dy, d/dz) from a SINGLE
// noise evaluation; the analytic spatial gradient is propagated through the
// FBM octave chain.  Used by the wave chop and by waterRefractionNoise.
vec4 waterFbmNoiseGrad(vec3 xyz, float spatialScale, float time, float timeScale,
                       int octaves, float persistence, float lacunarity, vec3 offset) {
    vec4 r = fbmGrad4D(vec4((xyz + offset) * spatialScale, time * timeScale),
                       octaves, persistence, lacunarity);
    r.yzw *= spatialScale; // chain rule through the spatial scaling
    return r;
}

// Ridged Perlin multifractal for the sharp wave crests. `q` is the 2D
// along/across wave coordinate (along = propagation direction). Returns
// vec4(value, d/dq.x, d/dq.y, 0): value in [0,1] with sharp ridge lines at
// the Perlin zero crossings, amplified by the multifractal weighting
// (each octave multiplied by the previous ridge), and `ridgeSharpness`
// controls how narrow the crests are. The `slice` offset decorrelates the
// components sampled on the same z plane. Analytic spatial gradient is
// propagated through the |.| kink, the power and the octave chain rule.
vec4 waterRidgedFbmGrad(vec2 q, float time, int octaves, float persistence,
                        float lacunarity, float ridgeSharpness, float slice) {
    float s = max(ridgeSharpness, 0.25);
    float sum = 0.0;
    vec2 grad = vec2(0.0);
    float freq = 1.0;
    float amp = 1.0;
    float prev = 1.0;
    vec2 prevGrad = vec2(0.0);
    float norm = 0.0;
    for (int i = 0; i < octaves; i++) {
        vec4 n = perlinNoise4DGrad(vec4(q * freq, slice, time));
        float r = max(1.0 - abs(n.x), 0.0);
        float rp = pow(r, s);
        // d((1-|n|)^s)/dq = s*(1-|n|)^(s-1)*(-sign(n))*dn/dq, chained with
        // the octave frequency.
        vec2 dr = -s * pow(max(r, 1e-4), max(s - 1.0, 0.0))
                  * sign(n.x) * n.yz * freq;
        // Multifractal weighting: term = amp * prev * rp.
        sum += amp * prev * rp;
        grad += amp * (prevGrad * rp + prev * dr);
        norm += amp;
        prev = rp;
        prevGrad = dr;
        amp *= persistence;
        freq *= lacunarity;
    }
    return vec4(sum / max(norm, 1e-4), grad / max(norm, 1e-4), 0.0);
}

// Shared shore movement for every wave component. ALL trains reuse this:
// anisotropic ridged-noise coordinates whose along axis is the shore
// direction, scrolled toward the shore at the component speed (with the
// zone shoaling factor), plus the shared Perlin domain warp and the curly
// breaker hook. Returns the sample coordinate and its analytic world
// gradients (d q.x/d xz, d q.y/d xz).
struct WaterWaveCoord {
    vec2 q;    // ridged-noise sample coordinate (along, across)
    vec2 dqx;  // d(q.x)/d(world xz)
    vec2 dqy;  // d(q.y)/d(world xz)
};

WaterWaveCoord waterShoreWaveCoord(vec3 pos, vec2 moveDir, float freq, float speed,
                                   float phaseOffset, float speedFactor, float time,
                                   float ridgeStretch, float warp, vec2 warpGrad,
                                   float curlCurve) {
    WaterWaveCoord c;
    vec2 perp = vec2(-moveDir.y, moveDir.x);
    float kc = freq / ridgeStretch;
    float acrossN = kc * dot(pos.xz, perp);
    // Bounded parabolic crest hook (curly breaker), with its derivative where
    // the clamp is inactive.
    float hook = min(acrossN * acrossN, 9.0);
    float dhook = (acrossN * acrossN < 9.0) ? 2.0 * acrossN : 0.0;
    float along = dot(pos.xz, moveDir) + phaseOffset - speed * speedFactor * time;
    c.q = vec2(freq * along + warp + curlCurve * hook, acrossN);
    c.dqx = freq * moveDir + warpGrad + curlCurve * dhook * (kc * perp);
    c.dqy = kc * perp;
    return c;
}

struct WaterWaveField {
    float height;  // vertical displacement along the base normal
    vec3  grad;    // analytic d(height)/d(world position), y = 0 (height field)
    float foam;    // 0..1 whitewater coverage
    float contact; // 0..1 shoreline contact foam (water meets solid at depth 0)
};

// Thickness-zoned directional shore-wave field. The wave trains and their
// foam propagate along `shoreDirIn`, the per-point direction of decreasing
// water depth (toward the shore, computed from the depth gradient by the
// caller; the configured shoreWaveAngle is the fallback). Zones, shape,
// shoaling, mask and foam are all per-layer params (see WaterParamsGPU), no
// shader magic numbers beyond numerical guards.
//
//   depth < 0  : thickness unknown (no measured bottom) -> open deep water
//   d >= zDeep : full open-ocean swell
//   zBreak..zDeep : shoaling band (gains height, sharpens, whitecaps)
//   ~ zBreak   : breaker line (extra crest + foam birth)
//   zShallow..zBreak : foam rides shoreward and fades, amplitude decays
//   d < zShallow : residual line wave ending at the waterline (d = 0)
WaterWaveField waterWaveField(vec3 xyz, float time, float depth, float amp,
                              vec2 shoreDirIn, WaterParamsGPU wp, bool withFoam) {
    WaterWaveField f;
    f.height = 0.0;
    f.grad = vec3(0.0);
    f.foam = 0.0;
    f.contact = 0.0;
    if (wp.waveToggles.x < 0.5 || amp <= 0.0) return f;

    float zDeep = max(wp.waveZones.x, 1.0);
    float zBreak = clamp(wp.waveZones.y, 0.0, zDeep);
    float zShallow = clamp(wp.waveZones.z, 0.0, zBreak);
    float d = (depth < 0.0) ? zDeep : max(depth, 0.0);

    // Global depth taper of the wave HEIGHT: full in the deep zone, falling
    // monotonically to 0 at the waterline, applied to EVERY component (both
    // ridged trains, the chop and the curl). The smoothstep base gives a
    // zero-slope fade at BOTH ends, so there is no visible seam where the
    // taper starts (a raw pow() had a slope jump at zDeep). 0 disables it.
    float heightFalloff = max(wp.waveBreaker.w, 0.0);
    float heightTaper = (heightFalloff > 0.0)
        ? pow(smoothstep(0.0, zDeep, d), heightFalloff)
        : 1.0;

    // ── Thickness zones: amplitude, crest sharpness, shoaling celerity and
    //    breaker activity. Every parameter is interpolated with a smoothstep,
    //    so both the value AND its slope match across zDeep, zBreak and
    //    zShallow — the fades between zones have no visible seams. ──
    float env;         // amplitude envelope
    float sharp;       // crest sharpness
    float speedFactor; // phase-speed multiplier (1 = deep water)
    float breaking;    // 0..1 foam generation from the breaker
    float shoreBand;   // 0..1 progress through the sub-shallow shore band
    float tBreakDeep = max(zDeep - zBreak, 1e-3);
    float tShallowBreak = max(zBreak - zShallow, 1e-3);
    if (d >= zDeep) {
        env = 1.0;
        sharp = wp.waveShape.x;
        speedFactor = 1.0;
        breaking = 0.0;
        shoreBand = 0.0;
    } else if (d >= zBreak) {
        // Shoaling band: the wave gains height and gets sharper as it
        // approaches the break line; whitecaps ignite toward the end.
        float t = clamp((zDeep - d) / tBreakDeep, 0.0, 1.0);
        float e = t * t * (3.0 - 2.0 * t);
        env = mix(1.0, 1.0 + wp.waveShape.w, e);
        sharp = mix(wp.waveShape.x, wp.waveShape.y, e);
        speedFactor = mix(1.0, 1.0 - wp.waveShoal.x, e);
        breaking = smoothstep(wp.waveBreaker.z, 1.0, t);
        shoreBand = 0.0;
    } else if (d >= zShallow) {
        // After the crash: amplitude decays shoreward, foam rides and fades.
        float t = clamp((zBreak - d) / tShallowBreak, 0.0, 1.0);
        float e = t * t * (3.0 - 2.0 * t);
        float breakEnv = 1.0 + wp.waveShape.w;
        env = mix(breakEnv, wp.waveShoal.z,
                  pow(e, max(wp.waveShoal.y, 1e-3)));
        sharp = mix(wp.waveShape.y, wp.waveShape.z, e);
        speedFactor = 1.0 - wp.waveShoal.x;
        breaking = 1.0 - e;
        shoreBand = e;
    } else {
        // Residual "line" wave that ends at the waterline (d -> 0).
        float t = clamp(d / max(zShallow, 1e-3), 0.0, 1.0);
        float e = t * t * (3.0 - 2.0 * t);
        env = wp.waveShoal.z * e;
        sharp = wp.waveShape.z;
        speedFactor = 1.0 - wp.waveShoal.x;
        breaking = 0.0;
        shoreBand = 1.0;
    }

    // Breaker bump: an extra, sharp crest concentrated at the break depth.
    if (wp.waveBreaker.x > 0.0) {
        float x = (d - zBreak) / max(wp.waveShoal.w, 1e-3);
        env += wp.waveBreaker.x * exp(-x * x);
    }

    // ── Shore movement (shared by every train and the detail noise) ──
    // Shore direction: the direction of DECREASING water depth, computed per
    // point from the solid-depth gradient by the caller (falls back to the
    // configured shoreWaveAngle when the bottom cannot be measured). Every
    // component below reuses the SAME movement: it scrolls toward the shore
    // along this direction; the second train only differs in scale, speed,
    // phase offset and noise seed.
    vec2 shoreDir = (dot(shoreDirIn, shoreDirIn) > 1e-6)
        ? normalize(shoreDirIn)
        : normalize(wp.waveDirection.xy + vec2(1e-5, 0.0));
    vec3 shoreDir3 = vec3(shoreDir.x, 0.0, shoreDir.y);

    float k1 = max(wp.waveComponent1.x, 1e-4);
    float c1 = wp.waveComponent1.y;
    float k2 = max(wp.waveComponent2.x, 1e-4);
    float c2 = wp.waveComponent2.y;

    // ── Organic modulation: one gradient-aware Perlin FBM drives the local
    //    crest amplitude variation and the chop detail. The chop rides the
    //    same shore movement as the trains (advected at the primary speed),
    //    so no part of the surface drifts across the shore direction. ──
    float chopAmount = wp.waveBreaker.y;
    float ampVar = clamp(wp.waveWarp.y, 0.0, 0.95);
    float warpAmount = wp.waveWarp.x;
    vec3 shoreDrift = shoreDir3 * (c1 * speedFactor * time);
    float chopVal = 0.0;
    vec2 chopGrad = vec2(0.0);
    if (chopAmount > 0.0 || warpAmount != 0.0 || ampVar > 0.0) {
        vec4 chop = waterFbmNoiseGrad(xyz - shoreDrift, wp.params2.y, time, wp.params3.x,
                                      int(max(wp.params2.z, 1.0)), wp.params2.w,
                                      wp.params3.y, vec3(0.0));
        chopVal = chop.x;
        chopGrad = vec2(chop.y, chop.w); // d/dx, d/dz (offset is constant)
    }

    // Organic calm patches: a low-frequency noise mask can remove the waves
    // entirely in some places. It rides the SAME shore movement as the waves
    // (advected with the primary drift), so the pattern of where waves exist
    // also travels toward the shore instead of being geographically fixed.
    // Octaves/persistence/lacunarity reuse the per-layer noise spectrum.
    float mask = 1.0;
    if (wp.waveMask.x > 0.0 && wp.waveMask.z > 0.0) {
        float m = waterFbmNoise(xyz - shoreDrift, wp.waveMask.x, time, wp.waveMask.w,
                                int(max(wp.params2.z, 1.0)), wp.params2.w, wp.params3.y,
                                vec3(11.0)) * 0.5 + 0.5;
        mask = smoothstep(wp.waveMask.y,
                          wp.waveMask.y + wp.waveMask.z, m);
    }

    float ampMod = 1.0 + ampVar * chopVal;
    vec2 ampModGrad = ampVar * chopGrad;

    float ridgeStretch = max(wp.waveWarp.z, 1.0);
    float sharpC = max(sharp, 0.25);
    int octavesN = int(max(wp.params2.z, 1.0));
    float persistenceN = wp.params2.w;
    float lacunarityN = wp.params3.y;
    float warp = warpAmount * chopVal;
    vec2 warpGrad = warpAmount * chopGrad;

    // Curly plunging breaker, shared by both trains: the lip leans forward
    // (odd profile skew) and the crest line hooks (bounded parabolic
    // curvature across the movement direction).
    float curlSkew = clamp(wp.waveCurl.x, -0.9, 0.9) * breaking;
    float curlCurve = max(wp.waveCurl.y, 0.0) * breaking;

    // Both trains reuse the same shore movement helper.
    WaterWaveCoord coord1 = waterShoreWaveCoord(xyz, shoreDir, k1, c1, 0.0,
                                                speedFactor, time, ridgeStretch,
                                                warp, warpGrad, curlCurve);
    WaterWaveCoord coord2 = waterShoreWaveCoord(xyz, shoreDir, k2, c2,
                                                wp.waveComponent2.w, speedFactor,
                                                time, ridgeStretch, warp, warpGrad,
                                                curlCurve);
    vec4 r1 = waterRidgedFbmGrad(coord1.q, time, octavesN, persistenceN,
                                 lacunarityN, sharpC, 0.0);
    vec4 r2 = waterRidgedFbmGrad(coord2.q, time, octavesN, persistenceN,
                                 lacunarityN, sharpC, 17.0);
    vec2 g1 = r1.y * coord1.dqx + r1.z * coord1.dqy;
    vec2 g2 = r2.y * coord2.dqx + r2.z * coord2.dqy;

    // Odd profile skew s + curl*s*|s|: steepens/stretches one face of the
    // crest (the lip) without moving the mean level. ds is its derivative.
    float s1 = 2.0 * r1.x - 1.0;
    float s2 = 2.0 * r2.x - 1.0;
    float ds1 = max(1.0 + 2.0 * curlSkew * abs(s1), 0.1);
    float ds2 = max(1.0 + 2.0 * curlSkew * abs(s2), 0.1);
    float prof1 = s1 + curlSkew * s1 * abs(s1);
    float prof2 = s2 + curlSkew * s2 * abs(s2);

    float amp1 = wp.waveComponent1.z;
    float amp2 = wp.waveComponent2.z;
    float am1 = amp1 * ampMod;
    float am2 = amp2 * ampMod;
    float h = am1 * prof1 + am2 * prof2;
    vec2 gxz = 2.0 * (am1 * ds1 * g1 + am2 * ds2 * g2);
    // Product rule for the amplitude-modulation envelope.
    gxz += (amp1 * prof1 + amp2 * prof2) * ampModGrad;

    // ── Chop detail rides on top (same FBM sample, same shore drift). ──
    if (chopAmount > 0.0) {
        h += chopAmount * chopVal;
        gxz += chopAmount * chopGrad;
    }

    float finalAmp = amp * env * mask * heightTaper;
    f.height = finalAmp * h;
    f.grad = vec3(finalAmp * gxz.x, 0.0, finalAmp * gxz.y);

    // ── Foam: born in the breaker band, carried shoreward by the crests and
    //    fades with depth; a residual line survives on the shallow band until
    //    the line wave reaches the waterline. ──
    if (withFoam && wp.waveToggles.y > 0.5) {
        float thr = clamp(wp.foamParams.x, 0.0, 0.98);
        float trailPhase = wp.foamParams.y;
        float lagGrowth = max(wp.foamShape.w, 0.0);
        // Defined foam edges: the threshold window narrows with Edge hardness.
        float edgeW = mix(0.35, 0.01, clamp(wp.foamShape.x, 0.0, 1.0));
        float thrHi = max(min(thr + edgeW, 1.0), thr + 1e-3);

        // Foam sits AFTER the curled lip: the trailing band is a first-order
        // lag of the SKEWED crest profile behind the crest, along the shore
        // direction, and the lag grows as the wave approaches the shore so the
        // foam falls behind (decelerates). Gradients are the analytic
        // d(prof)/dx = 2*ds*g, so no extra noise evaluation is needed.
        float crest = clamp(0.5 + 0.5 * max(prof1, prof2), 0.0, 1.0);
        float lag1 = (trailPhase / k1) * (1.0 + lagGrowth * shoreBand);
        float lag2 = (trailPhase / k2) * (1.0 + lagGrowth * shoreBand);
        float trailProf1 = clamp(prof1 - dot(2.0 * ds1 * g1, shoreDir) * lag1, -1.0, 1.0);
        float trailProf2 = clamp(prof2 - dot(2.0 * ds2 * g2, shoreDir) * lag2, -1.0, 1.0);
        float trail = max(0.5 + 0.5 * trailProf1, 0.5 + 0.5 * trailProf2);

        float whitecap = smoothstep(thr, thrHi, crest) * breaking;
        float trailing = smoothstep(thr, thrHi, trail) * breaking * breaking;
        float foam = max(whitecap, trailing);
        // Shoreline contact foam: the final line where the water meets the
        // solid. It arrives IN WAVES: each incoming crest pushes the line
        // further up the shore (wider band) and strengthens it, then it
        // recedes to the configured floor between crests. The foam noise/mask
        // break it up below, and the fragment stage forces the composite alpha
        // up for it so the last water pixels render the line.
        float contactWave = max(crest, trail);
        float contactPulse = mix(clamp(wp.foamContact.w, 0.0, 1.0), 1.0,
                                 smoothstep(thr, thrHi, contactWave));
        float contactWidth = max(wp.foamContact.x, 1e-3) * (0.5 + 0.5 * contactWave);
        float contact = (1.0 - smoothstep(0.0, contactWidth, d))
                      * clamp(wp.foamContact.y, 0.0, 1.0)
                      * contactPulse;
        if (d < zBreak) {
            // Extinction with distance below the break line.
            foam *= exp(-(zBreak - d) * max(wp.foamParams.z, 0.0));
            // Persistent foam line on the shallow band, vanishing with the
            // line wave at the waterline.
            float shoreFade = (d < zShallow)
                ? clamp(d / max(zShallow, 1e-3), 0.0, 1.0) : 1.0;
            foam = max(foam, wp.foamNoise.w * shoreBand * shoreFade *
                smoothstep(thr, thrHi, trail));
        }
        // Broken-up foam texture. The foam advects at its own speed profile:
        // fast at/after the curl, decaying toward the shore (foamShoreSpeed),
        // so whitewater races off the breaker and slows as it runs up.
        if (wp.foamNoise.z > 0.0) {
            float foamSpeedRel = mix(1.0, clamp(wp.foamShape.z, 0.0, 1.0), shoreBand);
            vec3 foamDrift = shoreDir3 * (c1 * speedFactor * foamSpeedRel * time);
            float fn = waterFbmNoise(xyz - foamDrift, wp.foamNoise.x, time, wp.foamNoise.y,
                                     int(max(wp.params2.z, 1.0)), wp.params2.w,
                                     wp.params3.y, vec3(37.0)) * 0.5 + 0.5;
            float fnMix = mix(1.0, fn, clamp(wp.foamNoise.z, 0.0, 1.0));
            foam *= fnMix;
            contact *= fnMix;
        }
        // Calm patches carry only the configured floor of foam.
        float maskMix = mix(clamp(wp.foamExtra.x, 0.0, 1.0), 1.0, mask);
        foam *= maskMix;
        contact *= maskMix;
        // Lighter foam: global coverage multiplier (translucency/airiness).
        float coverage = clamp(wp.foamShape.y, 0.0, 1.0);
        foam *= coverage;
        contact *= coverage;
        f.contact = clamp(contact, 0.0, 1.0);
        f.foam = clamp(max(foam, f.contact), 0.0, 1.0);
    }

    return f;
}

// Height-only entry point (tessellation adaptation, debug views).
float waterWaveDisplacement(vec3 xyz, float time, float depth, float amp,
                            vec2 shoreDir, WaterParamsGPU wp) {
    return waterWaveField(xyz, time, depth, amp, shoreDir, wp, false).height;
}

// Height + analytic gradient, vec4(height, dHeight/dx, dHeight/dy, dHeight/dz).
// The y component is 0: the field is a height field over xz.  This is the
// single wave field the TES displacement, per-pixel normal and caustics use.
vec4 waterWaveSample(vec3 xyz, float time, float depth, float amp,
                     vec2 shoreDir, WaterParamsGPU wp) {
    WaterWaveField f = waterWaveField(xyz, time, depth, amp, shoreDir, wp, false);
    return vec4(f.height, f.grad);
}

vec2 waterRefractionNoise(vec3 xyz, float noiseScale, float time,
                          int noiseOctaves, float noisePersistence, float noiseLacunarity) {
    float noise1 = waterFbmNoise(xyz, noiseScale * 0.15, time, 0.4,
                                 noiseOctaves, noisePersistence, noiseLacunarity, vec3(0.0));
    float noise2 = waterFbmNoise(xyz, noiseScale * 0.08, time, 0.25,
                                 max(noiseOctaves - 1, 1), noisePersistence, noiseLacunarity, vec3(100.0));
    float noise3 = waterFbmNoise(xyz, noiseScale * 0.30, time, 0.6,
                                 max(noiseOctaves - 2, 1), noisePersistence, noiseLacunarity, vec3(0.0));

    float nX = noise1 + noise2 * 0.5 + noise3 * 0.25;
    float nY = waterFbmNoise(xyz, noiseScale * 0.15, time, 0.4,
                             noiseOctaves, noisePersistence, noiseLacunarity, vec3(50.0)) + noise2 * 0.5;
    return vec2(nX, nY);
}
