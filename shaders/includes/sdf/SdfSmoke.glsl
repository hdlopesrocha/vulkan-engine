// Procedural SDF smoke-bomb + bullet interaction (CS-style smoke grenade).
//
// Built on the generic SDF framework: the smoke volume is a Smoke-primitive
// instance marched by the shared SDF raymarcher; this module adds the
// smoke-specific domain (growth envelope, layered noise, wind) and the
// bullet field (capsule path, tunnel, reference air compression / ripple /
// heat, wake, turbulence, refill). Everything evaluates procedurally on the
// GPU from global time: in-flight bullets need no CPU updates.
//
// Density pipeline per sample (§14 evaluation order):
//   ray/sphere interval (caller) -> sphere SDF test -> cheap bound reject
//   -> bullet geometry field (capsules only, no noise)
//   -> noise-domain displacement (pressure + wake + swirl + wind)
//   -> layered noise (large/medium/fine) -> envelope/mass
//   -> tunnel thinning + heat/compression responses + temporal dissipation
//   -> lighting (sun scatter + cheap self-shadow) -> accumulation (caller)
//
// Requires: sdf_noise.glsl included first. The SmokeBlock buffer below must
// be declared exactly once (SdfRenderer.frag includes this file).

#ifndef SDF_SMOKE_GLSL
#define SDF_SMOKE_GLSL

#include "../../types/Bullet.glsl"
#include "../../types/SdfDefinition.glsl"
#include "../../types/SdfInstance.glsl"
#include "../../types/SdfMaterial.glsl"
#include "../../types/SmokeBulletFX.glsl"
#include "../../types/SmokeBulletState.glsl"
#include "../../types/SmokeFragBullet.glsl"
#include "../../types/Smoke.glsl"
#include "../../types/SmokeSample.glsl"

// ── Local-frame contract ────────────────────────────────────────────────
// Every function below receives points in the smoke instance's LOCAL frame
// (meters, origin at the shape center, rotated by the generic SdfModel in
// sdf_model.glsl). The smoke module performs no transform of its own: wind
// and bullets are packed into this frame by the caller / streamed state.

// Runtime state block (set=1 binding 8); type in
// shaders/types/SmokeFragBullet.glsl, CPU twin
// vulkan/types/SmokeFragBullet.hpp.
layout(std430, set = 1, binding = 8) readonly buffer SmokeBlock {
    SmokeFragBullet smokeGpu;
};

// Loop-local time in [0, loopDur): the smoke animation repeats every loop.
float smokeLoopT(float time, float loopDur) {
    return mod(time, max(loopDur, 1e-3));
}

// Local TAU: sdf_smoke.glsl is included before the caller defines its own.
const float SDF_SMOKE_TAU = 6.2831853;

// Shape SDF at size r, local space: cloud (0) and sphere (1) use the
// sphere; cube (2) is a box whose bounding sphere equals r. Object rotation
// and placement come from the generic SdfModel (the local frame is already
// rotated/centered by the evaluator).
float smokeShapeSDF(vec3 o, float r) {
    if (smokeGpu.tuning.shape > 1.5) {
        float h = r * 0.5773503;
        vec3 d = abs(o) - vec3(h);
        return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
    }
    return length(o) - r;
}

// Density edge band (fraction of r): the cloud keeps its soft billowy rim;
// Sphere/Cube are crisper dense bodies.
float smokeShapeBand() {
    return (smokeGpu.tuning.shape > 0.5) ? 0.06 : 0.18;
}

// Smooth growth curve (§19): rapid expansion that decelerates over the
// growth window (velocity highest at t=0, decaying as it approaches the
// duration), normalized to land exactly at the master scale when it ends, so
// the cloud is full-size and visually still right at growthDur. Radius only
// grows; density handles thinning + end fade for a seamless loop.
float smokeGrowthRadius(float shapeScale, float loopT, float growthDur) {
    const float k = 3.5;
    float g = clamp(loopT / max(growthDur, 1e-3), 0.0, 1.0);
    float c = (1.0 - exp(-k * g)) / (1.0 - exp(-k));
    return max(shapeScale * c, 1e-3);
}

// End-of-loop fade so the wrap-around is seamless.
float smokeEndFade(float loopT, float loopDur) {
    return 1.0 - smoothstep(0.85, 1.0, loopT / max(loopDur, 1e-3));
}

// Reference debug field ramp (blue -> cyan -> green -> yellow -> red), used
// by the compression / ripple / heat field views.
vec3 smokeDebugRamp(float v) {
    v = clamp(v, 0.0, 1.0);
    vec3 c = mix(vec3(0.05, 0.1, 0.5), vec3(0.0, 0.8, 1.0), smoothstep(0.0, 0.35, v));
    c = mix(c, vec3(0.1, 0.9, 0.3), smoothstep(0.35, 0.6, v));
    c = mix(c, vec3(1.0, 0.85, 0.1), smoothstep(0.6, 0.8, v));
    return mix(c, vec3(1.0, 0.1, 0.1), smoothstep(0.8, 1.0, v));
}

// Closest point on the finite bullet segment + longitudinal coordinate.
// Returns (Q.xyz, s) with s clamped to [0, pathLen].
vec4 smokeBulletClosest(vec3 p, Bullet bl) {
    vec3 D = bl.velocity / max(length(bl.velocity), 1e-6);
    float s = clamp(dot(p - bl.start, D), 0.0, max(bl.pathLength, 1e-3));
    return vec4(bl.start + D * s, s);
}

// Finite capsule SDF of the bullet path (for the marched field, §4).
float smokeBulletSDF(vec3 p, Bullet bl, float radius) {
    vec4 q = smokeBulletClosest(p, bl);
    return length(p - q.xyz) - max(radius, 1e-3);
}

// Wake-aged carve radius: the bore starts at the bullet's own radius at
// the head (tau = 0, fresh passage) and widens behind it as entrained air
// drifts outward. Outflow u0 = entrainment * bullet speed (wake.y) loses
// energy to air drag lambda (tunnel.w, the refill rate); integrated width:
//   spread(tau) = (u0/lambda) * (1 - exp(-lambda * tau)),
// so the final radius grows with time toward base + u0/lambda, driven by
// bullet volume (r0 sets the displaced air mass) and velocity (sets u0).
// Cutoff: spread rate and fade fall below 1e-3 past
// tauCut = ln(1000)/lambda (~4.6 s at defaults), inside the wake grace.
float smokeBulletRadius(Bullet bl, float s, float traveled, float age, Smoke t) {
    float f = clamp(s / max(traveled, 1e-3), 0.0, 1.0);
    float base = mix(bl.radiusStart, bl.radiusEnd, f);
    float speed = max(length(bl.velocity), 1e-3);
    float tau = max(age - s / speed, 0.0);
    float lambda = max(t.wakeDissipation, 0.05);
    float u0 = max(t.wakeExpansion, 0.0) * speed;
    float spread = (u0 / max(lambda, 1e-3)) * (1.0 - exp(-lambda * tau));
    return base + spread;
}


SmokeBulletState smokeBulletState(Bullet bl, float time) {
    SmokeBulletState st;
    float speed = max(length(bl.velocity), 1e-3);
    float pathLen = max(bl.pathLength, 1e-3);
    float travelTime = pathLen / speed;
    // Looping projectile (auto bullet): one cycle per loopDuration, mod on
    // the loop clock. The round is PARKED at its initial position (path
    // start = smoke surface) for the whole smoke expansion, then moves:
    //   tMove = max(mod(time, loopDuration) - expansionDuration, 0).
    // bl.phase carries the expansion duration, so the start of movement and
    // the smoke bloom always agree.
    // One-shot manual round (c.y == 0): born at c.w, moves immediately.
    if (bl.loopDuration > 0.0) {
        float loopT = mod(time, max(bl.loopDuration, 1e-3));
        st.age = max(loopT - bl.phase, 0.0);
    } else {
        st.age = max(time - bl.phase, 0.0);
    }
    st.traveled = clamp(speed * st.age, 0.0, pathLen);
    st.headOnPath = st.age <= travelTime + 0.05;
    // Grace window past the pass for wake visibility + refill.
    st.live = (bl.intensity > 0.0) && (st.age <= travelTime + 8.0);
    return st;
}

// Motion-blurred tunnel core [0,1] with temporal fade INCLUDED: the max
// over trailing head positions keeps the bore continuous when the head
// jumps many meters per frame (low fps strobing looks like z-fighting).
// Capsule-only + one 4D wall-disturbance eval (not per blur tap) — cheap
// enough for march + shadow samples.
float smokeTunnelCore(vec3 p, Bullet bl, SmokeBulletState st, Smoke t, float time) {
    float speed = max(length(bl.velocity), 1e-3);
    float refill = max(t.wakeDissipation, 0.05);
    float fall = max(t.tunnelFalloff, 0.5);
    vec3 D = bl.velocity / max(length(bl.velocity), 1e-6);
    float sRaw = dot(p - bl.start, D);
    // Infinite cone behind the launch (no early-out at sRaw < 0): the wake
    // persists down the endlessly extended axis, aging (and refilling) with
    // distance behind. Only the tip is round (handled after the loop).
    // Tapered carve radius at this query's (clamped) path coordinate.
    float sQ = clamp(sRaw, 0.0, st.traveled);
    float radius = max(smokeBulletRadius(bl, sQ, st.traveled, st.age, t), 0.5);
    // Cone-surface disturbance: 4D Perlin (space + time) whose amplitude
    // ramps 0 at the bullet surface to a wave growing with sqrt(distance
    // from the wall) — shear-layer physics: the body constrains the
    // interface at the wall, instability waves grow downstream and away.
    // Strength/speed reuse the Turbulence knobs; bl.phase decorrelates rounds.
    float wob = sdfNoise4(vec4(p * (1.5 / max(radius, 1.0)) + bl.phase,
                               time * max(t.turbSpeed, 0.0))) - 0.5;
    float wobAmp = max(t.turbStrength, 0.0);
    float wScale = radius + fall;
    float span = min(speed * 0.25, radius * 2.0);
    float core = 0.0;
    for (int m = 0; m < 3; ++m) {
        float back = span * float(m) * 0.5;
        float sH = max(st.traveled - back, 0.0);
        if (sRaw > sH) continue;
        float rt = length(p - (bl.start + D * sRaw));
        float ramp = sqrt(max(rt - radius, 0.0) / max(wScale, 1e-3));
        float rEff = rt + wob * 2.0 * wobAmp * ramp;
        float cm = 1.0 - smoothstep(radius, radius + fall, rEff);
        float passAge = max(st.age - sRaw / speed, 0.0);
        core = max(core, cm * exp(-passAge * refill));
    }
    // Round tip: just past the head, thin against the head point so the cap
    // is spherical, not a flat cut. Fresh passage (passAge ~ 0 here).
    float over = sRaw - st.traveled;
    float Rhead = max(smokeBulletRadius(bl, st.traveled, st.traveled, st.age, t), 0.5);
    if (over > 0.0 && over <= Rhead) {
        float rtH = length(p - (bl.start + D * st.traveled));
        float cmH = 1.0 - smoothstep(Rhead, Rhead + fall, rtH);
        float fadeH = exp(-max(st.age - st.traveled / speed, 0.0) * refill);
        core = max(core, cmH * fadeH);
    }
    return core;
}


// ── Signed velocity field (SVF) of the bullet's air-drag cone ────────────
// The round drags air along its path; SVF(p) is that air's velocity <X,Y,Z>
// in m/s at p, with two parts: (1) radial outflow perpendicular to the
// path (the cone cross-section), (2) axial slipstream along +D — this is
// the smoke's wind, and it comes from bullet movement. Both shaped by the
// cone (full inside the bore wall, falling off over the tunnel-falloff
// band) and damped by air drag as the passage ages.
// The caller turns velocity into the air displacement (see the smoke sample
// path): disp = SVF * tau, tau = drag-integrated elapsed time since the
// smoke expansion finished — bounded by 1/lambda, so the push saturates and
// the wave provably has no more effect.
vec3 smokeBulletSVF(vec3 p, float time, Smoke t, out float magOut) {
    vec3 svf = vec3(0.0);
    magOut = 0.0;
    for (int i = 0; i < 8; ++i) {
        Bullet bl = smokeGpu.bullets[i];
        if (bl.intensity <= 0.0) continue;
        SmokeBulletState st = smokeBulletState(bl, time);
        if (!st.live || st.traveled <= 0.0) continue;
        vec3 D = bl.velocity / max(length(bl.velocity), 1e-6);
        float speed = max(length(bl.velocity), 1e-3);
        float sRaw = dot(p - bl.start, D);
        if (sRaw < 0.0 || sRaw > st.traveled) continue;
        vec3 Rv = p - (bl.start + D * sRaw);
        float r = length(Rv);
        // Pure radial direction: strip the axial component. Degenerate on
        // the path axis (no outward sign there -> no push).
        vec3 perp = Rv - D * dot(Rv, D);
        float rp = length(perp);
        if (rp < 1e-3) continue;
        vec3 rdir = perp / rp;
        float Rout = max(smokeBulletRadius(bl, sRaw, st.traveled, st.age, t), 0.5);
        float fall = max(t.tunnelFalloff, 0.5);
        float cone = 1.0 - smoothstep(Rout, Rout + fall, r);
        if (cone <= 1e-3) continue;
        float passAge = max(st.age - sRaw / speed, 0.0);
        float drag = exp(-passAge * max(t.wakeDissipation, 0.05));
        float wlong = exp(-(st.traveled - sRaw) / max(t.wakeLength, 1.0));
        float umag = max(t.wakeExpansion, 0.0) * speed;
        // (1) radial outflow, (2) axial slipstream along the flight
        // direction, strongest just behind the head, dying downstream.
        // Both gated by the cone: no push far from the bore.
        svf += (rdir * umag * cone + D * umag * wlong * cone) * drag;
        magOut = max(magOut, length(svf));
    }
    return svf;
}

SmokeBulletFX smokeBulletFX(vec3 p, float time, Smoke t) {
    SmokeBulletFX fx;
    fx.thin = 0.0;
    fx.displace = vec3(0.0);
    fx.wave = 0.0;
    fx.wake = 0.0;
    fx.turb = 0.0;
    fx.compress = 0.0;
    fx.heat = 0.0;
    float refillRate = max(t.wakeDissipation, 0.05);
    // Loop phase drives the compression ripple: the reference's phase*TAU
    // offsets are loop-locked, so the pattern wraps seamlessly.
    float loopDurFX = max(t.loopDuration, 1.0);
    float loopPhaseFX = smokeLoopT(time, loopDurFX) / loopDurFX;
    for (int i = 0; i < 8; ++i) {
        Bullet bl = smokeGpu.bullets[i];
        if (bl.intensity <= 0.0) continue;
        SmokeBulletState st = smokeBulletState(bl, time);
        if (!st.live) continue;
        vec3 D = bl.velocity / max(length(bl.velocity), 1e-6);
        float bSpeed = max(length(bl.velocity), 1e-3);
        float pathLen = max(bl.pathLength, 1e-3);
        float sRaw = dot(p - bl.start, D);
        float rBoreHead = max(smokeBulletRadius(bl, st.traveled, st.traveled, st.age, t), 0.5);
        // --- nose stagnation (reference): crushed air ahead of the head,
        // gaussian in nose-local coordinates scaled by the head radius ---
        if (sRaw > st.traveled - rBoreHead * 0.1) {
            vec3 headPos = bl.start + D * st.traveled;
            vec3 ahead = p - headPos;
            float sAhead = dot(ahead, D);
            float perpA = length(ahead - D * sAhead);
            float sa = sAhead / (0.35 * rBoreHead);
            float sr = perpA / (0.30 * rBoreHead);
            float stag = exp(-(sa * sa + sr * sr))
                       * smoothstep(-0.1, 0.15, sAhead / rBoreHead);
            // Signed response: add (like the reference's single compression
            // value); max() would shadow the negative core rarefaction.
            fx.compress += stag * 0.8;
            // Forward domain push (our SVF-domain advection, kept).
            float rr = max(length(ahead), 1e-3);
            float aheadW = max(1.0 - (sRaw - st.traveled) / max(t.wakeLength, 1.0), 0.0);
            float comp = exp(-rr / max(t.pressureRadius, 0.5)) * aheadW;
            fx.displace += (ahead / rr + D) * comp * t.pressureStrength * 0.5;
        }
        if (sRaw < 0.0 || sRaw > st.traveled) continue;
        float s = clamp(sRaw, 0.0, pathLen);
        vec3 Q = bl.start + D * s;
        vec3 R = p - Q;
        float r = length(R);
        // Passage age drives refill: the tunnel collapses back over time.
        // No hard cutoff here: fade decays smoothly to 0, so the refill
        // boundary never draws a visible seam ring on the cloud.
        float passAge = max(st.age - s / bSpeed, 0.0);
        float fade = exp(-passAge * refillRate);
        // --- tunnel core thinning (§5), fading as smoke refills (§12) ---
        float core = smokeTunnelCore(p, bl, st, t, time);
        fx.thin = max(fx.thin, core * clamp(t.tunnelStrength, 0.0, 1.0));
        // --- radial pressure displacement (§7): pushes noise domain outward,
        // and piles air at the rim ---
        float pr = max(t.pressureRadius, 0.5);
        float press = exp(-(r * r) / (2.0 * pr * pr));
        vec3 rdir = (r > 1e-4) ? (R / r) : vec3(0.0, 1.0, 0.0);
        fx.displace += rdir * press * t.pressureStrength * fade;
        // --- reference shock-cone fields, normalized by the local bore
        // radius so the reference coefficients apply directly: dn is the
        // signed wall distance in bore radii (negative inside the wall) ---
        float rBore = max(smokeBulletRadius(bl, s, st.traveled, st.age, t), 0.5);
        float dn = (r - rBore) / rBore;
        // Compression turbulence (the reference ripple / "wave"): the same
        // classic Perlin cnoise as the reference, loop-locked on the wall
        // distance. Replaces the old traveling sine.
        vec3 rq = p * (max(t.rippleFreq, 0.5) / rBore)
                + vec3(loopPhaseFX * SDF_SMOKE_TAU * 4.0, loopPhaseFX * SDF_SMOKE_TAU * 6.0, 0.0);
        float ripple = sdfCnoise(rq);
        float dnR = dn + ripple * max(t.rippleAmp, 0.0);
        fx.wave = max(fx.wave, ripple);
        // --- heat (reference): a band riding the rippled wall plus the
        // lingering wake trail behind the nose (analytic age since the nose
        // passed each station; constant-speed flight inverts exactly), no
        // attenuation while the round is live ---
        float heat = exp(-pow(dnR * 4.0, 2.0));
        if (s > 0.0 && passAge > 0.0) {
            float rrT = r / rBore;
            heat += exp(-rrT * rrT);
        }
        fx.heat = max(fx.heat, heat);
        // --- compression (reference): denser shell just outside the wall,
        // rarefied core inside; the nose stagnation is added above ---
        float shell = exp(-pow((dnR - 0.12) * 9.0, 2.0));
        float coreRare = 1.0 - smoothstep(-0.35, 0.05, dnR);
        fx.compress += shell * 1.2 - coreRare * 0.9;
        // --- wake metric (§9, debug view 8): bore-width at the head,
        // widening behind it, elongated and dissipating. The actual air
        // motion is no longer displaced here: the signed velocity field
        // (smokeBulletSVF) owns the wake displacement and is applied in
        // smokeSampleDensity as SVF * tau after the expansion finishes. ---
        float wakeR = max(max(rBore, t.wakeRadius), 0.5);
        float wlong = exp(-(st.traveled - s) / max(t.wakeLength, 1.0));
        float wakeI = exp(-pow(r / wakeR, 2.0)) * wlong * fade;
        fx.wake = max(fx.wake, wakeI * clamp(t.wakeStrength, 0.0, 2.0));
    }
    return fx;
}

// Layered smoke density WITHOUT bullets (shared by the march, the light
// march and shadow queries): shape envelope x soft boundary x warped
// large/medium/fine noise x expansion thinning. Pure GPU, animated.
// `o` is the primitive-local sample (the generic SdfModel already centered
// and rotated it), so the billow domain turns with the shape with no
// transform work in here. `windLocal` is the world wind pre-rotated into
// the same local frame by the caller.
float smokeBaseDensity(vec3 o, float rNow, float shapeScale, vec3 windLocal,
                       Smoke t, float time, float seed, float densityMul) {
    // Soft irregular boundary (SDF-side softness; breakup comes from noise).
    // Cloud: 18% rim (wider bands burned the march budget crossing near-empty
    // space, and the leftover steps then undersampled the noisy core — the
    // cloud's look tracked the camera). Sphere/Cube: 6% dense rim.
    float dShape = smokeShapeSDF(o, rNow);
    float band = smokeShapeBand();
    float edge = 1.0 - smoothstep(-band * rNow, 0.0, dShape);
    if (edge <= 0.0) return 0.0;
    // Wind advects the noise domain (pattern drifts with +wind, §18).
    vec3 wp = o - windLocal * time;
    // Noise sample scale, clamped to a resolvable band: out-of-range scales
    // (e.g. 256 1/m over a 256 m ball) alias to sub-millimetre features the
    // march cannot resolve, which renders as flat grey streaks.
    float ns = clamp(t.noiseScale, 0.001, 0.5);
    // Low-frequency warp of the sample domain (§2).
    vec3 wq = wp * ns * 0.15 + seed;
    vec3 warp = (vec3(sdfNoise(wq), sdfNoise(wq + 7.3), sdfNoise(wq + 3.1)) - 0.5)
              * max(t.noiseWarp, 0.0);
    vec3 q = wp * ns + warp;
    // 4D Perlin (x, y, z + time): the large/medium bands genuinely evolve
    // instead of just drifting. Time rate ~0.1/s with per-band offsets so
    // bands decorrelate; the cheap fine band keeps drifting in 3D.
    float t4 = time * 0.1;
    float nLarge = sdfFbmOct4(vec4(q + seed, t4 + seed), 4);
    float nMed = sdfFbmOct4(vec4(q * 2.7 + 13.7, t4 * 1.3 + seed + 7.3), 3);
    float nFine = sdfNoise(q * 6.1 + 4.2);
    float n = nLarge * 0.55 + nMed * 0.30 + nFine * 0.15;
    n *= max(t.noiseStrength, 0.0);
    // Billow shaping: quadratic falloff with a filament mix (wispy strands
    // read at mid noise, solid body at high noise), then silhouette erosion
    // chews the edge with the noise itself. Expansion thinning approximately
    // conserves mass as the volume grows (§19). Depth is measured by the
    // shape SDF: for the sphere this is exactly the old 1 - (dc/r)^2 curve
    // (dn = 1 - dc/r), while cube cores stay dense to their faces.
    float dn = clamp(-dShape / max(rNow, 1e-3), 0.0, 1.0);
    float fall = clamp(2.0 * dn - dn * dn, 0.0, 1.0);
    float filament = smoothstep(-0.25, 0.65, n - 0.5);
    float shaped = pow(fall, 1.5) * mix(0.25, 1.0, filament);
    shaped *= smoothstep(0.0, 0.45, fall + (n - 0.5) * 0.6);
    float mass = mix(1.0, 0.45, clamp(rNow / max(shapeScale, 1e-3), 0.0, 1.0));
    return clamp(shaped, 0.0, 2.0) * edge * mass * max(densityMul, 0.0);
}


SmokeSample smokeSampleDensity(vec3 p, float rNow, float shapeScale, vec3 windLocal,
                               Smoke t, float time, float seed, float densityMul) {
    SmokeSample s;
    s.sdf = smokeShapeSDF(p, rNow);
    s.bullet = 1e5;
    s.tunnel = 0.0;
    s.pressure = 0.0;
    s.wave = 0.0;
    s.turb = 0.0;
    s.wake = 0.0;
    // 1. Bullet geometry field first (capsules only — cheap, §14 order).
    // Track the nearest traveled head segment for the debug view + march carve.
    for (int i = 0; i < 8; ++i) {
        Bullet bl = smokeGpu.bullets[i];
        if (bl.intensity <= 0.0) continue;
        SmokeBulletState st = smokeBulletState(bl, time);
        if (!st.live || st.traveled <= 0.0) continue;
        vec3 D = bl.velocity / max(length(bl.velocity), 1e-6);
        float sHead = dot(p - bl.start, D);
        float sC = clamp(sHead, 0.0, st.traveled);
        float bd = length(p - (bl.start + D * sC))
                 - max(smokeBulletRadius(bl, sC, st.traveled, st.age, t), 0.25);
        s.bullet = min(s.bullet, bd);
    }
    SmokeBulletFX fx = smokeBulletFX(p, time, t);
    s.tunnel = fx.thin;
    s.wave = fx.wave;         // signed ripple (debug view 6)
    s.wake = fx.wake;
    s.pressure = fx.compress; // signed reference compression (debug view 5)
    // Signed velocity field (SVF): air dragged outward by the bullet cone.
    // Displacement = SVF * tau, tau = drag-integrated time since the smoke
    // expansion finished (t > growth duration), so the field only pushes
    // once the cloud exists, saturates as air resistance wins (1/lambda)
    // and is provably inert past the cutoff. Applied by SUBTRACTING from
    // the sample point (p - disp): the smoke pattern moves outward with
    // the air while the density function itself stays unchanged.
    float loopT = smokeLoopT(time, t.loopDuration);
    float svfElapsed = loopT - max(t.growthDuration, 0.0);
    if (svfElapsed > 0.0) {
        float svfMag = 0.0;
        vec3 svf = smokeBulletSVF(p, time, t, svfMag);
        float lambda = max(t.wakeDissipation, 0.05);
        float tau = (1.0 - exp(-lambda * svfElapsed)) / lambda;
        if (tau > 1e-3) fx.displace -= svf * tau;
        s.turb = svfMag; // debug view 7: SVF magnitude (m/s)
    }
    // 2. Displaced noise-domain density (base layered field).
    float base = smokeBaseDensity(p + fx.displace, rNow, shapeScale, windLocal, t, time, seed, densityMul);
    s.density = base;
    s.heat = fx.heat; // reference heat: wall band + lingering wake trail
    // 3. Tunnel thinning + reference heat / compression responses.
    float dens = base;
    dens *= 1.0 - clamp(fx.thin, 0.0, 0.95);
    // Hot air expands (thinner) and is churned by the compression ripple:
    // reference dens *= (1 - 0.7*heat) * (1 + heat*rippleAmp*9*cnoise).
    float heatM = clamp(fx.heat, 0.0, 2.0);
    dens *= (1.0 - 0.7 * heatM)
          * (1.0 + heatM * max(t.rippleAmp, 0.0) * 9.0 * clamp(fx.wave, -1.0, 1.0));
    // Air compression (reference): shock-shell squeeze + nose stagnation -
    // core rarefaction. Push scales with the strength slider (/4: default
    // 6 -> 1.5 like the reference).
    dens *= clamp(1.0 + (t.pressureStrength / 4.0) * clamp(fx.compress, -2.0, 2.0), 0.0, 3.0);
    // 4. Loop-end dissipation fade (seamless 10 s repeat).
    dens *= smokeEndFade(loopT, t.loopDuration);
    // Signed responses can drive the product negative near the hot core;
    // clamp like the reference (vec2(max(dens, 0.0), heat)).
    s.finalD = max(dens, 0.0);
    return s;
}

// Cheap density for light-march samples (§16): base field only, no bullet
// displacement/wake noise (the tunnel CORE is still honored so holes let
// light through).
float smokeShadowDensity(vec3 p, float rNow, float shapeScale, vec3 windLocal,
                         Smoke t, float time, float seed, float densityMul) {
    float dens = smokeBaseDensity(p, rNow, shapeScale, windLocal, t, time, seed, densityMul);
    for (int i = 0; i < 8; ++i) {
        Bullet bl = smokeGpu.bullets[i];
        if (bl.intensity <= 0.0) continue;
        SmokeBulletState st = smokeBulletState(bl, time);
        if (!st.live || st.traveled <= 0.0) continue;
        dens *= 1.0 - smokeTunnelCore(p, bl, st, t, time) * clamp(t.tunnelStrength, 0.0, 1.0);
    }
    float loopT = smokeLoopT(time, t.loopDuration);
    return dens * smokeEndFade(loopT, t.loopDuration);
}

// Sun scattering for smoke (§15/§17): gray ramp by lighting (dark gray ->
// gray -> warm gray), fixed-correlation HG-ish forward boost, cheap
// self-shadow march toward the sun (§16). All directions are in the smoke's
// local frame (the caller rotated them through the SdfModel).
vec3 smokeShade(vec3 p, vec3 viewDir, float rNow, float shapeScale, vec3 windLocal,
                Smoke t, SdfMaterial mat, float time,
                float seed, float densityMul, vec3 sunDir, vec3 sunColor,
                float densS, float heatS, out float outTrans) {
    int shadowSteps = int(clamp(t.shadowSamples, 1.0, 8.0));
    // LOCAL self-shadow: the march resolves density variation around the
    // sample (a few noise wavelengths), NOT the whole ball. The old
    // rNow*1.2 length integrated ~300 m of smoke in 3 steps, so every sample
    // saturated to trans≈0 and the cloud rendered as a black blob with only
    // a bright rim where the march exited the ball early.
    float shadowLen = clamp(rNow * 0.15, 8.0, 64.0);
    float ldt = shadowLen / float(shadowSteps);
    float trans = 1.0;
    vec3 lp = p + sunDir * ldt * 0.5;
    for (int j = 0; j < 8; ++j) {
        if (j >= shadowSteps) break;
        // Extinction scale matches the accumulation path (0.08): the raw
        // material value is far too strong for a cloud hundreds of meters
        // across (one step would go opaque).
        trans *= exp(-smokeShadowDensity(lp, rNow, shapeScale, windLocal, t, time,
                                         seed, densityMul)
                     * ldt * max(mat.absorption, 0.0) * 0.08);
        lp += sunDir * ldt;
    }
    trans = mix(1.0, trans, clamp(t.shadowStrength, 0.0, 1.0));
    // Scattering response: the body brightens with local density, the sun
    // scatters through a true Henyey-Greenstein lobe (g = 0.3), and hot air
    // adds its own warm emission. The scattering slider (mat.scattering,
    // default 4 -> 1.0x sun brightness) is the look's scatter control.
    float cosT = dot(viewDir, sunDir);
    const float hgG = 0.3;
    float hgDen = 1.0 + hgG * hgG - 2.0 * hgG * clamp(cosT, -1.0, 1.0);
    float phaseHG = (1.0 - hgG * hgG) / pow(max(hgDen, 1e-3), 1.5);
    float lightLevel = clamp(trans * (0.35 + 0.65 * clamp(phaseHG, 0.0, 2.0) * 0.5), 0.0, 1.0);
    // User smoke color: lit albedo; shadowed end scales with it so any tint
    // stays consistent.
    vec3 smokeCol = vec3(t.smokeColor);
    vec3 albedo = mix(smokeCol * 0.42, smokeCol, lightLevel);
    vec3 heatWarm = mix(smokeCol, vec3(1.0, 0.48, 0.15), 0.75);
    outTrans = trans;
    return albedo * (0.22 + 0.45 * densS)
         + sunColor * phaseHG * (max(mat.scattering, 0.0) * 0.25) * trans
         + heatWarm * (heatS * max(t.heatStrength, 0.0));
}

// Marched field for Smoke instances (§4 + §14), evaluated in the shape's
// LOCAL frame (the generic SdfModel already transformed the point): the
// grown shape, nothing else. The bullet tunnel is NOT carved here by
// design: it shows through the density (thinning) and the depth/shade path
// instead, so the march keeps clean sphere steps and the air-compression
// shading reads the undisturbed depth. (A carved SDF here used to force
// max-length steps through refilled smoke and alias the noise.)
float smokeMarchSDF(vec3 q, SdfDefinition def, float time) {
    float shapeScale = max(def.params0.x, 1.0);
    float loopDur = max(smokeGpu.tuning.loopDuration, 1.0);
    float loopT = smokeLoopT(time, loopDur);
    float rNow = smokeGrowthRadius(shapeScale, loopT, max(smokeGpu.tuning.growthDuration, 0.5));
    // Solid shapes (Sphere/Cube) march as exact opaque surfaces: the renderer
    // shades the zero crossing in every render mode (reference plastic bodies).
    // The envelope mirrors the reference loop: grow-in at the start, shrink
    // before the wrap, so restarting the loop never pops the solid away.
    if (smokeGpu.tuning.shape > 0.5) {
        float solidR = max(rNow * smokeEndFade(loopT, loopDur), shapeScale * 0.001);
        return smokeShapeSDF(q, solidR);
    }
    // Cloud: the marched field bounds the DENSITY ENVELOPE (edge starts one
    // band inside the shape), not the visual surface: sphere tracing then
    // skims the density-free rim instead of stepping through it at interior
    // resolution.
    return smokeShapeSDF(q, rNow * (1.0 - smokeShapeBand()));
}

#endif // SDF_SMOKE_GLSL
