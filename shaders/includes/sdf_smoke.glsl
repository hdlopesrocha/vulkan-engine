// Procedural SDF smoke-bomb + bullet interaction (CS-style smoke grenade).
//
// Built on the generic SDF framework: the smoke volume is a Smoke-primitive
// instance marched by the shared SDF raymarcher; this module adds the
// smoke-specific domain (growth envelope, layered noise, wind) and the
// bullet field (capsule path, tunnel, pressure, shock wave, wake,
// turbulence, refill). Everything evaluates procedurally on the GPU from
// global time: in-flight bullets need no CPU updates.
//
// Density pipeline per sample (§14 evaluation order):
//   ray/sphere interval (caller) -> sphere SDF test -> cheap bound reject
//   -> bullet geometry field (capsules only, no noise)
//   -> noise-domain displacement (pressure + wake + swirl + wind)
//   -> layered noise (large/medium/fine) -> envelope/mass
//   -> tunnel thinning + wave modulation + temporal dissipation
//   -> lighting (sun scatter + cheap self-shadow) -> accumulation (caller)
//
// Requires: sdf_noise.glsl included first. The SmokeBlock buffer below must
// be declared exactly once (sdf.frag includes this file).

#ifndef SDF_SMOKE_GLSL
#define SDF_SMOKE_GLSL

// Forward declaration: defined in sdf.frag below this include.
vec3 sdfWorldToLocal(vec3 wpos, SdfInstanceGPU inst, out float outScale);

// Mirrors SdfUBO.hpp (std430): SmokeGPU = 8 vec4, BulletGPU = 3 vec4.
struct SmokeGPU {
    vec4 timing;    // x = growth duration (s), y = loop duration (s),
                    // z = dissipation, w = unused
    vec4 noise;     // x = noise scale (1/m), y = noise strength,
                    // z = warp strength, w = unused
    vec4 wind;      // xy = wind velocity (m/s), z = live density scale, w = unused
    vec4 tunnel;    // x = tunnel strength, y = tunnel falloff (m),
                    // z = wake strength, w = wake dissipation = refill rate (1/s)
    vec4 wake;      // x = wake radius (m), y = wake expansion,
                    // z = wake length (m), w = unused
    vec4 pressure;  // x = pressure radius (m), y = pressure strength,
                    // z = wave speed (m/s), w = wave frequency (1/m)
    vec4 turbWave;  // x = wave falloff (1/m), y = turbulence scale (1/m),
                    // z = turbulence strength, w = turbulence speed (1/s)
    vec4 render;    // x = shadow samples, y = shadow strength, zw unused
};

struct BulletGPU {
    vec4 a; // xyz = path start (world), w = tunnel radius (m)
    vec4 b; // xyz = direction (unit), w = path length (m)
    vec4 c; // x = speed (m/s), y = birth time (s), z = intensity, w = flags (bit0 = auto-loop)
};

layout(std430, set = 1, binding = 8) readonly buffer SmokeBlock {
    SmokeGPU smokeTuning;
    BulletGPU smokeBullets[8];
};

// Loop-local time in [0, loopDur): the smoke animation repeats every loop.
float smokeLoopT(float time, float loopDur) {
    return mod(time, max(loopDur, 1e-3));
}

// Smooth growth curve (§19): rapid expansion over growthDur, then stable.
// Radius only grows; density handles thinning + end fade for a seamless loop.
float smokeGrowthRadius(float maxR, float loopT, float growthDur) {
    float g = clamp(loopT / max(growthDur, 1e-3), 0.0, 1.0);
    return max(maxR * (1.0 - exp(-3.5 * g)), 1e-3);
}

// End-of-loop fade so the wrap-around is seamless.
float smokeEndFade(float loopT, float loopDur) {
    return 1.0 - smoothstep(0.85, 1.0, loopT / max(loopDur, 1e-3));
}

// Closest point on the finite bullet segment + longitudinal coordinate.
// Returns (Q.xyz, s) with s clamped to [0, pathLen].
vec4 smokeBulletClosest(vec3 p, BulletGPU bl) {
    vec3 D = bl.b.xyz / max(length(bl.b.xyz), 1e-6);
    float s = clamp(dot(p - bl.a.xyz, D), 0.0, max(bl.b.w, 1e-3));
    return vec4(bl.a.xyz + D * s, s);
}

// Finite capsule SDF of the bullet path (for the marched field, §4).
float smokeBulletSDF(vec3 p, BulletGPU bl, float radius) {
    vec4 q = smokeBulletClosest(p, bl);
    return length(p - q.xyz) - max(radius, 1e-3);
}

// Per-bullet runtime state. Motion, wake age and refill are pure GPU
// functions of global time: auto-loop bullets are reborn every loop from
// widget params, manual bullets use their stamped birth time.
struct SmokeBulletState {
    bool live;
    float traveled; // head distance along path (m)
    float age;      // seconds since birth
};

SmokeBulletState smokeBulletState(BulletGPU bl, float time, float loopStart, float loopDur) {
    SmokeBulletState st;
    float birth = (((int(bl.c.w) & 1) != 0)) ? loopStart : bl.c.y;
    float speed = max(bl.c.x, 1e-3);
    float pathLen = max(bl.b.w, 1e-3);
    st.age = time - birth;
    st.traveled = clamp(speed * max(st.age, 0.0), 0.0, pathLen);
    // Grace window past the pass for wake visibility + refill.
    st.live = (bl.c.z > 0.0) && (st.age >= 0.0)
             && (st.age <= pathLen / speed + 8.0);
    return st;
}

// Motion-blurred tunnel core [0,1] with temporal fade INCLUDED: the max
// over trailing head positions keeps the bore continuous when the head
// jumps many meters per frame (low fps strobing looks like z-fighting).
// Capsule-only, no noise — cheap enough for march + shadow samples.
float smokeTunnelCore(vec3 p, BulletGPU bl, SmokeBulletState st, SmokeGPU t) {
    float speed = max(bl.c.x, 1e-3);
    float refill = max(t.tunnel.w, 0.05);
    float radius = max(bl.a.w, 0.5);
    float fall = max(t.tunnel.y, 0.5);
    vec3 D = bl.b.xyz / max(length(bl.b.xyz), 1e-6);
    float sRaw = dot(p - bl.a.xyz, D);
    if (sRaw < 0.0) return 0.0;
    float span = min(speed * 0.25, radius * 2.0);
    float core = 0.0;
    for (int m = 0; m < 3; ++m) {
        float back = span * float(m) * 0.5;
        float sH = max(st.traveled - back, 0.0);
        if (sRaw > sH) continue;
        float rt = length(p - (bl.a.xyz + D * sRaw));
        float cm = 1.0 - smoothstep(radius, radius + fall, rt);
        float passAge = max(st.age - sRaw / speed, 0.0);
        core = max(core, cm * exp(-passAge * refill));
    }
    return core;
}

// Bullet interaction field at p (geometry only, no noise — cheap): tunnel
// thinning [0,1], domain displacement, shock-wave value, wake and
// turbulence magnitudes (for debug views). Anisotropic by construction
// (§28): compression ahead of the head, radial push at the sides,
// expanding turbulent wake behind.
struct SmokeBulletFX {
    float thin;      // multiplicative density removal (tunnel core)
    vec3 displace;   // noise-domain displacement (pressure + wake + swirl)
    float wave;      // shock-wave modulation value
    float wake;      // wake influence (debug + thinning)
    float turb;      // turbulence magnitude (debug)
};

SmokeBulletFX smokeBulletFX(vec3 p, float time, float loopStart, float loopDur, SmokeGPU t) {
    SmokeBulletFX fx;
    fx.thin = 0.0;
    fx.displace = vec3(0.0);
    fx.wave = 0.0;
    fx.wake = 0.0;
    fx.turb = 0.0;
    float refillRate = max(t.tunnel.w, 0.05);
    for (int i = 0; i < 8; ++i) {
        BulletGPU bl = smokeBullets[i];
        if (bl.c.z <= 0.0) continue;
        SmokeBulletState st = smokeBulletState(bl, time, loopStart, loopDur);
        if (!st.live) continue;
        vec3 D = bl.b.xyz / max(length(bl.b.xyz), 1e-6);
        float pathLen = max(bl.b.w, 1e-3);
        float radius = max(bl.a.w, 0.5);
        float sRaw = dot(p - bl.a.xyz, D);
        // --- forward compression precursor ahead of the head (§28 front) ---
        if (sRaw > st.traveled && sRaw < st.traveled + max(t.wake.z, 1.0)) {
            vec3 headPos = bl.a.xyz + D * st.traveled;
            vec3 ahead = p - headPos;
            float rr = max(length(ahead), 1e-3);
            float aheadW = 1.0 - (sRaw - st.traveled) / max(t.wake.z, 1.0);
            float comp = exp(-rr / max(t.pressure.x, 0.5)) * aheadW;
            fx.displace += (ahead / rr + D) * comp * t.pressure.y * 0.5;
        }
        if (sRaw < 0.0 || sRaw > st.traveled) continue;
        float s = clamp(sRaw, 0.0, pathLen);
        vec3 Q = bl.a.xyz + D * s;
        vec3 R = p - Q;
        float r = length(R);
        // Passage age drives refill: the tunnel collapses back over time.
        float passAge = max(st.age - s / max(bl.c.x, 1e-3), 0.0);
        float fade = exp(-passAge * refillRate);
        if (fade <= 0.003) continue;
        // --- tunnel core thinning (§5), fading as smoke refills (§12) ---
        float core = smokeTunnelCore(p, bl, st, t);
        fx.thin = max(fx.thin, core * clamp(t.tunnel.x, 0.0, 1.0));
        // --- radial pressure displacement (§7): pushes noise domain outward ---
        float pr = max(t.pressure.x, 0.5);
        float press = exp(-(r * r) / (2.0 * pr * pr));
        vec3 rdir = (r > 1e-4) ? (R / r) : vec3(0.0, 1.0, 0.0);
        fx.displace += rdir * press * t.pressure.y * fade;
        // --- propagating shock wave (§8): radial rings, decaying with
        // passage age and distance, confined to the affected region ---
        float wave = sin(r * max(t.pressure.w, 0.5) - passAge * max(t.pressure.z, 0.5));
        wave *= exp(-r * max(t.turbWave.x, 1e-3)) * fade;
        fx.wave += wave;
        // --- turbulent wake (§9): narrow at the head, expanding downstream,
        // elongated, dissipating; analytic swirl avoids curl-noise cost ---
        float wakeR = max(t.wake.x + s * max(t.wake.y, 0.0), 0.5);
        float wlong = exp(-(st.traveled - s) / max(t.wake.z, 1.0));
        float wakeI = exp(-pow(r / wakeR, 2.0)) * wlong * fade;
        vec3 cx = cross(D, rdir);
        float ring = sin(min(r / wakeR * 3.14159, 3.14159)); // 0 on axis: no NaN swirl
        vec3 swirl = cx / max(length(cx), 1e-3) * ring;
        float turbPh = time * max(t.turbWave.w, 0.0) + r * max(t.turbWave.y, 1e-4);
        vec3 turbVec = swirl * sin(turbPh) * max(t.turbWave.z, 0.0);
        fx.displace += (rdir * wakeI * 1.5 + turbVec * wakeI) * clamp(t.tunnel.z, 0.0, 2.0);
        fx.wake = max(fx.wake, wakeI * clamp(t.tunnel.z, 0.0, 2.0));
        fx.turb = max(fx.turb, wakeI * max(t.turbWave.z, 0.0));
    }
    return fx;
}

// Layered smoke density WITHOUT bullets (shared by the march, the light
// march and shadow queries): growth envelope x soft boundary x warped
// large/medium/fine noise x expansion thinning. Pure GPU, animated.
float smokeBaseDensity(vec3 p, vec3 center, float rNow, float maxR,
                       SmokeGPU t, float time, float seed, float densityMul) {
    // Soft irregular boundary (SDF-side softness; breakup comes from noise).
    float dc = length(p - center);
    float edge = 1.0 - smoothstep(rNow * 0.55, rNow, dc);
    if (edge <= 0.0) return 0.0;
    // Wind advects the noise domain (pattern drifts with +wind, §18).
    vec3 wp = p - vec3(t.wind.x, 0.0, t.wind.y) * time;
    float ns = max(t.noise.x, 1e-5);
    // Low-frequency warp of the sample domain (§2).
    vec3 wq = wp * ns * 0.15 + seed;
    vec3 warp = (vec3(sdfNoise(wq), sdfNoise(wq + 7.3), sdfNoise(wq + 3.1)) - 0.5)
              * max(t.noise.z, 0.0);
    vec3 q = wp * ns + warp;
    float nLarge = sdfFbmOct(q + seed, 4);
    float nMed = sdfFbmOct(q * 2.7 + 13.7, 3);
    float nFine = sdfNoise(q * 6.1 + 4.2);
    float n = nLarge * 0.55 + nMed * 0.30 + nFine * 0.15;
    n *= max(t.noise.y, 0.0);
    // Wispy edge breakup from the fine band.
    float er = 0.75 + 0.5 * (nFine - 0.5);
    // Expansion thinning: approximately conserve mass as volume grows (§19).
    float mass = mix(1.0, 0.45, clamp(rNow / max(maxR, 1e-3), 0.0, 1.0));
    return clamp(n, 0.0, 2.0) * edge * er * mass * max(densityMul, 0.0);
}

// Full per-sample smoke evaluation (march path): base density at the
// bullet-displaced position, tunnel/wake thinning, wave modulation,
// temporal dissipation. Debug metrics are returned for the §24 views.
struct SmokeSample {
    float density;  // base layered density (pre-bullet, debug view 2)
    float sdf;      // grown sphere SDF (for debug)
    float bullet;   // min bullet-path SDF (for debug)
    float tunnel;
    float pressure;
    float wave;
    float turb;
    float wake;
    float finalD;   // post-tunnel/wave/fade density (debug view 9)
};

SmokeSample smokeSampleDensity(vec3 p, vec3 center, float rNow, float maxR,
                               SmokeGPU t, float time, float loopStart, float loopDur,
                               float seed, float densityMul) {
    SmokeSample s;
    s.sdf = length(p - center) - rNow;
    s.bullet = 1e5;
    s.tunnel = 0.0;
    s.pressure = 0.0;
    s.wave = 0.0;
    s.turb = 0.0;
    s.wake = 0.0;
    // 1. Bullet geometry field first (capsules only — cheap, §14 order).
    // Track the nearest traveled head segment for the debug view + march carve.
    for (int i = 0; i < 8; ++i) {
        BulletGPU bl = smokeBullets[i];
        if (bl.c.z <= 0.0) continue;
        SmokeBulletState st = smokeBulletState(bl, time, loopStart, loopDur);
        if (!st.live || st.traveled <= 0.0) continue;
        vec3 D = bl.b.xyz / max(length(bl.b.xyz), 1e-6);
        float sHead = dot(p - bl.a.xyz, D);
        float sC = clamp(sHead, 0.0, st.traveled);
        float bd = length(p - (bl.a.xyz + D * sC)) - max(bl.a.w * 0.5, 0.25);
        s.bullet = min(s.bullet, bd);
    }
    SmokeBulletFX fx = smokeBulletFX(p, time, loopStart, loopDur, t);
    s.tunnel = fx.thin;
    s.wave = fx.wave;
    s.wake = fx.wake;
    s.turb = length(fx.displace);
    s.pressure = fx.wave; // wave carries the pressure oscillation (debug)
    // 2. Displaced noise-domain density (base layered field).
    float base = smokeBaseDensity(p + fx.displace, center, rNow, maxR, t, time, seed, densityMul);
    s.density = base;
    // 3. Tunnel thinning + wake modulation + shock wave.
    float dens = base;
    dens *= 1.0 - clamp(fx.thin, 0.0, 0.95);
    dens *= 1.0 + clamp(fx.wave, -1.0, 1.0) * 0.15;
    // 4. Loop-end dissipation fade (seamless 10 s repeat).
    float loopT = smokeLoopT(time, t.timing.y);
    dens *= smokeEndFade(loopT, t.timing.y);
    s.finalD = dens;
    return s;
}

// Cheap density for light-march samples (§16): base field only, no bullet
// displacement/wake noise (the tunnel CORE is still honored so holes let
// light through).
float smokeShadowDensity(vec3 p, vec3 center, float rNow, float maxR,
                         SmokeGPU t, float time, float loopStart, float loopDur,
                         float seed, float densityMul) {
    float dens = smokeBaseDensity(p, center, rNow, maxR, t, time, seed, densityMul);
    for (int i = 0; i < 8; ++i) {
        BulletGPU bl = smokeBullets[i];
        if (bl.c.z <= 0.0) continue;
        SmokeBulletState st = smokeBulletState(bl, time, loopStart, loopDur);
        if (!st.live || st.traveled <= 0.0) continue;
        dens *= 1.0 - smokeTunnelCore(p, bl, st, t) * clamp(t.tunnel.x, 0.0, 1.0);
    }
    float loopT = smokeLoopT(time, t.timing.y);
    return dens * smokeEndFade(loopT, t.timing.y);
}

// Sun scattering for smoke (§15/§17): gray ramp by lighting (dark gray ->
// gray -> warm gray), fixed-correlation HG-ish forward boost, cheap
// self-shadow march toward the sun (§16).
vec3 smokeShade(vec3 p, vec3 viewDir, vec3 center, float rNow, float maxR,
                SmokeGPU t, SdfMaterialGPU mat, float time, float loopStart, float loopDur,
                float seed, float densityMul, vec3 sunDirW, vec3 sunColor, out float outTrans) {
    int shadowSteps = int(clamp(t.render.x, 1.0, 8.0));
    float shadowLen = max(rNow * 1.2, 1.0);
    float ldt = shadowLen / float(shadowSteps);
    float trans = 1.0;
    vec3 lp = p + sunDirW * ldt * 0.5;
    for (int j = 0; j < 8; ++j) {
        if (j >= shadowSteps) break;
        trans *= exp(-smokeShadowDensity(lp, center, rNow, maxR, t, time,
                                         loopStart, loopDur, seed, densityMul)
                     * ldt * max(mat.volumeParams.y, 0.0));
        lp += sunDirW * ldt;
    }
    trans = mix(1.0, trans, clamp(t.render.y, 0.0, 1.0));
    float cosT = dot(viewDir, sunDirW);
    float phase = 0.5 + 0.5 * pow(clamp(cosT * 0.5 + 0.5, 0.0, 1.0), 2.0);
    float lightLevel = clamp(trans * (0.35 + 0.65 * phase), 0.0, 1.0);
    vec3 albedo = mix(vec3(0.30), vec3(0.72, 0.69, 0.65), lightLevel);
    vec3 ambientCol = vec3(0.35, 0.38, 0.42) * 0.5;
    outTrans = trans;
    return albedo * (sunColor * phase * 2.0 * trans + ambientCol);
}

// Marched field for Smoke instances (§4 + §14): grown sphere with the
// bullet-tunnel core carved (fresh tunnels step through; slight overstep
// of residual smoke is the documented tradeoff for the explicit
// subtraction the spec requires).
float smokeMarchSDF(vec3 wpos, SdfInstanceGPU inst, SdfDefinitionGPU def, float time) {
    float ds;
    vec3 q = sdfWorldToLocal(wpos, inst, ds);
    float maxR = max(def.params0.x, 1.0);
    float loopDur = max(smokeTuning.timing.y, 1.0);
    float loopT = smokeLoopT(time, loopDur);
    float loopStart = time - loopT;
    float rNow = smokeGrowthRadius(maxR, loopT, max(smokeTuning.timing.x, 0.5));
    float d = length(q) - rNow / max(ds, 1e-4);
    for (int i = 0; i < 8; ++i) {
        BulletGPU bl = smokeBullets[i];
        if (bl.c.z <= 0.0) continue;
        SmokeBulletState st = smokeBulletState(bl, time, loopStart, loopDur);
        if (!st.live || st.traveled <= 0.0) continue;
        vec3 D = bl.b.xyz / max(length(bl.b.xyz), 1e-6);
        float sC = clamp(dot(wpos - bl.a.xyz, D), 0.0, st.traveled);
        vec3 core = bl.a.xyz + D * sC;
        float bd = (length(wpos - core) - max(bl.a.w * 0.35, 0.25)) / max(ds, 1e-4);
        d = max(d, -bd);
    }
    return d * ds;
}

#endif // SDF_SMOKE_GLSL
