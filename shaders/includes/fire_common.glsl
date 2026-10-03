// Shared helpers for animated fire billboards.
// Requires perlin.glsl (fbm) to be included BEFORE this file.
// Requires the push-constant block with `windTime` (vegetation family) to be
// declared before this include (used as the animation clock).

#ifndef FIRE_COMMON_GLSL
#define FIRE_COMMON_GLSL

// Must match FireParamsUBO (vulkan/ubo/FireUBO.hpp): 7 x vec4 = 112 bytes.
layout(set = 2, binding = 1) uniform FireParamsUBO {
    vec4 fireEnabledSizeSpeedIntensity; // x=enabled, y=size, z=speed, w=intensity
    vec4 fireShape;                     // x=flicker, y=noiseScale, z=heightScale, w=turbulence
    vec4 fireMotion;                    // x=riseSpeed, y=windInfluence, z=alpha, w=emissive
    vec4 fireInnerColor;                // rgb = hot core
    vec4 fireMidColor;                  // rgb = mid flame
    vec4 fireOuterColor;                // rgb = flame edge
    vec4 fireExtra;                     // x=smoke, yzw spare
} firePacked;

// Named view over the packed FireParamsUBO. The builder below is the only
// place the packed component letters are read; every other access uses the
// named attributes.
struct FireParamsNamed {
    bool enabled;
    float size;
    float speed;
    float intensity;
    float flicker;
    float noiseScale;
    float heightScale;
    float turbulence;
    float riseSpeed;
    float windInfluence;
    float alpha;
    float emissive;
    vec3 innerColor;
    vec3 midColor;
    vec3 outerColor;
    float smoke;
};

FireParamsNamed fireParamsNamed() {
    FireParamsNamed n;
    n.enabled       = firePacked.fireEnabledSizeSpeedIntensity.x > 0.5;
    n.size          = firePacked.fireEnabledSizeSpeedIntensity.y;
    n.speed         = firePacked.fireEnabledSizeSpeedIntensity.z;
    n.intensity     = firePacked.fireEnabledSizeSpeedIntensity.w;
    n.flicker       = firePacked.fireShape.x;
    n.noiseScale    = firePacked.fireShape.y;
    n.heightScale   = firePacked.fireShape.z;
    n.turbulence    = firePacked.fireShape.w;
    n.riseSpeed     = firePacked.fireMotion.x;
    n.windInfluence = firePacked.fireMotion.y;
    n.alpha         = firePacked.fireMotion.z;
    n.emissive      = firePacked.fireMotion.w;
    n.innerColor    = firePacked.fireInnerColor.rgb;
    n.midColor      = firePacked.fireMidColor.rgb;
    n.outerColor    = firePacked.fireOuterColor.rgb;
    n.smoke         = firePacked.fireExtra.x;
    return n;
}

FireParamsNamed fireParams = fireParamsNamed();

// FIRE_BILLBOARD_INDEX comes from locations.glsl (must be included first).

// Flame coverage in [0,1] for a fire-quad texel.
// uv: corner UV (x in [0,1] across the quad, y with 1 at the base, 0 at tip).
// seed: per-instance random in [0,1]. time: animation clock (windTime).
float fireFlameCoverage(vec2 uv, float seed, float time) {
    FireParamsNamed fp = fireParams;
    float h = clamp(1.0 - uv.y, 0.0, 1.0); // 0 at base, 1 at tip

    float t = time * max(fp.speed, 0.0);
    float ns = max(fp.noiseScale, 0.1);

    // Turbulent body noise scrolling upward (flames rise).
    vec3 np = vec3(uv.x * ns + seed * 13.7,
                   h * ns * 1.4 - t * max(fp.riseSpeed, 0.0) * 0.55,
                   seed * 7.0 + t * 0.23);
    float n = fbm(np, 3, 0.5, 2.0) * 0.5 + 0.5;

    // Sideways sway grows with height; per-instance phase so neighbours desync.
    float sway = (n - 0.5) * (0.25 + h * 0.9) * (0.35 + fp.turbulence)
               + sin(t * 2.2 + seed * 43.0 + h * 5.0) * 0.06 * (0.5 + fp.turbulence);

    // Teardrop silhouette: wide at the base, pinched at the tip.
    float halfW = mix(0.42, 0.03, pow(h, 0.65));
    float d = abs(uv.x - 0.5 - sway) / max(halfW, 1e-3);

    // Eroded body: noise eats the edges, more so toward the tip.
    float body = (1.0 - h * 0.85) * (0.55 + 0.45 * n) - d * 0.75;
    float cover = clamp(body * 1.6, 0.0, 1.0);
    // Tip fade with ragged edge.
    cover *= smoothstep(1.0, 0.55, h + (n - 0.5) * 0.35 * (0.5 + fp.turbulence));
    // Base fade so the quad bottom edge never shows a hard line.
    cover *= smoothstep(0.0, 0.06, h + 0.03);
    return clamp(cover, 0.0, 1.0);
}

// Full flame shading: rgb = HDR emissive color, a = coverage * global alpha.
// Unlit on purpose — fire is a light source, not a lit surface.
vec4 evaluateFire(vec2 uv, float seed, float time) {
    FireParamsNamed fp = fireParams;
    float h = clamp(1.0 - uv.y, 0.0, 1.0);
    float t = time * max(fp.speed, 0.0);

    float cover = fireFlameCoverage(uv, seed, time);

    vec3 np2 = vec3(uv.x * 3.0 + seed * 29.0, h * 3.0 - t * 0.8, seed * 11.0);
    float n2 = fbm(np2, 2, 0.5, 2.0) * 0.5 + 0.5;

    // Hot core near the base.
    float core = clamp(cover * 1.5 - h * 0.9 - (1.0 - n2) * 0.45, 0.0, 1.0);

    // Global flicker: slow breathing + fast shimmer.
    float flick = 1.0
        - fp.flicker * 0.30 * (0.5 + 0.5 * sin(t * (5.0 + fp.speed * 2.0) + seed * 39.0))
        - fp.flicker * 0.25 * (n2 - 0.5);
    flick = max(flick, 0.25);

    vec3 col = mix(fp.outerColor, fp.midColor, clamp(cover * 1.6, 0.0, 1.0));
    col = mix(col, fp.innerColor, clamp(core * 1.6, 0.0, 1.0));
    col *= max(fp.intensity, 0.0) * max(fp.emissive, 0.0) * flick;

    // Thin smoke veil above the flame.
    float smokeA = fp.smoke * smoothstep(0.45, 1.0, h) * (1.0 - cover) * n2;
    col = mix(col, vec3(0.12, 0.11, 0.11) * max(fp.intensity, 0.0), clamp(smokeA * 1.5, 0.0, 0.6));

    float alpha = clamp(cover * 1.25 + smokeA * 0.4, 0.0, 1.0) * clamp(fp.alpha, 0.0, 1.0);
    return vec4(col, alpha);
}

#endif // FIRE_COMMON_GLSL
