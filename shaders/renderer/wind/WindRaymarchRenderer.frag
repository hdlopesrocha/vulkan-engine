#version 450

// WindRaymarchRenderer: renders a given wind velocity as a shape.
//
// The shape is the isosurface |windSVF(p, time)| == targetSpeed of the shared
// 3D fBm wind field (set=0 binding 27, the exact field the vegetation, fire,
// clouds and wind trails use). A fullscreen pass marches the field along each
// view ray, clipped by the scene depth (foreground geometry occludes the
// volume), finds the first sign change of d(p) = target - |wind|, refines it
// with bisection, and shades the hit with a gradient normal + the local flow
// direction.
//
// The target speed (ubo.surface.x) is the primary adjustable parameter: raise
// it to isolate strong-gust/tornado cores, lower it to reveal the broader
// flow structure. March distance/steps/opacity are also adjustable.
//
// Requires Fullscreen.vert (no vertex input, fullscreen triangle).

#include "../../includes/noise/Perlin.glsl"
#include "../../includes/vegetation/WindField.glsl"
#include "../../ubo/WindRaymarchUBO.glsl"

layout(set = 0, binding = 0, std140) uniform WindRaymarchBlock {
    WindRaymarchUBO wr;
};
// Scene (solid pass) depth, SHADER_READ_ONLY. Used only to clip the march.
layout(set = 0, binding = 1) uniform sampler2D sceneDepthTex;
// Binding 27 (shared wind field) is declared by WindField.glsl.

layout(location = 0) out vec4 outColor;

// Field magnitude at a point, NaN-guarded (a poisoned sample must not create a
// phantom surface or blank the overlay).
float windSpeedAt(vec3 p, float t) {
    float s = length(windSVF(p, t));
    return isnan(s) ? 1e5 : s;
}

void main() {
    vec2 uv = gl_FragCoord.xy / max(wr.viewport.xy, vec2(1.0));
    float sceneDepth = texture(sceneDepthTex, uv).r;

    // World position of the scene surface under this pixel (or the far plane).
    vec4 worldH = wr.invViewProjection * vec4(uv * 2.0 - 1.0, sceneDepth, 1.0);
    vec3 world = worldH.xyz / (abs(worldH.w) > 1e-6 ? worldH.w : 1e-6);
    vec3 camPos = wr.cameraPosition.xyz;
    vec3 rayVec = world - camPos;
    float sceneDist = length(rayVec);
    if (!(sceneDist > 1e-4)) discard;
    vec3 rayDir = rayVec / sceneDist;

    float maxT = min(max(wr.surface.y, 0.0), sceneDist);
    if (!(maxT > 1e-3)) discard;

    float target = max(wr.surface.x, 1e-3);
    float time = wr.viewport.z;
    int steps = int(clamp(wr.surface.z, 8.0, 128.0));

    // Isosurface: d(p) = target - |wind(p)|. A sign change in either direction
    // marks a surface crossing (handles the camera starting inside the shape,
    // where the first crossing is an exit surface).
    float prevT = 0.0;
    float prevD = target - windSpeedAt(camPos, time);
    bool hit = false;
    vec3 hitP = camPos;
    for (int i = 1; i <= 128; ++i) {
        if (i > steps) break;
        float t = maxT * float(i) / float(steps);
        vec3 p = camPos + rayDir * t;
        float d = target - windSpeedAt(p, time);
        if (prevD * d < 0.0) {
            // Bisection refine on the bracket: 8 halvings (~0.4% of a step).
            float ta = prevT;
            float tb = t;
            for (int b = 0; b < 8; ++b) {
                float tm = 0.5 * (ta + tb);
                float dm = target - windSpeedAt(camPos + rayDir * tm, time);
                if (prevD * dm > 0.0) {
                    ta = tm;
                } else {
                    tb = tm;
                }
            }
            hitP = camPos + rayDir * (0.5 * (ta + tb));
            hit = true;
            break;
        }
        prevT = t;
        prevD = d;
    }
    if (!hit) discard;

    // Gradient normal of the field magnitude (tetrahedral 4-tap stencil; the
    // field is smooth at the ~500 m feature scale, so 8 m offsets give a
    // stable normal). Degenerate gradients fall back to the view ray.
    const float e = 8.0;
    const vec2 k = vec2(1.0, -1.0);
    float ga = windSpeedAt(hitP + vec3(k.x, k.y, k.y) * e, time);
    float gb = windSpeedAt(hitP + vec3(k.y, k.y, k.x) * e, time);
    float gc = windSpeedAt(hitP + vec3(k.y, k.x, k.y) * e, time);
    float gd = windSpeedAt(hitP + vec3(k.x, k.x, k.x) * e, time);
    vec3 grad = vec3(k.x, k.y, k.y) * ga + vec3(k.y, k.y, k.x) * gb +
                vec3(k.y, k.x, k.y) * gc + vec3(k.x, k.x, k.x) * gd;
    float gradLen2 = dot(grad, grad);
    if (isnan(gradLen2) || gradLen2 < 1e-12) grad = -rayDir;
    vec3 normal = normalize(grad);

    // Flow direction at the surface, as a color, plus a wrapped key light and
    // a bright rim so the isosurface reads as a solid shape against both sky
    // and terrain.
    vec3 vHit = windSVF(hitP, time);
    float speed = length(vHit);
    vec3 flowTint = (isnan(speed) || speed <= 1e-4) ? vec3(0.5)
                                                    : (vHit / speed) * 0.5 + 0.5;
    vec3 keyDir = normalize(vec3(0.35, 0.85, 0.25));
    float wrap = clamp(dot(normal, keyDir) * 0.5 + 0.5, 0.0, 1.0);
    float rim = pow(1.0 - clamp(dot(normal, -rayDir), 0.0, 1.0), 2.0);
    vec3 color = flowTint * (0.20 + 0.80 * wrap);
    color = mix(color, vec3(0.95, 0.98, 1.0), rim * 0.65);

    // Fade the surface toward the clip distance so the shape dissolves instead
    // of ending in a hard cut where the scene depth or the range limit sits.
    float tHit = length(hitP - camPos);
    float alpha = clamp(wr.surface.w, 0.0, 1.0);
    alpha *= 1.0 - smoothstep(0.85 * maxT, maxT, tHit);
    if (alpha <= 0.002) discard;
    outColor = vec4(color, alpha);
}
