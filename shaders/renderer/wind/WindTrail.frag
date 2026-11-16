#version 450

// Wind-trail fragment stage (WindRenderer).
//
// Trails are colored by the local wind SPEED using a 4-stop ramp
// (blue = slow -> green -> yellow -> red = fast), so the visualization shows
// the field's speed structure. The per-endpoint speed comes from the
// simulation's trail samples (WindTrail.vert) and is interpolated along each
// segment. Blended with straight alpha over the composited frame; depth
// test/write stay disabled so the trails read as a visualization overlay.

#include "../../ubo/WindParticlesUBO.glsl"

layout(set = 0, binding = 0, std140) uniform WindParticlesBlock {
    WindParticlesUBO wp;
};

layout(location = 0) in float vLateral;
layout(location = 1) in float vFade;
layout(location = 2) in float vSpeed;

layout(location = 0) out vec4 outColor;

// Blue (0) -> green (1/3) -> yellow (2/3) -> red (1).
vec3 windSpeedColor(float t) {
    const vec3 kBlue   = vec3(0.10, 0.25, 1.00);
    const vec3 kGreen  = vec3(0.10, 0.90, 0.25);
    const vec3 kYellow = vec3(1.00, 0.90, 0.10);
    const vec3 kRed    = vec3(1.00, 0.12, 0.08);
    t = clamp(t, 0.0, 1.0);
    if (t < 1.0 / 3.0) return mix(kBlue, kGreen, t * 3.0);
    if (t < 2.0 / 3.0) return mix(kGreen, kYellow, (t - 1.0 / 3.0) * 3.0);
    return mix(kYellow, kRed, (t - 2.0 / 3.0) * 3.0);
}

void main() {
    // Soft round edge across the trail width; vLateral is -1..+1.
    float edge = 1.0 - clamp(vLateral * vLateral, 0.0, 1.0);
    float alpha = vFade * edge * clamp(wp.style.y, 0.0, 1.0);
    if (alpha <= 0.002) discard;
    // 0 m/s = blue, style.x m/s (and above) = red.
    vec3 color = windSpeedColor(vSpeed / max(wp.style.x, 1e-4));
    outColor = vec4(color, alpha);
}
