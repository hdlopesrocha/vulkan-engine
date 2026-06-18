#version 450

#include "includes/locations.glsl"

layout(location = VARY_SDF) in float fragSdf;
layout(location = VARY_BRUSHPATCH) flat in int fragBrushIndex;

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

void main() {
    float rng = 10.0;

    if (abs(fragSdf) > rng) {
        discard;
    }

    float normalizedMagnitude = clamp(abs(fragSdf) / rng, 0.0, 1.0);
    vec3 negativeColor = vec3(0.0, 0.0, 0.0);

    // Palette copied from main.frag debug palette (16 colors)
    const int PALETTE_SIZE = 16;
    const vec3 palette[PALETTE_SIZE] = vec3[](
        vec3(0.90, 0.10, 0.10), // red
        vec3(0.10, 0.90, 0.10), // green
        vec3(0.10, 0.10, 0.90), // blue
        vec3(0.90, 0.90, 0.10), // yellow
        vec3(0.90, 0.10, 0.90), // magenta
        vec3(0.10, 0.90, 0.90), // cyan
        vec3(1.00, 0.55, 0.10), // orange
        vec3(0.55, 0.35, 0.15), // brown
        vec3(0.60, 0.20, 0.80), // purple
        vec3(1.00, 0.50, 0.70), // pink
        vec3(0.70, 1.00, 0.30), // lime
        vec3(0.00, 0.45, 0.55), // teal
        vec3(0.05, 0.10, 0.35), // navy
        vec3(0.45, 0.50, 0.10), // olive
        vec3(0.60, 0.60, 0.60), // gray
        vec3(1.00, 1.00, 1.00)  // white
    );

    vec3 positiveColor;
    if (fragBrushIndex >= 0) {
        int idx = int(mod(float(fragBrushIndex), float(PALETTE_SIZE)) + 0.5);
        positiveColor = palette[idx];
    } else {
        // fallback when no brush index is available
        positiveColor = vec3(0.0, 1.0, 0.0);
    }
    vec3 color = fragSdf >= 0.0 ? positiveColor : negativeColor;

    float surfaceWidth = max(fwidth(fragSdf) * 1.5, 0.015);
    float surfaceLine = 1.0 - smoothstep(0.0, surfaceWidth, abs(fragSdf));

    color *= 0.35 + 0.65 * (1.0 - normalizedMagnitude);
    color = mix(color, vec3(1.0), surfaceLine);
    outColor = vec4(color, 1.0);
}
