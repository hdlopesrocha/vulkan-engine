#version 450

// Vegetation EVSM shadow fragment shader with alpha rejection.
// Samples the opacity texture and discards transparent fragments
// so the shadow silhouette respects leaf cutouts.

#include "includes/locations.glsl"
#include "includes/ubo.glsl"
#include "includes/evsm_write.glsl"

layout(location = VARY_UV) in vec3 inTexCoord;
layout(location = VARY_POSWORLD) in vec3 inWorldPos;

layout(location = FRAG_OUT_COLOR) out vec4 outEVSM;

layout(set = 1, binding = 2) uniform sampler2DArray opacityArray;

void main() {
    vec3 coord = vec3(inTexCoord.xy, inTexCoord.z);

    float opacity = texture(opacityArray, coord).r;
    float opacityWeight = smoothstep(0.35, 0.65, opacity);
    if (opacityWeight < 0.3) discard;

    vec4 lsPos = ubo.viewProjection * vec4(inWorldPos, 1.0);
    float depth = clamp(lsPos.z / lsPos.w, 0.0, 1.0);

    outEVSM = evsmMoments(depth);
}
