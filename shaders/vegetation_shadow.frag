#version 450

// Vegetation EVSM shadow fragment shader.
// Writes EVSM moments from the vertex world position.
// Alpha rejection is intentionally omitted: the 6 crossed billboard
// planes are too thin to produce a visible shadow with leaf cutouts.
// Keeping the solid silhouette ensures the shadow is visible.

#include "includes/locations.glsl"
#include "includes/ubo.glsl"
#include "includes/evsm_write.glsl"

layout(location = VARY_POSWORLD) in vec3 inWorldPos;

layout(location = FRAG_OUT_COLOR) out vec4 outEVSM;

void main() {
    vec4 lsPos = ubo.viewProjection * vec4(inWorldPos, 1.0);
    float depth = clamp(lsPos.z / lsPos.w, 0.0, 1.0);

    outEVSM = evsmMoments(depth);
}
