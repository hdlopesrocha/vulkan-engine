#version 450

#include "../../includes/Locations.glsl"

// Bladed grass depth prepass: opaque triangles, no cutout.
// Declares the shared blade VS outputs (required to avoid SPIR-V interface
// warnings — the depth pipeline shares the vertex module with color) but
// performs no discard: every blade pixel writes depth.

layout(location = VARY_UV) in vec3 inTexCoord;
layout(location = VARY_BRUSHPATCH) flat in int inBrushIndex;
layout(location = VARY_POSWORLD) in vec3 inWorldPos;
layout(location = VARY_PLANE_NORMAL) flat in vec3 inPlaneNormal;
layout(location = VARY_POSLIGHT) flat in vec3 inTangentWS;
layout(location = VARY_TANGENTWS) flat in vec3 inLightDirN;

layout(set = 2, binding = 0) uniform WindParamsUBO {
    vec4 windDirAndStrength;
    vec4 windNoise;
    vec4 windShape;
    vec4 windTurbulence;
    vec4 densityParams;
    vec4 cameraPosAndFalloff;
} windParams;


layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float impostorDistance;
};

void main() {
    // Opaque: depth written by default for every blade fragment.
    // The impostor hand-off is a direct per-instance swap in the vertex
    // shader, so the prepass writes exactly what the shading pass keeps.
}
