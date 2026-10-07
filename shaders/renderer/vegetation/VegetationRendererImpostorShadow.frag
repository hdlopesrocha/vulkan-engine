#version 450

#include "../../ubo/UniformObject.glsl"

// Impostor EVSM2 shadow pass: uses vertex position from vertex shader
// to write EVSM moments.

#include "../../includes/Locations.glsl"
#include "../../includes/shadow/EvsmWrite.glsl"

layout(location = VARY_POSWORLD) in vec3 inWorldPos;
layout(location = VARY_TANGENTWS) flat in vec3 inInstanceOffset;

// Canonical scene UBO (set=0 binding=0): the shared struct IS the block
// layout — no packed/named-view split.
layout(std140, set = 0, binding = 0) uniform SolidParamsBlock {
    UniformObject ubo;
};

layout(location = FRAG_OUT_COLOR) out vec2 outEVSM;

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
    // Direct hand-off at impostorDistance (see impostors_depth.vert): no
    // dithered shadow cross-fade.

    vec4 lsPos = ubo.viewProjection * vec4(inWorldPos, 1.0);
    float depth = clamp(lsPos.z / lsPos.w, 0.0, 1.0);

    gl_FragDepth = depth;

    outEVSM = evsmMoments(depth);
}
