#version 450

#include "includes/locations.glsl"

layout(location = VARY_UV) in vec3 inTexCoord;  // xy=UV, z=float(layerIdx)
layout(location = VARY_TANGENTWS) flat in vec3 inInstanceOffset;

layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection; // light VP
    vec4 viewPos;
} ubo;

layout(set = 1, binding = 0) uniform sampler2DArray depthArray;

layout(set = 1, binding = 1) readonly buffer CaptureInvVP {
    mat4 invVP[];
};

void main() {
    int layer = int(inTexCoord.z);

    float texDepth = texture(depthArray, inTexCoord).r;
    if (texDepth >= 1.0 || texDepth <= 0.0) discard;

    // Reconstruct world position from captured depth.
    // ndc_xy: [0,1] → [-1,1]; ndc_z: Vulkan [0,1] (no remap needed).
    vec2 ndc_xy = inTexCoord.xy * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc_xy, texDepth, 1.0);
    vec4 worldPos = invVP[layer] * clipPos;
    worldPos /= worldPos.w;

    // Translate from capture origin to instance position.
    worldPos.xyz += inInstanceOffset;

    // Reproject to light space — already in Vulkan [0,1] NDC.
    vec4 lightClipPos = ubo.viewProjection * worldPos;
    gl_FragDepth = clamp(lightClipPos.z / lightClipPos.w, 0.0, 1.0);
}
