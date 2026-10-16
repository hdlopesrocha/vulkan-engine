#version 450

#include "../../ubo/UniformObject.glsl"

#include "../../includes/Locations.glsl"
#include "../../includes/vegetation/NormalTilt.glsl"

layout(location = VARY_UV) in vec3 inTexCoord;  // xy=UV, z=float(layerIdx)
// xyz = instance base, w = runtime height scale (hs); see the depth VS.
layout(location = VARY_POSLIGHT) flat in vec4 inInstanceOffset;
// Instance rotation + terrain surface normal: the reconstruction applies the
// same rotateY(hs * p) + tilt-to-surface transform the blade VS bakes in.
layout(location = VARY_ROTFRAC) flat in vec2 inCosSinT;
layout(location = VARY_PLANE_NORMAL) flat in vec3 inSurfN;
// Interface match: the shared depth vertex stage also writes VARY_POSWORLD
// (read by the impostor-shadow pipeline using this same VS); unread here.

// Canonical scene UBO (set=0 binding=0): the shared struct IS the block
// layout — no packed/named-view split.
layout(std140, set = 0, binding = 0) uniform SolidParamsBlock {
    UniformObject ubo;
};

layout(set = 1, binding = 0) uniform sampler2DArray depthArray;

layout(set = 1, binding = 1) readonly buffer CaptureInvVP {
    mat4 invVP[];
};

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
    int layer = int(inTexCoord.z);

    float texDepth = texture(depthArray, inTexCoord).r;
    if (texDepth >= 1.0 || texDepth <= 0.0) discard;

    // Direct hand-off (see impostors_depth.vert): no dithered fade; the
    // billboard depth prepass stops at the same distance.

    // Reconstruct world position from captured depth, applying the exact
    // runtime instance transform (base + tilt_surfN(rotateY(hs * cap))), so
    // the prepass depth matches both the rasterized quad and the color pass
    // bit for bit.
    vec2 ndc_xy = inTexCoord.xy * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc_xy, texDepth, 1.0);
    vec4 capPos = invVP[layer] * clipPos;
    capPos /= capPos.w;

    vec3 worldPos = inInstanceOffset.xyz + grassTiltToNormal(
        rotateY(capPos.xyz * inInstanceOffset.w, inCosSinT.x, inCosSinT.y),
        normalize(inSurfN));

    // Reproject to camera space.
    vec4 camClipPos = ubo.viewProjection * vec4(worldPos, 1.0);
    gl_FragDepth = clamp(camClipPos.z / camClipPos.w, 0.0, 1.0);
}
