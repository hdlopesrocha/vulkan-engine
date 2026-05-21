#version 450

#include "includes/locations.glsl"

layout(location = ATTR_POS) in vec3 inPosition;
layout(location = ATTR_NORMAL) in vec3 inNormal;
layout(location = ATTR_UV) in vec2 inTexCoord;
layout(location = ATTR_BRUSH_INDEX) in int  inBrushIndex;
layout(location = ATTR_INSTANCE) in vec4 instanceData; // xyz=world pos, w=billboard index

layout(location = VARY_POSWORLD) out vec3 outWorldPos;
layout(location = VARY_BRUSHPATCH) flat out int outBillboardIndex;
layout(location = VARY_ROTFRAC) out float outRotFrac;  // per-instance Y-rotation fraction [0,1)

// Must match SolidParamsUBO — only read the first two fields.
layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} ubo;

layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float impostorDistance;
    vec4 windDirAndStrength;
    vec4 windNoise;
    vec4 windShape;
    vec4 windTurbulence;
    vec4 densityParams;
    vec4 cameraPosAndFalloff;
};

void main() {
    vec3 worldPos = instanceData.xyz;
    // Pass world position as clip coords placeholder; geo shader does the real projection.
    gl_Position = ubo.viewProjection * vec4(worldPos, 1.0);
    outWorldPos       = worldPos;
    // instanceData.w = float(billboardIndex) + rotFrac; strip rotFrac with floor().
    outBillboardIndex = int(floor(instanceData.w));
    outRotFrac        = fract(instanceData.w);
}
