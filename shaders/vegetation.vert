#version 450

#include "includes/locations.glsl"

layout(location = ATTR_POS) in vec3 inPosition;
layout(location = ATTR_NORMAL) in vec3 inNormal;
layout(location = ATTR_UV) in vec2 inTexCoord;
layout(location = ATTR_BRUSH_INDEX) in int inBrushIndex;
layout(location = ATTR_INSTANCE) in vec4 instanceData; // .xyz = world pos, .w = billboard index

layout(location = VARY_UV) out vec3 fragTexCoord;
layout(location = VARY_BRUSHPATCH) flat out int fragBrushIndex;
layout(location = VARY_POSWORLD) out vec3 fragWorldPos;

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
    // inPosition is always (0,0,0) for the base vertex; world position = instanceData.xyz.
    vec3 worldPos = instanceData.xyz;
    gl_Position = ubo.viewProjection * vec4(worldPos, 1.0);
    // Pass layer index as z component of texcoord; xy will be set in geometry shader
    // instanceData.w = float(billboardIndex) + rotFrac, where:
    //   floor(w) = billboard index
    //   fract(w) = Y-axis rotation fraction in [0,1) → angle = fract * 2*PI in geometry shader
    fragTexCoord = vec3(0.0, 0.0, instanceData.w);
    fragBrushIndex = int(floor(instanceData.w)); // billboard index (strip rotation fraction)
    fragWorldPos = worldPos;
}
