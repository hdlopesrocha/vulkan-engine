#version 450

// Pass-through vertex shader for impostor capture.
// Reads instance data and forwards to the geometry shader (vegetation.geom)
// which expands each point into a full billboard.

#include "includes/locations.glsl"

layout(location = ATTR_POS) in vec3 inPosition;
layout(location = ATTR_COLOR) in vec3 inColor;       // unused
layout(location = ATTR_UV) in vec2 inTexCoord;        // unused
layout(location = ATTR_NORMAL) in vec3 inNormal;       // unused
layout(location = ATTR_BRUSH_INDEX) in int inBrushIndex; // unused
layout(location = ATTR_INSTANCE) in vec4 instanceData;  // xyz=worldPos, w=billboardIndex+rotFrac

layout(location = VARY_UV) out vec3 fragTexCoord;
layout(location = VARY_BRUSHPATCH) flat out int fragBrushIndex;
layout(location = VARY_POSWORLD) out vec3 fragWorldPos;

layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} ubo;

void main() {
    vec3 worldPos = instanceData.xyz;
    gl_Position = ubo.viewProjection * vec4(worldPos, 1.0);
    fragTexCoord    = vec3(0.0, 0.0, instanceData.w);
    fragBrushIndex  = int(floor(instanceData.w));
    fragWorldPos    = worldPos;
}
