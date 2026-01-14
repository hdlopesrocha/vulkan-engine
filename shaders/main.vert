#version 450

#include "includes/ubo.glsl"


layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inNormal;
// tangent removed: computed in fragment shader for triplanar mapping
layout(location = 5) in int inTexIndex;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal;
layout(location = 5) flat out int fragTexIndex;
layout(location = 4) out vec3 fragPosWorld;
layout(location = 6) out vec4 fragPosLightSpace;
layout(location = 7) out vec3 fragLocalPos;
layout(location = 8) out vec3 fragLocalNormal;
// fragTangent removed: computed in fragment shader for triplanar mapping

void main() {
    fragColor = inColor;
    fragUV = inUV;
    // Transform normal to world space using the model matrix from push constants
    // For uniform scaling, mat3(model) works. For non-uniform scaling, use transpose(inverse(model))
    fragNormal = normalize(inNormal);
    // Provide per-vertex tex index for TCS to assemble per-patch indices
    fragTexIndex = inTexIndex;
    // compute world-space position and pass to fragment
    vec4 worldPos = vec4(inPos, 1.0);
    fragPosWorld = worldPos.xyz;
    // compute light-space position for shadow mapping
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;
    // pass local-space position (used by tessellation/displacement)
    fragLocalPos = inPos;
    // also pass local-space normal for tessellation/displacement
    fragLocalNormal = inNormal;
    // apply MVP transform to the vertex position
    gl_Position = ubo.viewProjection * worldPos;
}