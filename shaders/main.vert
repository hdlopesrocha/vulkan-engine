#version 450

#include "includes/ubo.glsl"


layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec4 inTangent;
layout(location = 5) in int inTexIndex;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal;
layout(location = 5) flat out int fragTexIndex;
layout(location = 4) out vec3 fragPosWorld;
layout(location = 6) out vec4 fragPosLightSpace;
layout(location = 7) out vec3 fragLocalPos;
layout(location = 8) out vec3 fragLocalNormal;
layout(location = 9) out vec4 fragTangent;

void main() {
    fragColor = inColor;
    fragUV = inUV;
    // Transform normal to world space using the model matrix
    // For uniform scaling, mat3(model) works. For non-uniform scaling, use transpose(inverse(model))
    fragNormal = normalize(mat3(ubo.model) * inNormal);
    // Provide per-vertex tex index for TCS to assemble per-patch indices
    fragTexIndex = inTexIndex;
    // compute world-space position and pass to fragment
    vec4 worldPos = ubo.model * vec4(inPos, 1.0);
    fragPosWorld = worldPos.xyz;
    // compute light-space position for shadow mapping
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;
    // pass local-space position (used by tessellation/displacement)
    fragLocalPos = inPos;
    // also pass local-space normal for tessellation/displacement
    fragLocalNormal = inNormal;
    // pass tangent as a vec4: xyz = tangent, w = handedness sign
    fragTangent = vec4(normalize(mat3(ubo.model) * inTangent.xyz), inTangent.w);
    // apply MVP transform to the vertex position (MVP already includes model transform)
    gl_Position = ubo.mvp * vec4(inPos, 1.0);
}