#version 450

#include "includes/ubo.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;
layout(location = 5) in float inTexIndex;

layout(location = 0) out vec2 pc_inUV;
layout(location = 1) out vec3 pc_inNormal;
layout(location = 3) out vec3 pc_inPosWorld;
layout(location = 4) out vec4 pc_inTangent;
layout(location = 5) out float pc_inTexIndex;
layout(location = 7) out vec3 pc_inLocalPos;
layout(location = 8) out vec3 pc_inLocalNormal;
layout(location = 9) out vec4 pc_inLocalTangent;

void main() {
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    pc_inPosWorld = worldPos.xyz;
    pc_inUV = inTexCoord;
    pc_inNormal = mat3(ubo.model) * inNormal;
    pc_inTangent = vec4(mat3(ubo.model) * inTangent.xyz, inTangent.w);
    pc_inTexIndex = inTexIndex;
    pc_inLocalPos = inPosition;
    pc_inLocalNormal = inNormal;
    pc_inLocalTangent = inTangent;
    
    // MVP already includes model transform, apply to local position
    gl_Position = ubo.mvp * vec4(inPosition, 1.0);
}
