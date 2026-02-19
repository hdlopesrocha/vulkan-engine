#version 450
#extension GL_ARB_shader_draw_parameters : require

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
layout(location = 4) out vec3 fragPosWorld;
layout(location = 5) flat out int fragTexIndex;      // per-vertex texture index for TCS
layout(location = 6) out vec4 fragPosLightSpace;
layout(location = 7) out vec3 fragLocalPos;          // provide local/world pos to TCS
layout(location = 8) out vec3 fragLocalNormal;       // provide local/world normal to TCS
layout(location = 10) out vec3 fragSharpNormal;      // face normal

void main() {
    fragColor = inColor;
    fragUV = inUV;
    // Models removed: always use identity model matrix
    mat4 model = mat4(1.0);
    // Transform normal to world space (model is identity here)
    // For uniform scaling, mat3(model) works. For non-uniform scaling, use transpose(inverse(model))
    fragNormal = normalize(mat3(model) * inNormal);
    
    // Pass per-vertex texture index as flat int for patch compression in TCS
    fragTexIndex = inTexIndex;
    
    // compute world-space position and pass to fragment
    vec4 worldPos = model * vec4(inPos, 1.0);
    fragPosWorld = worldPos.xyz;
    fragLocalPos = worldPos.xyz;       // Use world-space position as local basis for displacement
    
    // compute light-space position for shadow mapping
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;
    
    // Compute face normal (sharp normal) from model normal
    fragLocalNormal = fragNormal;      // Propagate normal for tessellation stages
    fragSharpNormal = normalize(mat3(model) * inNormal);
    
    // apply MVP transform to the vertex position
    gl_Position = ubo.viewProjection * worldPos;
}