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
layout(location = 5) flat out ivec3 fragTexIndices;  // Changed from int to ivec3
layout(location = 11) out vec3 fragTexWeights;       // Added
layout(location = 4) out vec3 fragPosWorld;
layout(location = 6) out vec4 fragPosLightSpace;
layout(location = 10) out vec3 fragSharpNormal;      // Added
layout(location = 7) out vec3 fragPosWorldNotDisplaced;  // Renamed from fragLocalPos
// layout(location = 8) out vec3 fragLocalNormal;    // Removed - not used by fragment shader

void main() {
    fragColor = inColor;
    fragUV = inUV;
    // Get model matrix from models SSBO
    // For indirect draws: use gl_BaseInstanceARB which gets firstInstance from the command
    mat4 model = models[gl_BaseInstanceARB];
    // Transform normal to world space using the model matrix
    // For uniform scaling, mat3(model) works. For non-uniform scaling, use transpose(inverse(model))
    fragNormal = normalize(mat3(model) * inNormal);
    
    // Provide per-vertex tex index as ivec3 (fragment shader expects this)
    fragTexIndices = ivec3(inTexIndex, inTexIndex, inTexIndex);
    // Set default texture weights (no blending in vertex shader)
    fragTexWeights = vec3(1.0, 0.0, 0.0);
    
    // compute world-space position and pass to fragment
    vec4 worldPos = model * vec4(inPos, 1.0);
    fragPosWorld = worldPos.xyz;
    fragPosWorldNotDisplaced = worldPos.xyz;  // No displacement in vertex shader
    
    // compute light-space position for shadow mapping
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;
    
    // Compute face normal (sharp normal) from model normal
    fragSharpNormal = normalize(mat3(model) * inNormal);
    
    // apply MVP transform to the vertex position
    gl_Position = ubo.viewProjection * worldPos;
}