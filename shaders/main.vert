#version 450
#extension GL_ARB_shader_draw_parameters : require

#include "includes/ubo.glsl"
#include "includes/locations.glsl"

layout(location = ATTR_POS) in vec3 inPos;
layout(location = ATTR_COLOR) in vec3 inColor;
layout(location = ATTR_UV) in vec2 inUV;
layout(location = ATTR_NORMAL) in vec3 inNormal;
layout(location = ATTR_BRUSH_INDEX) in int inBrushIndex;

layout(location = VARY_COLOR) out vec3 fragColor;
layout(location = VARY_UV) out vec2 fragUV;
layout(location = VARY_NORMAL) out vec3 fragNormal;
layout(location = VARY_POSWORLD) out vec3 fragPosWorld;
layout(location = VARY_BRUSHPATCH) out int fragBrushIndex;      // per-vertex texture index for TCS
layout(location = VARY_POSLIGHT) out vec4 fragPosLightSpace;
layout(location = VARY_LOCALPOS) out vec3 fragLocalPos;          // provide local/world pos to TCS
layout(location = VARY_LOCALNORMAL) out vec3 fragLocalNormal;       // provide local/world normal to TCS
layout(location = VARY_SHARPNORMAL) out vec3 fragSharpNormal;      // face normal

void main() {
    fragColor = inColor;
    fragUV = inUV;
    // Models removed: always use identity model matrix
    mat4 model = mat4(1.0);
    // Transform normal to world space (model is identity here)
    // For uniform scaling, mat3(model) works. For non-uniform scaling, use transpose(inverse(model))
    fragNormal = normalize(mat3(model) * inNormal);
    
    // Pass per-vertex texture index as flat int for patch compression in TCS
    fragBrushIndex = inBrushIndex;
    
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