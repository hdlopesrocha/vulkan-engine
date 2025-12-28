#version 450

layout(triangles, equal_spacing, cw) in;

#include "includes/ubo.glsl"

// Inputs from TCS (per-vertex arrays)
layout(location = 0) in vec3 tc_fragColor[];
layout(location = 1) in vec2 tc_fragUV[];
layout(location = 2) in vec3 tc_fragNormal[]; // keep for compatibility (world-space if provided)
layout(location = 4) in vec3 tc_fragPosWorld[]; // world pos passed through by TCS (not used for displacement)
layout(location = 5) flat in ivec3 tc_fragTexIndex[];
layout(location = 11) in vec3 tc_fragTexWeights[];
layout(location = 7) in vec3 tc_fragLocalPos[]; // local-space position
layout(location = 8) in vec3 tc_fragLocalNormal[];
layout(location = 9) in vec4 tc_fragTangent[];

// Outputs to fragment shader (match main.frag inputs)
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal; // world-space normal
layout(location = 5) flat out ivec3 fragTexIndices;
layout(location = 11) out vec3 fragTexWeights;
layout(location = 4) out vec3 fragPosWorld;
layout(location = 7) out vec3 fragPosWorldNotDisplaced;
layout(location = 6) out vec4 fragPosLightSpace;
layout(location = 9) out vec4 fragTangent;

#include "includes/textures.glsl"

#include "includes/common.glsl"

#include "includes/displacement.glsl"



void main() {
    // barycentric coordinates
    vec3 bc = gl_TessCoord;

    // Interpolate local-space position, normal, tangent, uv, and texIndex
    vec3 localPos = tc_fragLocalPos[0] * bc.x + tc_fragLocalPos[1] * bc.y + tc_fragLocalPos[2] * bc.z;
    vec3 localNormal = normalize(tc_fragLocalNormal[0] * bc.x + tc_fragLocalNormal[1] * bc.y + tc_fragLocalNormal[2] * bc.z);
    vec2 uv = tc_fragUV[0] * bc.x + tc_fragUV[1] * bc.y + tc_fragUV[2] * bc.z;
    // preserve per-corner tex indices and barycentric weights for blending in the fragment shader
    ivec3 texIndices = tc_fragTexIndex[0];

    // Interpolate per-invocation weight basis vectors provided by the TCS so non-tess and tess paths match
    vec3 weights = tc_fragTexWeights[0] * bc.x + tc_fragTexWeights[1] * bc.y + tc_fragTexWeights[2] * bc.z;
    bool isShadowPass = ubo.passParams.x > 0.5;
    if (!isShadowPass) {
        fragColor = tc_fragColor[0] * bc.x + tc_fragColor[1] * bc.y + tc_fragColor[2] * bc.z;
    } else {
        fragColor = vec3(0.0);
    }

    mat4 model = pushConstants.model;

    // Compute world position and normal for triplanar sampling
    vec3 worldNormal = normalize(mat3(model) * localNormal);
    vec4 worldPos = model * vec4(localPos, 1.0);

    // Apply displacement (only when global tessellation enabled)
    float mappingFlag = materials[texIndices.x].mappingParams.x * weights.x + materials[texIndices.y].mappingParams.x * weights.y + materials[texIndices.z].mappingParams.x * weights.z;
    // Respect global tessellation enabled flag (passParams.y). If tessellation is disabled, mappingFlag becomes 0 and no displacement occurs
    mappingFlag *= ubo.passParams.y;
    vec3 displacedLocalPos = mappingFlag > 0.5 ? applyDisplacement(localPos, localNormal, worldPos.xyz, worldNormal, uv, texIndices, weights) : localPos;
    

    // Compute world-space position and normals
    fragPosWorldNotDisplaced = worldPos.xyz;
    worldPos = model * vec4(displacedLocalPos, 1.0);

    fragPosWorld = worldPos.xyz;
    if (!isShadowPass) {
        fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;
    } else {
        fragPosLightSpace = vec4(0.0);
    }

    fragUV = uv;
    fragTexIndices = texIndices;
    fragTexWeights = weights;
    // Output clip-space position using MVP (MVP includes model matrix)
    gl_Position = ubo.viewProjection * model * vec4(displacedLocalPos, 1.0);

    // Compute fragNormal: do not apply normal mapping here — use transformed geometry normal
    fragNormal = normalize(mat3(model) * localNormal);
    // Compute world-space tangent handedness-aware
    vec4 tangentLocal = tc_fragTangent[0] * bc.x + tc_fragTangent[1] * bc.y + tc_fragTangent[2] * bc.z;
    fragTangent = vec4(normalize(mat3(model) * tangentLocal.xyz), tangentLocal.w);
}