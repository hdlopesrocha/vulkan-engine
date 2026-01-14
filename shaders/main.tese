#version 450

layout(triangles, equal_spacing, cw) in;

#include "includes/ubo.glsl"

// Inputs from TCS (per-vertex arrays)
layout(location = 0) in vec3 tc_fragColor[];
layout(location = 1) in vec2 tc_fragUV[];
layout(location = 2) in vec3 tc_fragNormal[]; // keep for compatibility (world-space if provided)
layout(location = 4) in vec3 tc_fragPosWorld[]; // world pos passed through by TCS (not used for displacement)
layout(location = 5) flat in ivec3 tc_fragTexIndex[];
layout(location = 7) in vec3 tc_fragLocalPos[]; // local-space position
layout(location = 8) in vec3 tc_fragLocalNormal[];
layout(location = 11) in vec3 tc_fragTexWeights[];

// Outputs to fragment shader (match main.frag inputs)
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal; // world-space normal
layout(location = 5) flat out ivec3 fragTexIndices;
layout(location = 11) out vec3 fragTexWeights;
layout(location = 4) out vec3 fragPosWorld;
layout(location = 7) out vec3 fragPosWorldNotDisplaced;
layout(location = 6) out vec4 fragPosLightSpace;
// fragTangent removed: fragment shader computes tangent per-pixel for triplanar mapping
layout(location = 10) out vec3 fragSharpNormal; // face normal computed from triangle corners (sharp)

#include "includes/textures.glsl"

#include "includes/common.glsl"

#include "includes/displacement.glsl"



void main() {
    bool isDepthPass = ubo.passParams.x > 0.5;

    // barycentric coordinates
    vec3 bc = gl_TessCoord;

    // Interpolate local-space position and normal
    vec3 localPos = tc_fragLocalPos[0] * bc.x + tc_fragLocalPos[1] * bc.y + tc_fragLocalPos[2] * bc.z;
    vec3 localNormal = normalize(tc_fragLocalNormal[0] * bc.x + tc_fragLocalNormal[1] * bc.y + tc_fragLocalNormal[2] * bc.z);
    vec2 uv = tc_fragUV[0] * bc.x + tc_fragUV[1] * bc.y + tc_fragUV[2] * bc.z;
    ivec3 texIndices = tc_fragTexIndex[0];
    vec3 weights = tc_fragTexWeights[0] * bc.x + tc_fragTexWeights[1] * bc.y + tc_fragTexWeights[2] * bc.z;


    // Calculate position with displacement (needed for both passes)
    vec3 worldNormal = normalize(localNormal);
    vec4 worldPos = vec4(localPos, 1.0);
    
    float mappingFlag = materials[texIndices.x].mappingParams.x * weights.x + materials[texIndices.y].mappingParams.x * weights.y + materials[texIndices.z].mappingParams.x * weights.z;
    mappingFlag *= ubo.passParams.y;
    vec3 displacedLocalPos = mappingFlag > 0.5 ? applyDisplacement(localPos, localNormal, worldPos.xyz, worldNormal, uv, texIndices, weights) : localPos;
    
    gl_Position = ubo.viewProjection * vec4(displacedLocalPos, 1.0);

    if (isDepthPass) {
        // Depth pass: set dummy outputs (fragment shader early-returns anyway)
        fragColor = vec3(0.0);
        fragUV = vec2(0.0);
        fragNormal = vec3(0.0);
        fragTexIndices = ivec3(0);
        fragTexWeights = vec3(0.0);
        fragPosWorld = vec3(0.0);
        fragPosWorldNotDisplaced = vec3(0.0);
        fragPosLightSpace = vec4(0.0);
        fragSharpNormal = vec3(0.0);
    } else {
        // Full pass: calculate all outputs for shading
        fragColor = tc_fragColor[0] * bc.x + tc_fragColor[1] * bc.y + tc_fragColor[2] * bc.z;
        
        fragPosWorldNotDisplaced = worldPos.xyz;
        worldPos = vec4(displacedLocalPos, 1.0);
        fragPosWorld = worldPos.xyz;
        fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;
        
        fragUV = uv;
        fragTexIndices = texIndices;
        fragTexWeights = weights;
        fragNormal = normalize(localNormal);

        // Compute explicit face (sharp) normal from triangle corners
        vec3 p0_sh = tc_fragLocalPos[0];
        vec3 p1_sh = tc_fragLocalPos[1];
        vec3 p2_sh = tc_fragLocalPos[2];
        vec3 faceLocal = normalize(cross(p2_sh - p0_sh, p1_sh - p0_sh));
        fragSharpNormal = normalize(faceLocal);
    }

    // Per-vertex tangents are no longer propagated; fragment will compute T/B/N as needed.
}