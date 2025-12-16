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

    // Apply displacement
    vec3 displacedLocalPos = localPos;
    float disp = 0.0;
    // Determine if any of the three corner materials enable mapping (blend by barycentric weights)
    float mappingFlag = materials[texIndices.x].mappingParams.x * weights.x + materials[texIndices.y].mappingParams.x * weights.y + materials[texIndices.z].mappingParams.x * weights.z;
    if (mappingFlag > 0.5) {
        // Use triplanar or UV height sampling per-layer then blend
        vec3 worldPosForSampling = (ubo.model * vec4(localPos, 1.0)).xyz;
        float h0 = 0.0;
        float h1 = 0.0;
        float h2 = 0.0;
        if (materials[texIndices.x].triplanarParams.z > 0.5) h0 = sampleHeightTriplanar(worldPosForSampling, localNormal, texIndices.x);
        else h0 = sampleHeight(uv, texIndices.x);
        if (materials[texIndices.y].triplanarParams.z > 0.5) h1 = sampleHeightTriplanar(worldPosForSampling, localNormal, texIndices.y);
        else h1 = sampleHeight(uv, texIndices.y);
        if (materials[texIndices.z].triplanarParams.z > 0.5) h2 = sampleHeightTriplanar(worldPosForSampling, localNormal, texIndices.z);
        else h2 = sampleHeight(uv, texIndices.z);
        float height = h0 * weights.x + h1 * weights.y + h2 * weights.z;
        float heightScale = materials[texIndices.x].mappingParams.w * weights.x + materials[texIndices.y].mappingParams.w * weights.y + materials[texIndices.z].mappingParams.w * weights.z;
        // Displace outward along surface normal based on sampled height
        disp = height * heightScale;
        displacedLocalPos += localNormal * disp;
    }
    // displacement magnitude used only on-TES; no debug output

    // Compute world-space position and normals
    vec4 worldPos = ubo.model * vec4(displacedLocalPos, 1.0);
    fragPosWorld = worldPos.xyz;
    if (!isShadowPass) {
        fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;
    } else {
        fragPosLightSpace = vec4(0.0);
    }

    fragUV = uv;
    fragTexIndices = texIndices;
    fragTexWeights = weights;
    // Output clip-space position using viewProjection * model
    gl_Position = ubo.viewProjection * ubo.model * vec4(displacedLocalPos, 1.0);

    // Compute fragNormal: do not apply normal mapping here — use transformed geometry normal
    fragNormal = normalize(mat3(ubo.model) * localNormal);
    // Compute world-space tangent handedness-aware
    vec4 tangentLocal = tc_fragTangent[0] * bc.x + tc_fragTangent[1] * bc.y + tc_fragTangent[2] * bc.z;
    fragTangent = vec4(normalize(mat3(ubo.model) * tangentLocal.xyz), tangentLocal.w);
}