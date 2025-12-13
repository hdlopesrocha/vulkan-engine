#version 450

layout(triangles, equal_spacing, cw) in;

#include "includes/ubo.glsl"

// Inputs from TCS (per-vertex arrays)
layout(location = 0) in vec3 tc_fragColor[];
layout(location = 1) in vec2 tc_fragUV[];
layout(location = 2) in vec3 tc_fragNormal[]; // keep for compatibility (world-space if provided)
layout(location = 4) in vec3 tc_fragPosWorld[]; // world pos passed through by TCS (not used for displacement)
layout(location = 5) flat in int tc_fragTexIndex[];
layout(location = 7) in vec3 tc_fragLocalPos[]; // local-space position
layout(location = 8) in vec3 tc_fragLocalNormal[];

// Outputs to fragment shader (match main.frag inputs)
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal; // world-space normal
layout(location = 5) flat out int fragTexIndex;
layout(location = 4) out vec3 fragPosWorld;
layout(location = 6) out vec4 fragPosLightSpace;
// (removed debug displacement output)

#include "includes/textures.glsl"

#include "includes/common.glsl"

#include "includes/displacement.glsl"

void main() {
    // barycentric coordinates
    vec3 bc = gl_TessCoord;

    // Interpolate local-space position, normal, tangent, uv, and texIndex
    vec3 localPos = tc_fragLocalPos[0] * bc.x + tc_fragLocalPos[1] * bc.y + tc_fragLocalPos[2] * bc.z;
    // Prefer the explicitly-passed local-space normal/tangent for correct displacement
    vec3 localNormal = normalize(tc_fragLocalNormal[0] * bc.x + tc_fragLocalNormal[1] * bc.y + tc_fragLocalNormal[2] * bc.z);
    // Tangent attribute removed; do not interpolate local tangent
    vec2 uv = tc_fragUV[0] * bc.x + tc_fragUV[1] * bc.y + tc_fragUV[2] * bc.z;
    int texIndex = int(float(tc_fragTexIndex[0]) * bc.x + float(tc_fragTexIndex[1]) * bc.y + float(tc_fragTexIndex[2]) * bc.z + 0.5);
    fragColor = tc_fragColor[0] * bc.x + tc_fragColor[1] * bc.y + tc_fragColor[2] * bc.z;

    // Apply displacement (also compute magnitude for debug visualization)
    vec3 displacedLocalPos = localPos;
    float disp = 0.0;
    if (ubo.mappingParams.x > 0.5) {
        float height = sampleHeight(uv, texIndex);
        float heightScale = ubo.mappingParams.w;
        disp = height * heightScale;
        displacedLocalPos += localNormal * disp;
    }
    // displacement magnitude used only on-TES; no debug output

    // Compute world-space position and normals
    vec4 worldPos = ubo.model * vec4(displacedLocalPos, 1.0);
    fragPosWorld = worldPos.xyz;
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;

    fragUV = uv;
    fragTexIndex = texIndex;
    // Output clip-space position using MVP (MVP includes model matrix)
    gl_Position = ubo.mvp * vec4(displacedLocalPos, 1.0);

    // Compute fragNormal
    if (ubo.mappingParams.x > 0.5) {
        // Compute tangent and bitangent from the patch
        vec3 edge1 = tc_fragLocalPos[1] - tc_fragLocalPos[0];
        vec3 edge2 = tc_fragLocalPos[2] - tc_fragLocalPos[0];
        vec2 deltaUV1 = tc_fragUV[1] - tc_fragUV[0];
        vec2 deltaUV2 = tc_fragUV[2] - tc_fragUV[0];
        float denom = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
        vec3 tangent, bitangent;
        if (abs(denom) > 1e-6) {
            float f = 1.0 / denom;
            tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
            bitangent = f * (-deltaUV2.x * edge1 + deltaUV1.x * edge2);
        } else {
            // Fallback: assume tangent along X, bitangent along Y or something
            tangent = vec3(1.0, 0.0, 0.0);
            bitangent = vec3(0.0, 1.0, 0.0);
        }
        // Transform to world space
        tangent = mat3(ubo.model) * tangent;
        bitangent = mat3(ubo.model) * bitangent;
        vec3 normal = mat3(ubo.model) * localNormal;
        // Orthonormalize
        tangent = normalize(tangent - normal * dot(normal, tangent));
        bitangent = normalize(bitangent - normal * dot(normal, bitangent));
        mat3 TBN = mat3(tangent, bitangent, normal);
        // Sample normal map
        vec3 normalMap = texture(normalArray, vec3(fragUV, float(fragTexIndex))).rgb * 2.0 - 1.0;
        normalMap = normalize(normalMap);
        fragNormal = normalize(TBN * normalMap);
    } else {
        fragNormal = normalize(mat3(ubo.model) * localNormal);
    }
}
