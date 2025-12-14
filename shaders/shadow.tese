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
layout(location = 9) in vec4 tc_fragTangent[];

// Outputs to fragment shader (match main.frag inputs)
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal; // world-space normal
layout(location = 5) flat out int fragTexIndex;
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
    int texIndex = int(float(tc_fragTexIndex[0]) * bc.x + float(tc_fragTexIndex[1]) * bc.y + float(tc_fragTexIndex[2]) * bc.z + 0.5);
    
    // Not required by shadow shader
    fragColor = tc_fragColor[0] * bc.x + tc_fragColor[1] * bc.y + tc_fragColor[2] * bc.z;

    // Apply displacement
    vec3 displacedLocalPos = localPos;
    float disp = 0.0;
    if (ubo.mappingParams.x > 0.5) {
        // Use triplanar height sampling if enabled
        vec3 worldPosForSampling = (ubo.model * vec4(localPos, 1.0)).xyz;
        float height = 0.0;
        if (ubo.triplanarParams.z > 0.5) {
            height = sampleHeightTriplanar(worldPosForSampling, localNormal, texIndex);
        } else {
            height = sampleHeight(uv, texIndex);
        }
        float heightScale = ubo.mappingParams.w;
        // Displace outward along surface normal based on sampled height
        disp = height * heightScale;
        displacedLocalPos += localNormal * disp;
    }
    // displacement magnitude used only on-TES; no debug output

    // Compute world-space position and normals
    vec4 worldPos = ubo.model * vec4(displacedLocalPos, 1.0);
    fragPosWorld = worldPos.xyz;
    fragUV = uv;
    fragTexIndex = texIndex;
    fragNormal = normalize(mat3(ubo.model) * localNormal);
    
    vec4 tangentLocal = tc_fragTangent[0] * bc.x + tc_fragTangent[1] * bc.y + tc_fragTangent[2] * bc.z;
    fragTangent = vec4(normalize(mat3(ubo.model) * tangentLocal.xyz), tangentLocal.w);

    gl_Position = ubo.lightSpaceMatrix * worldPos;
}
