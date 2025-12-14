#version 450

layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 4) in vec3 fragPosWorld;
layout(location = 5) flat in int fragTexIndex;
layout(location = 9) in vec4 fragTangent;

#include "includes/ubo.glsl"

// Samplers (textures.glsl defines normalArray, heightArray, etc.)
#include "includes/textures.glsl"
#include "includes/tbn.glsl"
#include "includes/triplanar.glsl"

void main() {
    // Optionally use normal mapping/triplanar to compute per-fragment world normal
    // and apply a very small depth offset to reduce self-shadowing artifacts
    int texIndex = int(fragTexIndex + 0.5);
    vec3 N = normalize(fragNormal);
    vec3 worldNormal = N;

    // Precompute triplanar blend weights
    vec3 triW = abs(N);
    triW = triW * triW;
    float triWSum = triW.x + triW.y + triW.z + 1e-6;
    triW /= triWSum;

    if (ubo.triplanarParams.z > 0.5) {
        worldNormal = computeTriplanarNormal(fragPosWorld, triW, texIndex, N);
    } else if (ubo.mappingParams.x > 0.5 || ubo.materialFlags.w > 0.5) {
        vec3 nmap = texture(normalArray, vec3(fragUV, float(texIndex))).rgb * 2.0 - 1.0;
        vec3 T = vec3(0.0);
        vec3 B = vec3(0.0);
        vec3 tmpNormal = worldNormal;
        if (computeWorldNormalFromNormalMap(fragTangent, fragPosWorld, fragUV, tmpNormal, nmap, worldNormal, T, B)) {
            // worldNormal updated
        }
    }

    // Apply a tiny depth offset based on normal/angle to mitigate self-shadowing
    vec3 toLight = -normalize(ubo.lightDir.xyz);
    float NdotL = max(dot(worldNormal, toLight), 0.0);
    float bias = max(0.002 * (1.0 - NdotL), 0.0005);
    // Convert bias into a small window-space offset; the scale chosen empirically small
    float depthOffset = bias * 0.001;
    gl_FragDepth = clamp(gl_FragCoord.z + depthOffset, 0.0, 1.0);
}
