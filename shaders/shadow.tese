#version 450

layout(triangles, equal_spacing, cw) in;

#include "includes/ubo.glsl"

// Inputs from TCS (per-vertex arrays) — use matching locations
layout(location = 7) in vec3 tc_fragLocalPos[];
layout(location = 8) in vec3 tc_fragLocalNormal[];
layout(location = 1) in vec2 tc_fragUV[];
layout(location = 5) flat in int tc_fragTexIndex[];

// samplers (height map needed here for displacement)
layout(binding = 3) uniform sampler2DArray heightArray;

// Outputs to next stage / not used in depth-only pass

#include "includes/common.glsl"

#include "includes/displacement.glsl"

void main() {
    // barycentric coordinates
    vec3 bc = gl_TessCoord;

    // Interpolate local-space attributes
    vec3 localPos = tc_fragLocalPos[0] * bc.x + tc_fragLocalPos[1] * bc.y + tc_fragLocalPos[2] * bc.z;
    vec3 localNormal = normalize(tc_fragLocalNormal[0] * bc.x + tc_fragLocalNormal[1] * bc.y + tc_fragLocalNormal[2] * bc.z);
    vec2 uv = tc_fragUV[0] * bc.x + tc_fragUV[1] * bc.y + tc_fragUV[2] * bc.z;
    int texIndex = int(float(tc_fragTexIndex[0]) * bc.x + float(tc_fragTexIndex[1]) * bc.y + float(tc_fragTexIndex[2]) * bc.z + 0.5);

    vec3 displacedLocalPos = applyDisplacement(localPos, localNormal, uv, texIndex);

    vec4 worldPos = ubo.model * vec4(displacedLocalPos, 1.0);
    // For shadow pass, write light-space clip position
    gl_Position = ubo.lightSpaceMatrix * worldPos;
}
