#version 450

layout(triangles, equal_spacing, cw) in;

// UBO must match CPU-side UniformObject layout (we use same UBO as other shaders)
layout(binding = 0) uniform UBO {
    mat4 mvp;
    mat4 model;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 pomParams; // x=heightScale, y=minLayers, z=maxLayers, w=enabled
    vec4 pomFlags;
    vec4 parallaxLOD;
    vec4 mappingParams; // x=mappingMode
    vec4 specularParams;
    mat4 lightSpaceMatrix;
    vec4 shadowEffects;
} ubo;

// Inputs from TCS (per-vertex arrays) — use matching locations
layout(location = 7) in vec3 tc_fragLocalPos[];
layout(location = 8) in vec3 tc_fragLocalNormal[];
layout(location = 1) in vec2 tc_fragUV[];
layout(location = 5) flat in int tc_fragTexIndex[];

// samplers (height map needed here for displacement)
layout(binding = 1) uniform sampler2DArray heightArray;

// Outputs to next stage / not used in depth-only pass

// helper to sample height respecting per-material interpretation
float sampleHeight(vec2 uv, int texIndex) {
    float raw = texture(heightArray, vec3(uv, float(texIndex))).r;
    return (ubo.mappingParams.z > 0.5) ? raw : 1.0 - raw;
}

void main() {
    // barycentric coordinates
    vec3 bc = gl_TessCoord;

    // Interpolate local-space attributes
    vec3 localPos = tc_fragLocalPos[0] * bc.x + tc_fragLocalPos[1] * bc.y + tc_fragLocalPos[2] * bc.z;
    vec3 localNormal = normalize(tc_fragLocalNormal[0] * bc.x + tc_fragLocalNormal[1] * bc.y + tc_fragLocalNormal[2] * bc.z);
    vec2 uv = tc_fragUV[0] * bc.x + tc_fragUV[1] * bc.y + tc_fragUV[2] * bc.z;
    int texIndex = int(float(tc_fragTexIndex[0]) * bc.x + float(tc_fragTexIndex[1]) * bc.y + float(tc_fragTexIndex[2]) * bc.z + 0.5);

    vec3 displacedLocalPos = localPos;
    int mappingMode = int(ubo.mappingParams.x + 0.5);
    if (mappingMode == 2) {
        float height = sampleHeight(uv, texIndex);
        float heightScale = ubo.mappingParams.w;
        displacedLocalPos += localNormal * (height * heightScale);
    }

    vec4 worldPos = ubo.model * vec4(displacedLocalPos, 1.0);
    // For shadow pass, write light-space clip position
    gl_Position = ubo.lightSpaceMatrix * worldPos;
}
