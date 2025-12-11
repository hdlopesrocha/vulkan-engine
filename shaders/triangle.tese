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
    vec4 pomFlags;  // x=flipNormalY, y=flipTangentHandedness, z=ambient, w=flipParallaxDirection
    vec4 parallaxLOD; // x=parallaxNear, y=parallaxFar, z=reductionAtFar, w=unused
    vec4 mappingParams; // x=mappingMode
    vec4 specularParams; // x=specularStrength, y=shininess
    mat4 lightSpaceMatrix;
    vec4 shadowEffects;
} ubo;

// Inputs from TCS (per-vertex arrays)
layout(location = 0) in vec3 tc_fragColor[];
layout(location = 1) in vec2 tc_fragUV[];
layout(location = 2) in vec3 tc_fragNormal[]; // local-space normal
layout(location = 3) in vec3 tc_fragTangent[]; // local-space tangent
layout(location = 4) in vec3 tc_fragPosWorld[]; // world pos passed through by TCS (not used for displacement)
layout(location = 5) flat in int tc_fragTexIndex[];
layout(location = 7) in vec3 tc_fragLocalPos[]; // local-space position

// Outputs to fragment shader (match triangle.frag inputs)
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal; // world-space normal
layout(location = 3) out vec3 fragTangent; // world-space tangent
layout(location = 5) flat out int fragTexIndex;
layout(location = 4) out vec3 fragPosWorld;
layout(location = 6) out vec4 fragPosLightSpace;

// samplers (height map needed here for displacement)
layout(binding = 3) uniform sampler2DArray heightArray;

void main() {
    // barycentric coordinates
    vec3 bc = gl_TessCoord;

    // Interpolate local-space position, normal, tangent, uv, and texIndex
    vec3 localPos = tc_fragLocalPos[0] * bc.x + tc_fragLocalPos[1] * bc.y + tc_fragLocalPos[2] * bc.z;
    vec3 localNormal = normalize(tc_fragNormal[0] * bc.x + tc_fragNormal[1] * bc.y + tc_fragNormal[2] * bc.z);
    vec3 localTangent = normalize(tc_fragTangent[0] * bc.x + tc_fragTangent[1] * bc.y + tc_fragTangent[2] * bc.z);
    vec2 uv = tc_fragUV[0] * bc.x + tc_fragUV[1] * bc.y + tc_fragUV[2] * bc.z;
    int texIndex = int(float(tc_fragTexIndex[0]) * bc.x + float(tc_fragTexIndex[1]) * bc.y + float(tc_fragTexIndex[2]) * bc.z + 0.5);
    fragColor = tc_fragColor[0] * bc.x + tc_fragColor[1] * bc.y + tc_fragColor[2] * bc.z;

    // Default: no displacement
    vec3 displacedLocalPos = localPos;

    // Only apply displacement when mapping mode indicates tessellation AND POM enabled on material
    int mappingMode = int(ubo.mappingParams.x + 0.5);
    if (mappingMode == 2 && ubo.pomParams.w > 0.5) {
        // Sample height (invert as used elsewhere)
        float height = 1.0 - texture(heightArray, vec3(uv, float(texIndex))).r;
        float heightScale = ubo.pomParams.x;
        // Displace along local normal (model transform will move to world later)
        displacedLocalPos += localNormal * (height * heightScale);
    }

    // Compute world-space position and normals
    vec4 worldPos = ubo.model * vec4(displacedLocalPos, 1.0);
    fragPosWorld = worldPos.xyz;
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;

    // Transform normal/tangent to world space
    fragNormal = normalize(mat3(ubo.model) * localNormal);
    fragTangent = normalize(mat3(ubo.model) * localTangent);

    fragUV = uv;
    fragTexIndex = texIndex;

    // Output clip-space position using MVP (MVP includes model matrix)
    gl_Position = ubo.mvp * vec4(displacedLocalPos, 1.0);
}
