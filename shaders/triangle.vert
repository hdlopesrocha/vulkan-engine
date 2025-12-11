#version 450

// UBO layout must match the CPU-side UniformObject (std140-like):
// mat4 mvp; mat4 model; vec4 viewPos; vec4 lightDir; vec4 lightColor;
layout(binding = 0) uniform UBO {
    mat4 mvp;
    mat4 model;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 pomParams; // x=heightScale, y=minLayers, z=maxLayers, w=enabled
    vec4 pomFlags;  // x=flipNormalY, y=flipTangentHandedness, z=ambient, w=unused
    vec4 parallaxLOD; // x=parallaxNear, y=parallaxFar, z=reductionAtFar, w=unused
    vec4 mappingParams; // x=mappingMode (0=none,1=parallax,2=tessellation), y/z/w unused
    vec4 specularParams; // x=specularStrength, y=shininess, z=unused, w=unused
    mat4 lightSpaceMatrix; // for shadow mapping
    vec4 shadowEffects; // x=enableSelfShadow, y=enableShadowDisplacement, z=selfShadowQuality, w=unused
} ubo;


layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec3 inTangent;
layout(location = 5) in float inTexIndex;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal;
layout(location = 3) out vec3 fragTangent;
layout(location = 5) flat out int fragTexIndex;
layout(location = 4) out vec3 fragPosWorld;
layout(location = 6) out vec4 fragPosLightSpace;
layout(location = 7) out vec3 fragLocalPos;
layout(location = 8) out vec3 fragLocalNormal;
layout(location = 9) out vec3 fragLocalTangent;

void main() {
    fragColor = inColor;
    fragUV = inUV;
    // Transform normal to world space using the model matrix
    // For uniform scaling, mat3(model) works. For non-uniform scaling, use transpose(inverse(model))
    fragNormal = normalize(mat3(ubo.model) * inNormal);
    fragTangent = normalize(mat3(ubo.model) * inTangent);
    fragTexIndex = int(inTexIndex + 0.5);
    // compute world-space position and pass to fragment
    vec4 worldPos = ubo.model * vec4(inPos, 1.0);
    fragPosWorld = worldPos.xyz;
    // compute light-space position for shadow mapping
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;
    // pass local-space position (used by tessellation/displacement)
    fragLocalPos = inPos;
    // also pass local-space normal/tangent (before model transform) for tessellation displacement
    fragLocalNormal = inNormal;
    fragLocalTangent = inTangent;
    // apply MVP transform to the vertex position (MVP already includes model transform)
    gl_Position = ubo.mvp * vec4(inPos, 1.0);
}
