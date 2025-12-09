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

void main() {
    fragColor = inColor;
    fragUV = inUV;
    // approximate normal from position (cube is centered at origin)
    fragNormal = normalize(inNormal);
    fragTangent = normalize(inTangent);
        fragTexIndex = int(inTexIndex + 0.5);
    // compute world-space position and pass to fragment
    vec4 worldPos = ubo.model * vec4(inPos, 1.0);
    fragPosWorld = worldPos.xyz;
    // apply MVP transform from UBO
    gl_Position = ubo.mvp * vec4(inPos, 1.0);
}
