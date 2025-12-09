#version 450


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragTangent;

// UBO must match binding 0 defined in vertex shader / host code
layout(binding = 0) uniform UBO {
    mat4 mvp;
    vec4 lightDir;
    vec4 lightColor;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D normalMap;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 tex = texture(texSampler, fragUV);
    // normal mapping: sample normal map (in tangent space) and transform to world/view using TBN
    vec3 N = normalize(fragNormal);
    vec3 T = normalize(fragTangent);
    vec3 B = normalize(cross(N, T));
    vec3 mapN = texture(normalMap, fragUV).xyz * 2.0 - 1.0; // from [0,1] to [-1,1]
    vec3 mapped = normalize(mapN.x * T + mapN.y * B + mapN.z * N);
    vec3 L = normalize(ubo.lightDir.xyz);
    // align convention: L is direction to light, so use dot(mapped, L)
    float diff = max(dot(mapped, L), 0.0);
    vec3 lighting = (0.25 + diff * ubo.lightColor.w) * ubo.lightColor.rgb; // ambient + diffuse*intensity
    vec3 shaded = tex.rgb * fragColor * lighting;
    outColor = vec4(shaded, tex.a);
}
