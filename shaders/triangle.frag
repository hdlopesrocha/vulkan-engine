#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;

// UBO must match binding 0 defined in vertex shader / host code
layout(binding = 0) uniform UBO {
    mat4 mvp;
    vec4 lightDir;
    vec4 lightColor;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 tex = texture(texSampler, fragUV);
    // simple Lambertian diffuse using approximated normal
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(ubo.lightDir.xyz);
    float diff = max(dot(N, -L), 0.0);
    vec3 lighting = (0.2 + diff * ubo.lightColor.w) * ubo.lightColor.rgb; // ambient + diffuse*intensity
    vec3 shaded = tex.rgb * fragColor * lighting;
    outColor = vec4(shaded, tex.a);
}
