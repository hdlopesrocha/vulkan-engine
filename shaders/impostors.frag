#version 450

// xy=UV, z=float(layerIdx) packed by the geometry shader.
layout(location = 0) in vec3 inTexCoord;
layout(location = 1) in vec3 inWorldPos;
layout(location = 2) flat in vec3 inFaceNormal;

layout(location = 0) out vec4 outColor;

// Full SolidParamsUBO — same layout as vegetation.frag.
layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
    vec4 lightDir;   // direction FROM light source toward scene
    vec4 lightColor;
} ubo;

// 60-layer impostor array: 3 billboard types × 20 Fibonacci views.
layout(set = 1, binding = 0) uniform sampler2DArray impostorArray;

void main() {
    vec4 color = texture(impostorArray, inTexCoord);
    if (color.a < 0.5) discard;

    // ── Lighting — same model as vegetation.frag ──────────────────────────
    vec3 N     = normalize(inFaceNormal);
    vec3 L     = normalize(-ubo.lightDir.xyz);
    float NdotL = max(dot(N, L), 0.0);

    vec3  V     = normalize(ubo.viewPos.xyz - inWorldPos);
    vec3  H     = normalize(L + V);
    float NdotH = (NdotL > 0.0) ? max(dot(N, H), 0.0) : 0.0;

    const float kAmbient  = 0.30;
    const float kSpecular = 0.08;
    const float kShine    = 16.0;
    vec3 ambient  = kAmbient            * ubo.lightColor.rgb;
    vec3 diffuse  = NdotL               * ubo.lightColor.rgb;
    vec3 specular = pow(NdotH, kShine) * kSpecular * ubo.lightColor.rgb;

    vec3 lighting = ambient + diffuse + specular;

    outColor = vec4(color.rgb * lighting, 1.0);
}
