#version 450
layout(location = 0) in vec3 inTexCoord;  // xy=uv, z=array layer
layout(location = 1) in flat int inTexIndex;
layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0) uniform sampler2DArray albedoArray;
layout(set = 1, binding = 1) uniform sampler2DArray normalArray;
layout(set = 1, binding = 2) uniform sampler2DArray opacityArray;

layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float pad0;
    vec4 windDirAndStrength;
    vec4 windNoise;
    vec4 windShape;
    vec4 windTurbulence;
    vec4 densityParams;
    vec4 cameraPosAndFalloff;
};

void main() {
    vec3 coord = vec3(inTexCoord.xy, inTexCoord.z);

    // Per-pixel samples.
    vec4  leafAlbedo  = texture(albedoArray,  coord);
    float opacity     = texture(opacityArray, coord).r;
    vec3  leafNormEnc = texture(normalArray,  coord).rgb;

    // ---------------------------------------------------------------
    // Background colour from a high mip-level sample.
    // The CPU compositor pre-fills empty billboard pixels with the
    // opacity-weighted average leaf colour, so a high-mip sample
    // already returns a spatially-averaged, leaf-derived background
    // without needing the division trick.
    // ---------------------------------------------------------------
    const float kAvgMip = 5.0;
    vec3 bgAlbedo  = textureLod(albedoArray, coord, kAvgMip).rgb;
    vec3 bgNormEnc = textureLod(normalArray, coord, kAvgMip).rgb;

    // Decode normals from [0,1] to [-1,1] tangent space.
    vec3  leafNorm = normalize(leafNormEnc * 2.0 - 1.0);
    vec3  bgNorm   = normalize(bgNormEnc   * 2.0 - 1.0);

    // Normal confidence: Z in tangent space — 1 = facing viewer (leaf interior),
    // 0 = grazing / edge pixel.
    float leafNConf = clamp(leafNorm.z, 0.0, 1.0);
    float bgNConf   = clamp(bgNorm.z,  0.0, 1.0);

    // Opacity mask: steep sigmoid kills fringe pixels.
    float opacityWeight = smoothstep(0.35, 0.65, opacity);

    // Normal weight: at fringe pixels fall back to the background normal
    // confidence so the fill colour stays surface-coherent.
    float normalWeight = mix(bgNConf, leafNConf, opacityWeight);

    // Combined alpha — both maps gate the edge.
    float weight = opacityWeight * normalWeight;

    // Compose: pure leaf colour in the interior, smooth fill at the fringe.
    outColor.rgb = mix(bgAlbedo, leafAlbedo.rgb, weight);
    outColor.a   = weight;

    if (outColor.a < 0.5) discard;
}
