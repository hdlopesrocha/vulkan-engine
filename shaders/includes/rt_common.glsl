// Shared RT descriptor layout + payload structs + shading helpers.
// Included by every ray-tracing shader stage (rgen/rmiss/rchit/rahit).
// Set 0 is private to the RT pipeline and references the engine's EXISTING
// resources (no duplicated materials/textures):
//   materials  -> MaterialManager SSBO (same MaterialGPU layout as raster)
//   waterParams-> scene water params SSBO
//   sky        -> SkyUniform UBO
//   arrays     -> TextureArrayManager sampler2DArrays
//   solid/waterVtx/Idx -> IndirectRenderer packed pools (uint views; Vertex is
//     64 bytes = 16 uints: pos[0..2] color[3..5] uv[6..7] normal[8..10]
//     brushIndex[11] pad[12] hsv[13..15]; verified by static_assert in
//     AccelerationStructureManager.cpp)
#ifndef RT_COMMON_GLSL
#define RT_COMMON_GLSL

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

// ── Descriptors ─────────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba16f) uniform image2D outImage;

struct RtFrameData {
    mat4 viewProj;
    mat4 invViewProj;
    vec4 viewPos;        // xyz = camera position
    vec4 lightDir;       // xyz = direction TO the light (normalized on host)
    vec4 lightColor;     // rgb = color, a = intensity
    vec4 rtParams;       // x = time, y = debugMode, z = maxDepth, w = shadowsOn
    vec4 waterAbsorption;// rgb = Beer-Lambert sigma, a = waterEnabled
    vec4 waterMisc;      // x = IOR, y = reflectionStrength, z = refractionOn, w = reflectionsOn
    vec4 traceMask;      // x = radiance cull mask (0xFF normally, 0x01 = hide water)
    vec4 triplanarParams;// x = threshold, y = exponent (mirrors ubo.triplanarSettings)
    vec4 featureToggles; // x = normalMapping (ubo.materialFlags.w), y = roughnessOn, z = aoOn, w = tessellation (ubo.passParams.y)
};
layout(set = 0, binding = 2) uniform RtFrame { RtFrameData rt; };

struct MaterialGPU {
    vec4 materialFlags;
    vec4 mappingParams;
    vec4 specularParams;
    vec4 triplanarParams;
    vec4 normalParams;
    vec4 tessLevelParams;
    vec4 roughnessAOParams;
};
layout(std430, set = 0, binding = 3) readonly buffer Materials { MaterialGPU materials[]; };

struct WaterParamsGPU {
    vec4 params1; vec4 params2; vec4 params3;
    vec4 shallowColor; vec4 deepColor; vec4 waveParams;
    vec4 reserved1; vec4 reserved2; vec4 reserved3;
    vec4 tessParams; vec4 causticColor; vec4 causticParams; vec4 causticExtraParams;
};
layout(std430, set = 0, binding = 4) readonly buffer WaterParamsBlock { WaterParamsGPU waterParams[]; };

layout(set = 0, binding = 5) uniform SkyUBO {
    vec4 skyHorizon; vec4 skyZenith; vec4 skyParams;
    vec4 nightHorizon; vec4 nightZenith; vec4 nightParams;
} sky;

layout(set = 0, binding = 6) uniform sampler2DArray albedoArray;
layout(set = 0, binding = 7) uniform sampler2DArray normalArray;
layout(set = 0, binding = 8) uniform sampler2DArray heightArray;
layout(set = 0, binding = 9) uniform sampler2DArray roughnessArray;
layout(set = 0, binding = 10) uniform sampler2DArray aoArray;

struct SlotMeta { uint baseVertex; uint firstIndex; uint vertexCount; uint indexCount; };
layout(std430, set = 0, binding = 11) readonly buffer SolidMeta { SlotMeta solidMeta[]; };
layout(std430, set = 0, binding = 12) readonly buffer WaterMeta { SlotMeta waterMeta[]; };
layout(set = 0, binding = 13) readonly buffer SolidVtx { uint solidVtx[]; };
layout(set = 0, binding = 14) readonly buffer SolidIdx { uint solidIdx[]; };
layout(set = 0, binding = 15) readonly buffer WaterVtx { uint waterVtx[]; };
layout(set = 0, binding = 16) readonly buffer WaterIdx { uint waterIdx[]; };
// NDC depth output (same space as the raster depth buffer, so the existing
// post-process obstacle tests compare correctly). Written by rgen.
layout(set = 0, binding = 17, r32f) uniform image2D outDepth;

// ── Payloads (kept small: primary + shadow only) ────────────────────────────
struct RadiancePayload {
    vec3 radiance;
    float hitT;
    vec3 normal;
    float fresnel;
    int materialId;
    int slotId;
    int primitiveId;
    float waterThickness;
    int bounceCount;
    float reflectWeight;
    float refractWeight;
    float shadow;
    vec3 worldPos; // primary hit point (for NDC depth reconstruction in rgen)
};
struct ShadowPayload { float occlusion; };

// ── Constants ───────────────────────────────────────────────────────────────
// instanceCustomIndex is a 24-bit field: bit 23 = water, bit 22 = vegetation.
const uint WATER_BIT = 0x00800000u;
const uint VEG_BIT = 0x00400000u;
const uint SLOT_MASK = 0x003FFFFFu;
const float RAY_TMIN = 0.001f;
const float RAY_ORIGIN_EPS = 1.0e-3f;

// ── Vertex fetch (uint-view over the packed Vertex pools) ───────────────────
// 64-byte Vertex = 16 uints: pos[0..2] color[3..5] uv[6..7] normal[8..10]
// brushIndex[11] pad[12] hsv[13..15] (verified by host static_asserts).
void rtFetchVertex(bool isWater, uint slot, uint vertIndex,
                   out vec3 pos, out vec3 normal, out vec2 uv, out int brushIndex, out vec3 hsv) {
    SlotMeta m = isWater ? waterMeta[slot] : solidMeta[slot];
    uint w = (m.baseVertex + vertIndex) * 16u;
    if (isWater) {
        pos = vec3(uintBitsToFloat(waterVtx[w]), uintBitsToFloat(waterVtx[w + 1u]), uintBitsToFloat(waterVtx[w + 2u]));
        uv = vec2(uintBitsToFloat(waterVtx[w + 6u]), uintBitsToFloat(waterVtx[w + 7u]));
        normal = vec3(uintBitsToFloat(waterVtx[w + 8u]), uintBitsToFloat(waterVtx[w + 9u]), uintBitsToFloat(waterVtx[w + 10u]));
        brushIndex = int(waterVtx[w + 11u]);
        hsv = vec3(uintBitsToFloat(waterVtx[w + 13u]), uintBitsToFloat(waterVtx[w + 14u]), uintBitsToFloat(waterVtx[w + 15u]));
    } else {
        pos = vec3(uintBitsToFloat(solidVtx[w]), uintBitsToFloat(solidVtx[w + 1u]), uintBitsToFloat(solidVtx[w + 2u]));
        uv = vec2(uintBitsToFloat(solidVtx[w + 6u]), uintBitsToFloat(solidVtx[w + 7u]));
        normal = vec3(uintBitsToFloat(solidVtx[w + 8u]), uintBitsToFloat(solidVtx[w + 9u]), uintBitsToFloat(solidVtx[w + 10u]));
        brushIndex = int(solidVtx[w + 11u]);
        hsv = vec3(uintBitsToFloat(solidVtx[w + 13u]), uintBitsToFloat(solidVtx[w + 14u]), uintBitsToFloat(solidVtx[w + 15u]));
    }
}

void rtFetchTriangle(bool isWater, uint slot, uint primId,
                     out vec3 p0, out vec3 p1, out vec3 p2,
                     out vec3 n0, out vec3 n1, out vec3 n2,
                     out vec2 uv0, out vec2 uv1, out vec2 uv2,
                     out ivec3 brush, out vec3 h0, out vec3 h1, out vec3 h2) {
    SlotMeta m = isWater ? waterMeta[slot] : solidMeta[slot];
    uint f = m.firstIndex + primId * 3u;
    uint i0 = isWater ? waterIdx[f] : solidIdx[f];
    uint i1 = isWater ? waterIdx[f + 1u] : solidIdx[f + 1u];
    uint i2 = isWater ? waterIdx[f + 2u] : solidIdx[f + 2u];
    int b;
    rtFetchVertex(isWater, slot, i0, p0, n0, uv0, b, h0); brush.x = b;
    rtFetchVertex(isWater, slot, i1, p1, n1, uv1, b, h1); brush.y = b;
    rtFetchVertex(isWater, slot, i2, p2, n2, uv2, b, h2); brush.z = b;
}

// ── Sky / environment (miss shader + reflection fallback) ───────────────────
vec3 rtSkyColor(vec3 dir) {
    vec3 d = normalize(dir);
    float h = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = mix(sky.skyHorizon.rgb, sky.skyZenith.rgb, pow(h, 1.2));
    // Sun disk + flare from the scene light direction.
    vec3 sunDir = normalize(rt.lightDir.xyz);
    float s = clamp(dot(d, sunDir), 0.0, 1.0);
    float flare = sky.skyParams.z;
    col += rt.lightColor.rgb * (pow(s, 1500.0) * 8.0 + pow(s, 24.0) * 0.35 * (0.5 + flare));
    // Warmth tint near the horizon (matches the raster sky's warmth term).
    col = mix(col, col * vec3(1.06, 0.96, 0.88), (1.0 - abs(d.y)) * sky.skyParams.x * 0.4);
    return max(col, vec3(0.0));
}

// Schlick Fresnel (dielectric F0=0.04, metallic uses albedo as F0).
float rtSchlick(float cosTheta, float f0) {
    float c = clamp(1.0 - cosTheta, 0.0, 1.0);
    return f0 + (1.0 - f0) * c * c * c * c * c;
}

// Beer-Lambert attenuation for a water path of `thickness` meters.
vec3 rtBeerLambert(vec3 sigma, float thickness) {
    return exp(-sigma * max(thickness, 0.0));
}

#endif // RT_COMMON_GLSL
