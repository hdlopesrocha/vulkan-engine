// Unified grass shading shared by VegetationRenderer.frag (blades) and
// VegetationRendererImpostor.frag (captured tufts), so the LOD hand-off has
// no brightness edge. Requires before inclusion: SceneBindings.glsl (ubo),
// SkyView.glsl (sky), Shadows.glsl (ShadowCalculationHard, cloudShadowAt)
// and the fragPosWorld global. Callers supply the albedo, the two-sided
// world normal, the (flat, pre-normalized) light direction, the true surface
// world position and the shadowPass gate.
#ifndef GRASS_SHADING_GLSL
#define GRASS_SHADING_GLSL

const float kGrassAmbient  = 0.32;
const float kGrassSpecular = 0.06;
const float kGrassShine    = 24.0;
// Backlit translucency gain. Deliberately height-independent so blades and
// impostors (which have no heightT) evaluate bit-identical values.
const float kGrassTransl   = 0.25;

// Shortest-arc tilt mapping +Y onto a (unit) surface normal, mirroring
// tiltToNormal in VegetationCommon.glsl (kept local so this header stays
// dependency-free). The blade VS tilts normals with it; the impostor path
// must apply the same tilt to captured (flat-land) normals at runtime.
vec3 grassTiltToNormal(vec3 v, vec3 n) {
    vec3 k = vec3(n.z, 0.0, -n.x); // cross(+Y, n)
    float s = length(k);
    if (s < 1e-4) return (n.y >= 0.0) ? v : vec3(v.x, -v.y, v.z);
    k /= s;
    float c = clamp(n.y, -1.0, 1.0);
    return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c);
}

// Per-tuft albedo variation. Blades call it with the fragment world pos,
// impostors with the depth-reconstructed blade pos: same 3 m cells, same
// single factor applied to root and tip, so both LODs agree exactly.
float grassVariation(vec3 worldPos) {
    float varHash = fract(sin(dot(floor(worldPos.xz * 3.0), vec2(127.1, 311.7))) * 43758.5453);
    return (varHash - 0.5) * 0.22;
}

// Ambient + Lambert + Blinn-Phong + backlit translucency, gated by hard CSM
// and cloud shadows. Writes the fragPosWorld global for cascades 1 & 2.
vec3 grassShade(vec3 albedo, vec3 N, vec3 L, vec3 worldPos, bool shadowPass) {
    fragPosWorld = worldPos;
    float NdotL = max(dot(N, L), 0.0);
    vec3 V = normalize(ubo.viewPosition - worldPos);
    vec3 H = normalize(L + V);
    float NdotH = (NdotL > 0.0) ? max(dot(N, H), 0.0) : 0.0;
    // Light shining through the blade from behind brightens it.
    float backlit = clamp(dot(-N, L) * 0.5 + 0.5, 0.0, 1.0);
    float transl = pow(backlit, 2.0) * kGrassTransl;
    vec3 ambient  = kGrassAmbient            * ubo.lightColor;
    vec3 diffuse  = NdotL                    * ubo.lightColor;
    vec3 specular = pow(NdotH, kGrassShine) * kGrassSpecular * ubo.lightColor;
    float shadow = 0.0;
    if (!shadowPass && (ubo.shadowsEnabled != 0u)) {
        if (NdotL > 0.01) {
            float bias = max(0.0005 * (1.0 - NdotL), 0.00005);
            vec4 fragPosLightSpace = ubo.lightSpaceMatrix * vec4(worldPos, 1.0);
            shadow = ShadowCalculationHard(fragPosLightSpace, worldPos, bias);
        } else {
            shadow = 1.0;
        }
    }
    if (!shadowPass && (sky.cloudsEnabled != 0u) && NdotL > 0.01) {
        float cloudShadow = cloudShadowAt(worldPos);
        shadow = 1.0 - (1.0 - shadow) * (1.0 - cloudShadow);
    }
    vec3 lighting = ambient + (diffuse + specular) * (1.0 - shadow);
    return albedo * lighting + albedo * transl * ubo.lightColor * (1.0 - shadow);
}

#endif // GRASS_SHADING_GLSL
