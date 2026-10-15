#version 450
// Impostor capture for bladed grass — stores composite albedo WITHOUT baked
// lighting. Real-time ambient + diffuse + specular is applied in impostors.frag.
// No billboard textures: procedural blade palette (same as VegetationRenderer.frag).

#include "../../includes/Locations.glsl"
#include "../../includes/noise/Perlin.glsl"
#include "../../includes/vegetation/BladeGrass.glsl"

layout(location = VARY_UV) in vec3 inTexCoord;  // x=side, y=heightT, z=variant
layout(location = VARY_BRUSHPATCH) flat in int inBrushIndex;
layout(location = VARY_POSWORLD) in      vec3 inWorldPos;
layout(location = VARY_PLANE_NORMAL) flat in vec3 inPlaneNormal;
layout(location = VARY_POSLIGHT) flat in vec3 inTangentWS;

layout(location = FRAG_OUT_COLOR) out vec4 outColor;
layout(location = FRAG_OUT_NORMAL) out vec4 outNormal; // world-space normal, encoded to [0,1]
layout(location = FRAG_OUT_DEPTH) out float outDepth;  // device Z for shadow-map reprojection

// Canonical capture camera UBO (set=0 binding=0)
#include "../../ubo/CaptureUBO.glsl"

layout(std140, set = 0, binding = 0) uniform CaptureUBOBlock {
    CaptureUBO ubo;
};

// Legacy billboard texture set (set=1) retained in the pipeline layout for
// compatibility but intentionally undeclared: capture is fully procedural.

layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float impostorDistance;
};

void main() {
    float heightT = clamp(inTexCoord.y, 0.0, 1.0);
    float side = inTexCoord.x;
    int variant = inBrushIndex;

    vec3 rootColor, tipColor;
    bladePalette(variant, rootColor, tipColor);
    vec3 albedo = mix(rootColor, tipColor, pow(heightT, 1.25));
    albedo *= 1.0 - 0.18 * clamp(abs(side), 0.0, 1.0);

    // Raw composite albedo — no lighting baked in.
    outColor = vec4(albedo, 1.0);

    // World-space blade normal for real-time lighting in impostors.frag.
    vec3 faceN = gl_FrontFacing ? normalize(inPlaneNormal) : -normalize(inPlaneNormal);
    outNormal = vec4(faceN * 0.5 + 0.5, 1.0);
    outDepth = gl_FragCoord.z;
}
