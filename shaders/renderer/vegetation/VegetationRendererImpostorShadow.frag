#version 450

#include "../../ubo/UniformObject.glsl"

// Impostor EVSM2 shadow pass: reconstructs the true blade surface from the
// captured depth array (NOT the quad plane) so far grass casts bladed
// shadows instead of solid quads. Empty capture texels discard.

#include "../../includes/Locations.glsl"
#include "../../includes/shadow/EvsmWrite.glsl"
#include "../../includes/vegetation/NormalTilt.glsl"

layout(location = VARY_POSWORLD) in vec3 inWorldPos; // quad pos (interface only)
// xyz = instance base, w = runtime height scale (hs); see the depth VS.
layout(location = VARY_POSLIGHT) flat in vec4 inInstanceOffset;
layout(location = VARY_UV) in vec3 inTexCoord; // xy=crop UV, z=capture layer
// Instance rotation + terrain surface normal: the reconstruction applies the
// same rotateY(hs * p) + tilt-to-surface transform the blade VS bakes in.
layout(location = VARY_ROTFRAC) flat in vec2 inCosSinT;
layout(location = VARY_PLANE_NORMAL) flat in vec3 inSurfN;

// Canonical scene UBO (set=0 binding=0). In the shadow pass this carries
// the cascade light VP, so projecting the reconstructed point yields the
// light-space depth for the EVSM moments.
layout(std140, set = 0, binding = 0) uniform SolidParamsBlock {
    UniformObject ubo;
};

layout(location = FRAG_OUT_COLOR) out vec2 outEVSM;

// Same depth bindings as the impostor depth descriptor set (bound by the
// shadow draw alongside the cascade set).
layout(set = 1, binding = 0) uniform sampler2DArray depthArray;
layout(set = 1, binding = 1) readonly buffer CaptureInvVP {
    mat4 invVP[];
};

void main() {
    // Direct hand-off at impostorDistance (see impostors_depth.vert): no
    // dithered shadow cross-fade.

    int layer = int(inTexCoord.z);

    // Exact texel fetch: linear filtering would smear blade device-Z with
    // the cleared background at silhouettes (most texels of thin blades
    // are boundary texels), fabricating surfaces for the EVSM moments.
    // The moments themselves are blurred later instead.
    ivec2 texSize = textureSize(depthArray, 0).xy;
    ivec2 texel = ivec2(int(inTexCoord.x * float(texSize.x)),
                        int(inTexCoord.y * float(texSize.y)));
    texel = clamp(texel, ivec2(0), texSize - ivec2(1));
    float texDepth = texelFetch(depthArray, ivec3(texel, layer), 0).r;
    if (texDepth >= 1.0 || texDepth <= 0.0) discard; // no blade at this texel

    // Capture-space blade point -> runtime world: base + tilt(rotateY(hs*p))
    // (the exact instance transform the blade VS applies), so the EVSM
    // moments carry the true blade-surface light depth — scaled, rotated and
    // terrain-tilted — not a flat quad plane.
    vec2 ndc_xy = inTexCoord.xy * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc_xy, texDepth, 1.0);
    vec4 capPos = invVP[layer] * clipPos;
    capPos /= capPos.w;

    vec3 worldPos = inInstanceOffset.xyz + grassTiltToNormal(
        rotateY(capPos.xyz * inInstanceOffset.w, inCosSinT.x, inCosSinT.y),
        normalize(inSurfN));

    vec4 lsPos = ubo.viewProjection * vec4(worldPos, 1.0);
    float depth = clamp(lsPos.z / lsPos.w, 0.0, 1.0);

    gl_FragDepth = depth;

    outEVSM = evsmMoments(depth);
}
