#version 450

// Video layer fragment shader: samples the material's slot of the dedicated
// video texture array and writes the layer color together with the material's
// blend mode. The post-process composite (Solid -> Water -> Video) decodes the
// alpha channel and applies the blend mode against the composed scene color.
//
// Vertex stage: SolidRendererNoTess.vert (same varyings/interpolants as the
// solid layer, no tessellation because video geometry is not displaced).

#include "../../includes/SceneBindings.glsl"
#include "../../includes/Locations.glsl"
#include "../../includes/TextureMapping.glsl"

layout(location = VARY_UV) in vec2 fragUV;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;
layout(location = VARY_BRUSHPATCH) flat in ivec3 fragTexIndices;

layout(set = 1, binding = 0) uniform sampler2DArray videoArray;

layout(location = 0) out vec4 outColor;

void main() {
    const int brushIndex = fragTexIndices.x;
    const MaterialNamed mat = materialNamed(materials[brushIndex]);
    // textureIndex selects the video slot; -1 keeps the one-material-per-layer
    // behaviour (the material index itself is the slot).
    const int slot = mat.textureIndex >= 0 ? mat.textureIndex : brushIndex;

    // Same coordinate techniques as the solid layer (Triplanar, Projection
    // X/Y/Z, SDF Given, Custom UV). Video materials normally use one of the
    // non-triplanar modes, but the weighted path is supported too.
    vec2 uvX, uvY, uvZ;
    vec3 weights;
    computeMaterialUVs(brushIndex, fragPosWorld, normalize(fragNormal), fragUV,
                       ubo.triplanarThreshold, ubo.triplanarExponent,
                       uvX, uvY, uvZ, weights);
    const vec2 uv = dominantMappingUV(weights, uvX, uvY, uvZ);

    // The conversion compute shader stores linear-space RGBA8, matching the
    // albedo arrays and the linear scene color the composite blends against.
    vec3 color = texture(videoArray, vec3(uv, float(slot))).rgb;

    // Alpha encodes the blend mode: (mode + 1) / 32, 0 means "no video" (the
    // attachment is cleared to 0 and untouched pixels stay transparent to the
    // composite).
    outColor = vec4(color, (float(mat.blendMode) + 1.0) / 32.0);
}
