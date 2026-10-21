#version 450

// Video layer fragment shader: samples the material's slot of the dedicated
// video texture array and writes the layer color together with the material's
// blend mode. The post-process composite (Solid -> Water -> Video) decodes the
// alpha channel and applies the blend mode against the composed scene color.
//
// Vertex stage: SolidRendererNoTess.vert (same varyings/interpolants as the
// solid layer, no tessellation because video geometry is not displaced).
//
// The albedo is the current video frame (its only non-default material input).
// Normal, bump/height and roughness/specular are derived from that frame by
// VideoYuvToRgba.comp and packed in the second array (RG = tangent-space
// normal XY, B = height, A = roughness). Lighting uses the material's default
// ambient/specular/shininess/roughness values, exactly like a normal solid
// material; the video layer is then composited with the material's blend mode.

#include "../../includes/SceneBindings.glsl"
#include "../../includes/Locations.glsl"
#include "../../includes/TextureMapping.glsl"

layout(location = VARY_UV) in vec2 fragUV;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;
layout(location = VARY_BRUSHPATCH) flat in ivec3 fragTexIndices;
layout(location = VARY_TEXTURETYPE) flat in ivec3 fragTextureTypes;

layout(set = 1, binding = 0) uniform sampler2DArray videoArray;
// Packed per-frame derived maps (RG normal XY, B bump, A roughness).
layout(set = 1, binding = 1) uniform sampler2DArray videoDerivedArray;

layout(location = 0) out vec4 outColor;

void main() {
    const int brushIndex = fragTexIndices.x;
    const MaterialNamed mat = materialNamed(materials[brushIndex]);
    // textureIndex selects the video slot; -1 keeps the one-material-per-layer
    // behaviour (the material index itself is the slot).
    const int slot = mat.textureIndex >= 0 ? mat.textureIndex : brushIndex;

    // Same coordinate techniques as the solid layer (Triplanar, Projection
    // X/Y/Z, Angular, Custom UV), selected per vertex like the solid path.
    vec2 uvX, uvY, uvZ;
    vec3 weights;
    computeMaterialUVs(fragTextureTypes.x, brushIndex, fragPosWorld, normalize(fragNormal), fragUV,
                       ubo.triplanarThreshold, ubo.triplanarExponent,
                       uvX, uvY, uvZ, weights);
    const vec2 uv = dominantMappingUV(weights, uvX, uvY, uvZ);

    // The conversion compute shader stores linear-space RGBA8, matching the
    // albedo arrays and the linear scene color the composite blends against.
    const vec3 albedo = texture(videoArray, vec3(uv, float(slot))).rgb;

    // Current-frame derived material maps (base level: never mip-filter
    // normals/heights).
    const vec4 derived = textureLod(videoDerivedArray, vec3(uv, float(slot)), 0.0);
    vec2 normalXY = clamp(derived.rg * 2.0 - 1.0, vec2(-1.0), vec2(1.0));
    const float bump = derived.b;
    const float roughness = clamp(derived.a, 0.0, 1.0);
    const float normalZ = sqrt(max(1.0 - dot(normalXY, normalXY), 0.0));
    const vec3 tangentNormal = normalize(vec3(normalXY, max(normalZ, 0.05)));

    // Tangent frame from the UV mapping's screen-space derivatives, so the
    // derived normal works for triplanar, projection and SDF-spherical UVs.
    const vec3 geomN = normalize(fragNormal);
    vec3 dPdx = dFdx(fragPosWorld);
    vec3 dPdy = dFdy(fragPosWorld);
    vec2 dUVdx = dFdx(uv);
    vec2 dUVdy = dFdy(uv);
    const float det = dUVdx.x * dUVdy.y - dUVdy.x * dUVdx.y;
    vec3 tangent;
    if (abs(det) < 1e-6) {
        const vec3 axis = abs(geomN.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        tangent = normalize(cross(axis, geomN));
    } else {
        tangent = normalize((dPdx * dUVdy.y - dPdy * dUVdx.y) / det);
    }
    tangent = normalize(tangent - geomN * dot(geomN, tangent));
    const vec3 bitangent = normalize(cross(geomN, tangent));
    const vec3 shadingN = normalize(tangent * tangentNormal.x +
                                    bitangent * tangentNormal.y +
                                    geomN * tangentNormal.z);

    // Default material lighting (the material only overrides these if the user
    // edits them in the Textures panel).
    const vec3 toLight = normalize(-ubo.lightDirection);
    const float ndl = max(dot(shadingN, toLight), 0.0);
    const vec3 viewDir = normalize(ubo.viewPosition - fragPosWorld);
    const vec3 halfDir = normalize(toLight + viewDir);
    const float specPower = mix(max(mat.shininess, 1.0), 1.0, roughness * mat.roughnessFactor);
    const float specular = pow(max(dot(shadingN, halfDir), 0.0), specPower) * mat.specularStrength;
    // Bump height subtly modulates the ambient term, keeping dark/bright frame
    // regions from flattening the surface relief.
    const float heightTerm = mix(0.85, 1.0, clamp(bump, 0.0, 1.0));
    const vec3 lit = albedo * (mat.ambientFactor * heightTerm + ubo.lightColor * ndl) +
                     ubo.lightColor * specular;

    // Alpha encodes the blend mode: (mode + 1) / 32, 0 means "no video" (the
    // attachment is cleared to 0 and untouched pixels stay transparent to the
    // composite).
    outColor = vec4(lit, (float(mat.blendMode) + 1.0) / 32.0);
}
