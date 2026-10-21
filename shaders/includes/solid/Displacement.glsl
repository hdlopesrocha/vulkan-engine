
#include "../../types/MaterialNamed.glsl"
#include "../TextureMapping.glsl"
// Tessellation displacement function
vec3 applyDisplacement(vec3 localPos, vec3 localNormal, vec3 worldPos, vec3 worldNormal, vec2 uv, ivec3 texIndices, ivec3 texTypes, vec3 weights) {

    MaterialNamed matX = materialNamed(materials[texIndices.x]);
    MaterialNamed matY = materialNamed(materials[texIndices.y]);
    MaterialNamed matZ = materialNamed(materials[texIndices.z]);

    // Sample per-material heights (default to neutral 0.5 when a material has mapping disabled)
    float h0 = 0.5;
    float h1 = 0.5;
    float h2 = 0.5;

    // Compute triplanar blend weights once for all three materials (same worldNormal)
    vec3 triplanarWeights = computeTriplanarWeights(worldNormal);

    // Material 0. Triplanar materials keep the world-space triplanar fetch;
    // every other mapping mode samples at the UV of its dominant axis.
    if (matX.mappingEnabled) {
        if (mappingUsesTriplanar(texTypes.x, matX)) {
            h0 = sampleHeightTriplanarW(worldPos, worldNormal, triplanarWeights, texIndices.x);
        } else {
            vec2 uvX, uvY, uvZ;
            vec3 mappingWeights;
            computeMaterialUVs(texTypes.x, texIndices.x, worldPos, worldNormal, uv, ubo.triplanarThreshold, ubo.triplanarExponent, uvX, uvY, uvZ, mappingWeights);
            h0 = sampleHeight(dominantMappingUV(mappingWeights, uvX, uvY, uvZ), texIndices.x);
        }
    }
    // Material 1
    if (matY.mappingEnabled) {
        if (mappingUsesTriplanar(texTypes.y, matY)) {
            h1 = sampleHeightTriplanarW(worldPos, worldNormal, triplanarWeights, texIndices.y);
        } else {
            vec2 uvX, uvY, uvZ;
            vec3 mappingWeights;
            computeMaterialUVs(texTypes.y, texIndices.y, worldPos, worldNormal, uv, ubo.triplanarThreshold, ubo.triplanarExponent, uvX, uvY, uvZ, mappingWeights);
            h1 = sampleHeight(dominantMappingUV(mappingWeights, uvX, uvY, uvZ), texIndices.y);
        }
    }
    // Material 2
    if (matZ.mappingEnabled) {
        if (mappingUsesTriplanar(texTypes.z, matZ)) {
            h2 = sampleHeightTriplanarW(worldPos, worldNormal, triplanarWeights, texIndices.z);
        } else {
            vec2 uvX, uvY, uvZ;
            vec3 mappingWeights;
            computeMaterialUVs(texTypes.z, texIndices.z, worldPos, worldNormal, uv, ubo.triplanarThreshold, ubo.triplanarExponent, uvX, uvY, uvZ, mappingWeights);
            h2 = sampleHeight(dominantMappingUV(mappingWeights, uvX, uvY, uvZ), texIndices.z);
        }
    }

    // Blend heights and per-material height scales
    float h = h0 * weights.x + h1 * weights.y + h2 * weights.z;
    float scale = matX.tessHeightScale * weights.x +
                  matY.tessHeightScale * weights.y +
                  matZ.tessHeightScale * weights.z;

    // Convert to signed displacement centered at 0.5 and apply scale
    float displacement = (h - 0.5) * scale;

    // Move local position along the local-space worldNormal
    return localPos + localNormal * displacement;
}