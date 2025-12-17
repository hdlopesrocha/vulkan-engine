// Tessellation displacement function
vec3 applyDisplacement(vec3 localPos, vec3 localNormal, vec3 worldPos, vec3 worldNormal, vec2 uv, ivec3 texIndices, vec3 weights) {

    // Compute blended triplanar flag (weighted by barycentric weights)
    float triFlag = materials[texIndices.x].triplanarParams.z * weights.x +
                    materials[texIndices.y].triplanarParams.z * weights.y +
                    materials[texIndices.z].triplanarParams.z * weights.z;

    // Sample per-material heights (default to neutral 0.5 when a material has mapping disabled)
    float h0 = 0.5;
    float h1 = 0.5;
    float h2 = 0.5;

    // Material 0
    if (materials[texIndices.x].mappingParams.x > 0.5) {
        if (materials[texIndices.x].triplanarParams.z > 0.5 || triFlag > 0.5) {
            h0 = sampleHeightTriplanar(worldPos, worldNormal, texIndices.x);
        } else {
            h0 = sampleHeight(uv, texIndices.x);
        }
    }
    // Material 1
    if (materials[texIndices.y].mappingParams.x > 0.5) {
        if (materials[texIndices.y].triplanarParams.z > 0.5 || triFlag > 0.5) {
            h1 = sampleHeightTriplanar(worldPos, worldNormal, texIndices.y);
        } else {
            h1 = sampleHeight(uv, texIndices.y);
        }
    }
    // Material 2
    if (materials[texIndices.z].mappingParams.x > 0.5) {
        if (materials[texIndices.z].triplanarParams.z > 0.5 || triFlag > 0.5) {
            h2 = sampleHeightTriplanar(worldPos, worldNormal, texIndices.z);
        } else {
            h2 = sampleHeight(uv, texIndices.z);
        }
    }

    // Blend heights and per-material height scales
    float h = h0 * weights.x + h1 * weights.y + h2 * weights.z;
    float scale = materials[texIndices.x].mappingParams.w * weights.x +
                  materials[texIndices.y].mappingParams.w * weights.y +
                  materials[texIndices.z].mappingParams.w * weights.z;

    // Convert to signed displacement centered at 0.5 and apply scale
    float displacement = (h - 0.5) * scale;

    // Move local position along the local-space worldNormal
    return localPos + localNormal * displacement;
}