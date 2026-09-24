// Tessellation displacement function
vec3 applyDisplacement(vec3 localPos, vec3 localNormal, vec3 worldPos, vec3 worldNormal, vec2 uv, ivec3 texIndices, vec3 weights) {

    MaterialNamed matX = materialNamed(materials[texIndices.x]);
    MaterialNamed matY = materialNamed(materials[texIndices.y]);
    MaterialNamed matZ = materialNamed(materials[texIndices.z]);

    // Compute blended triplanar flag (weighted by barycentric weights)
    float triFlag = float(matX.triplanarEnabled) * weights.x +
                    float(matY.triplanarEnabled) * weights.y +
                    float(matZ.triplanarEnabled) * weights.z;
                    
    // Sample per-material heights (default to neutral 0.5 when a material has mapping disabled)
    float h0 = 0.5;
    float h1 = 0.5;
    float h2 = 0.5;

    // Compute triplanar blend weights once for all three materials (same worldNormal)
    vec3 triplanarWeights = computeTriplanarWeights(worldNormal);

    // Material 0
    if (matX.mappingEnabled) {
        if (matX.triplanarEnabled || triFlag > 0.5) {
            h0 = sampleHeightTriplanarW(worldPos, worldNormal, triplanarWeights, texIndices.x);
        } else {
            h0 = sampleHeight(uv, texIndices.x);
        }
    }
    // Material 1
    if (matY.mappingEnabled) {
        if (matY.triplanarEnabled || triFlag > 0.5) {
            h1 = sampleHeightTriplanarW(worldPos, worldNormal, triplanarWeights, texIndices.y);
        } else {
            h1 = sampleHeight(uv, texIndices.y);
        }
    }
    // Material 2
    if (matZ.mappingEnabled) {
        if (matZ.triplanarEnabled || triFlag > 0.5) {
            h2 = sampleHeightTriplanarW(worldPos, worldNormal, triplanarWeights, texIndices.z);
        } else {
            h2 = sampleHeight(uv, texIndices.z);
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