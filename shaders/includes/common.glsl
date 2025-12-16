// Common helper functions

// Sample height helper that respects per-material height interpretation
float sampleHeight(vec2 texCoords, int texIndex) {
    // Sample the height from the height texture array (binding = 3)
    // texIndex selects the layer in the array. If the material requests
    // inverted heights (mappingParams.z), invert the sampled value.
    float h = texture(heightArray, vec3(texCoords, float(texIndex))).r;
    if (materials[texIndex].mappingParams.z > 0.5) {
        h = 1.0 - h;
    }
    return clamp(h, 0.0, 1.0);
}

// Triplanar height sampling: blend height from world-space projected UVs
float sampleHeightTriplanar(vec3 worldPos, vec3 normal, int texIndex) {
    // Build three projection UVs using triplanar scale from UBO
    vec2 uvX = worldPos.yz * vec2(materials[texIndex].triplanarParams.x, materials[texIndex].triplanarParams.y);
    vec2 uvY = worldPos.xz * vec2(materials[texIndex].triplanarParams.x, materials[texIndex].triplanarParams.y);
    vec2 uvZ = worldPos.xy * vec2(materials[texIndex].triplanarParams.x, materials[texIndex].triplanarParams.y);
    // Calculate blend weights from world normal
    vec3 w = abs(normal);
    w = w * w;
    float sum = w.x + w.y + w.z + 1e-6;
    w /= sum;
    // Sample heights
    float hX = texture(heightArray, vec3(uvX, float(texIndex))).r;
    float hY = texture(heightArray, vec3(uvY, float(texIndex))).r;
    float hZ = texture(heightArray, vec3(uvZ, float(texIndex))).r;
    float h = hX * w.x + hY * w.y + hZ * w.z;
    if (materials[texIndex].mappingParams.z > 0.5) {
        h = 1.0 - h;
    }
    return clamp(h, 0.0, 1.0);
}