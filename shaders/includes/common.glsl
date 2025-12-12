// Common helper functions

// Sample height helper that respects per-material height interpretation
float sampleHeight(vec2 texCoords, int texIndex) {
    float raw = texture(heightArray, vec3(texCoords, float(texIndex))).r;
    // mappingParams.z == 1.0 means height is direct (white=high)
    return (ubo.mappingParams.z > 0.5) ? raw : 1.0 - raw;
}