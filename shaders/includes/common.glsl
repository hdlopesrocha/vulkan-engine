// Common helper functions

// Sample height helper that respects per-material height interpretation
float sampleHeight(vec2 texCoords, int texIndex) {
    // Sample the height from the height texture array (binding = 3)
    // texIndex selects the layer in the array. If the material requests
    // inverted heights (mappingParams.z), invert the sampled value.
    float h = texture(heightArray, vec3(texCoords, float(texIndex))).r;
    if (ubo.mappingParams.z > 0.5) {
        h = 1.0 - h;
    }
    return clamp(h, 0.0, 1.0);
}