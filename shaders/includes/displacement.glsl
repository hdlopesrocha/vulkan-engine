// Tessellation displacement function
vec3 applyDisplacement(vec3 localPos, vec3 localNormal, vec2 uv, int texIndex) {
    vec3 displacedLocalPos = localPos;

    // Only apply displacement when mapping mode indicates tessellation
    // mappingParams.x is now a boolean toggle: >0.5 enables tessellation displacement
        if (materials[texIndex].mappingParams.x > 0.5) {
        // Choose sampling method: triplanar if enabled, otherwise regular UV
        vec3 worldPosForSampling = (pushConstants.model * vec4(localPos, 1.0)).xyz;
        float height = 0.0;
            if (materials[texIndex].triplanarParams.z > 0.5) {
                height = sampleHeightTriplanar(worldPosForSampling, localNormal, texIndex);
            } else {
                height = sampleHeight(uv, texIndex);
            }
            // Use per-material tessellation height scale passed in mappingParams.w
            float heightScale = materials[texIndex].mappingParams.w;
        // Displace outward along surface normal based on sampled height (relative to surface)
        float d = height * heightScale;
        displacedLocalPos += localNormal * d;
    }

    return displacedLocalPos;
}