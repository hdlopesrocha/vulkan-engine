// Tessellation displacement function
vec3 applyDisplacement(vec3 localPos, vec3 localNormal, vec2 uv, int texIndex) {
    vec3 displacedLocalPos = localPos;

    // Only apply displacement when mapping mode indicates tessellation
    int mappingMode = int(ubo.mappingParams.x + 0.5);
    if (mappingMode == 2) {
        // Sample height according to material flag and displace outward
        float height = sampleHeight(uv, texIndex);
        // Use per-material tessellation height scale passed in mappingParams.w
        float heightScale = ubo.mappingParams.w;
        displacedLocalPos += localNormal * (height * heightScale);
    }

    return displacedLocalPos;
}