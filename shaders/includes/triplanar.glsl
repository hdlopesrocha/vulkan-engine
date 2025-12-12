// Triplanar mapping helpers
// Uses `ubo.triplanarParams`: xy = scaleU/scaleV, z = enabled (1.0)

vec3 triplanarBlendAlbedo(int texIndex, vec3 pos, vec3 n) {
    vec3 absN = abs(n);
    absN = max(absN, vec3(1e-4));
    float sum = absN.x + absN.y + absN.z;
    vec3 w = absN / sum;
    vec2 scale = ubo.triplanarParams.xy;
    vec2 uvX = pos.yz * scale;
    vec2 uvY = pos.zx * scale;
    vec2 uvZ = pos.xy * scale;
    vec3 cX = texture(albedoArray, vec3(uvX, float(texIndex))).rgb;
    vec3 cY = texture(albedoArray, vec3(uvY, float(texIndex))).rgb;
    vec3 cZ = texture(albedoArray, vec3(uvZ, float(texIndex))).rgb;
    return cX * w.x + cY * w.y + cZ * w.z;
}

vec3 triplanarBlendNormal(int texIndex, vec3 pos, vec3 n) {
    vec3 absN = abs(n);
    absN = max(absN, vec3(1e-4));
    float sum = absN.x + absN.y + absN.z;
    vec3 w = absN / sum;
    vec2 scale = ubo.triplanarParams.xy;
    vec2 uvX = pos.yz * scale;
    vec2 uvY = pos.zx * scale;
    vec2 uvZ = pos.xy * scale;
    vec3 nX = texture(normalArray, vec3(uvX, float(texIndex))).rgb * 2.0 - 1.0;
    vec3 nY = texture(normalArray, vec3(uvY, float(texIndex))).rgb * 2.0 - 1.0;
    vec3 nZ = texture(normalArray, vec3(uvZ, float(texIndex))).rgb * 2.0 - 1.0;
    // Approximate blended normal (not strictly correct for tangent-space normal maps,
    // but provides reasonable detail when triplanar mapping is enabled)
    vec3 blended = normalize(nX * w.x + nY * w.y + nZ * w.z);
    return blended;
}
