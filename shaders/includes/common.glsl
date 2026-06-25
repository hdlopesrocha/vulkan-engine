// Common helper functions

// Sample height helper that respects per-material height interpretation
float sampleHeight(vec2 texCoords, int brushIndex) {
    vec2 tc = texCoords;
    if (materials[brushIndex].mappingParams.z > 0.5) {
        tc.y = 1.0 - tc.y;
    }
    if (materials[brushIndex].normalParams.z > 0.5) {
        tc.x = 1.0 - tc.x;
    }
    float h = texture(heightArray, vec3(tc, float(brushIndex))).r;
    return clamp(h, 0.0, 1.0);
}

// Triplanar height sampling: blend height from world-space projected UVs
float sampleHeightTriplanar(vec3 worldPos, vec3 normal, int brushIndex) {
    // Calculate blend weights from world normal using global triplanar settings (threshold + exponent)
    vec3 w = abs(normal);
    // apply threshold (dead-zone) and exponent (steepness)
    float t = ubo.triplanarSettings.x; // threshold
    vec3 wt = max(vec3(0.0), w - vec3(t));
    float e = max(1.0, ubo.triplanarSettings.y);
    wt = pow(wt, vec3(e));
    float sum = wt.x + wt.y + wt.z + 1e-6;
    w = wt / sum;

    vec2 scale = vec2(materials[brushIndex].triplanarParams.x, materials[brushIndex].triplanarParams.y);
    if (materials[brushIndex].mappingParams.z > 0.5) {
        scale.y = -scale.y;
    }
    if (materials[brushIndex].normalParams.z > 0.5) {
        scale.x = -scale.x;
    }

    float hX = 0.0;
    float hY = 0.0;
    float hZ = 0.0;

    // UV math must match computeTriplanarUVs() in triplanar.glsl so height and albedo align
    if (w.x > 0.0) {
        vec2 uvX = (normal.x >= 0.0) ? vec2(-worldPos.z, -worldPos.y) : vec2(worldPos.z, -worldPos.y);
        hX = texture(heightArray, vec3(uvX * scale, float(brushIndex))).r;
    }

    if (w.y > 0.0) {
        vec2 uvY = (normal.y >= 0.0) ? vec2(worldPos.x, worldPos.z) : vec2(worldPos.x, -worldPos.z);
        hY = texture(heightArray, vec3(uvY * scale, float(brushIndex))).r;
    }

    if (w.z > 0.0) {
        vec2 uvZ = (normal.z >= 0.0) ? vec2(worldPos.x, -worldPos.y) : vec2(-worldPos.x, -worldPos.y);
        hZ = texture(heightArray, vec3(uvZ * scale, float(brushIndex))).r;
    }

    float h = hX * w.x + hY * w.y + hZ * w.z;
    return clamp(h, 0.0, 1.0);
}