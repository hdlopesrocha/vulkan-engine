// Common helpers for vegetation and impostor shaders.
// Requires perlin2d.glsl and the WindParamsUBO block (declared as `windParams`)
// to be visible before this include.

// Rotate a direction vector around the world-up (Y) axis.
vec3 rotateY(vec3 v, float c, float s) {
    return vec3(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
}

// Shortest-arc tilt mapping +Y onto a (unit) surface normal. Vegetation is
// authored growing along +Y; applying this to the billboard frame makes grass
// grow along the terrain normal instead. Identity for +Y, safe for degenerate
// (zero) normals.
vec3 tiltToNormal(vec3 v, vec3 n) {
    vec3 k = vec3(n.z, 0.0, -n.x); // cross(+Y, n)
    float s = length(k);
    if (s < 1e-4) return (n.y >= 0.0) ? v : vec3(v.x, -v.y, v.z);
    k /= s;
    float c = clamp(n.y, -1.0, 1.0);
    return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c);
}

// Per-instance height variation driven by 2D Perlin noise on world XZ.
// Returns a scale in [0.6, 1.4] that should be applied to billboardScale.
float vegetationHeightScale(vec2 worldXZ) {
    float n = perlin2(worldXZ * 0.008);
    return 0.6 + 0.8 * (n * 0.5 + 0.5);
}

// Distance-based density thinning shared by vegetation and impostor shaders.
// Requires WindParamsUBO (windParams) to be declared before this include.
float densityFactorForDistance(float distanceToCamera) {
    if (!windParams.densityEnabled) return 1.0;

    float nearDistance = max(0.0, windParams.nearDistance);
    float minFactor    = clamp(windParams.minFactor, 0.0, 1.0);
    float falloff      = windParams.densityFalloff;
    if (distanceToCamera <= nearDistance || minFactor >= 1.0 || falloff <= 0.0) {
        return 1.0;
    }

    float density = exp(-falloff * (distanceToCamera - nearDistance));
    return clamp(density, minFactor, 1.0);
}
