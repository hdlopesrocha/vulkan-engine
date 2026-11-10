#ifndef SDF_EFFECT_VORONOI_CARVE_GLSL
#define SDF_EFFECT_VORONOI_CARVE_GLSL

// CPU VoronoiCarveDistanceEffect mirror. The CPU uses stb_perlin for cell
// offsets; this integer hash is the documented deterministic GPU equivalent.
float sdfEffectCellRand(ivec3 cell, uint channel) {
    uvec3 c = uvec3(cell) + uvec3(13u, 29u, 43u) * (channel + 1u);
    return float(sdfHashU(c)) * (1.0 / 4294967295.0);
}

float sdfEffectVoronoi3D(vec3 p, float cellSize) {
    if (cellSize <= 0.0) return 0.0;
    vec3 q = p / cellSize;
    q += 0.3 * vec3(perlinNoise3D(q), perlinNoise3D(q.yzx), perlinNoise3D(q.zxy));
    ivec3 baseCell = ivec3(floor(q));
    float minDist = 1e10;
    for (int k = -1; k <= 1; ++k) {
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                ivec3 neighbor = baseCell + ivec3(i, j, k);
                vec3 cellSeed = vec3(neighbor) + vec3(
                    sdfEffectCellRand(neighbor, 0u),
                    sdfEffectCellRand(neighbor, 1u),
                    sdfEffectCellRand(neighbor, 2u));
                minDist = min(minDist, length(q - cellSeed));
            }
        }
    }
    return minDist / sqrt(3.0);
}

#endif // SDF_EFFECT_VORONOI_CARVE_GLSL
