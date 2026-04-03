// Shared water noise helpers.
// Requires `includes/perlin.glsl` to be included first.

float waterFbmNoise(vec2 xz, float spatialScale, float time, float timeScale,
                    int octaves, float persistence, vec2 offset) {
    return fbm(vec4((xz + offset) * spatialScale, 0.0, time * timeScale), octaves, persistence);
}

float waterWaveDisplacement(vec2 xz, float time,
                            float foamNoiseScale, int foamNoiseOctaves, float foamNoisePersistence,
                            float bumpAmp, float waveScale) {
    float baseNoise = waterFbmNoise(xz, foamNoiseScale * 0.15, time, 0.15,
                                    foamNoiseOctaves, foamNoisePersistence, vec2(0.0));
    float baseNoise2 = waterFbmNoise(xz, foamNoiseScale * 0.07, time, 0.12,
                                     max(foamNoiseOctaves - 1, 1), foamNoisePersistence, vec2(50.0));
    return (baseNoise + baseNoise2 * 0.5) * bumpAmp * waveScale;
}

vec2 waterRefractionNoise(vec2 xz, float noiseScale, float time,
                          int noiseOctaves, float noisePersistence) {
    float noise1 = waterFbmNoise(xz, noiseScale * 0.15, time, 0.4,
                                 noiseOctaves, noisePersistence, vec2(0.0));
    float noise2 = waterFbmNoise(xz, noiseScale * 0.08, time, 0.25,
                                 max(noiseOctaves - 1, 1), noisePersistence, vec2(100.0));
    float noise3 = waterFbmNoise(xz, noiseScale * 0.30, time, 0.6,
                                 max(noiseOctaves - 2, 1), noisePersistence, vec2(0.0));

    float nX = noise1 + noise2 * 0.5 + noise3 * 0.25;
    float nY = waterFbmNoise(xz, noiseScale * 0.15, time, 0.4,
                             noiseOctaves, noisePersistence, vec2(50.0)) + noise2 * 0.5;
    return vec2(nX, nY);
}
