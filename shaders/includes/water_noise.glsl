// Shared water noise helpers.
// Requires `includes/perlin.glsl` to be included first.

float waterFbmNoise(vec3 xyz, float spatialScale, float time, float timeScale,
                    int octaves, float persistence, vec3 offset) {
    return fbm(vec4((xyz + offset) * spatialScale, time * timeScale), octaves, persistence);
}

float waterWaveDisplacement(vec3 xyz, float time,
                            float foamNoiseScale, int foamNoiseOctaves, float foamNoisePersistence,
                            float bumpAmp, float waveScale) {
    float baseNoise = waterFbmNoise(xyz, foamNoiseScale * 0.15, time, 0.15,
                                    foamNoiseOctaves, foamNoisePersistence, vec3(0.0));
    float baseNoise2 = waterFbmNoise(xyz, foamNoiseScale * 0.07, time, 0.12,
                                     max(foamNoiseOctaves - 1, 1), foamNoisePersistence, vec3(50.0));
    float signedWave = baseNoise + baseNoise2 * 0.5;
    float wave01 = clamp(signedWave * 0.5 + 0.5, 0.0, 1.0);
    return wave01 * bumpAmp * waveScale;
}

vec2 waterRefractionNoise(vec3 xyz, float noiseScale, float time,
                          int noiseOctaves, float noisePersistence) {
    float noise1 = waterFbmNoise(xyz, noiseScale * 0.15, time, 0.4,
                                 noiseOctaves, noisePersistence, vec3(0.0));
    float noise2 = waterFbmNoise(xyz, noiseScale * 0.08, time, 0.25,
                                 max(noiseOctaves - 1, 1), noisePersistence, vec3(100.0));
    float noise3 = waterFbmNoise(xyz, noiseScale * 0.30, time, 0.6,
                                 max(noiseOctaves - 2, 1), noisePersistence, vec3(0.0));

    float nX = noise1 + noise2 * 0.5 + noise3 * 0.25;
    float nY = waterFbmNoise(xyz, noiseScale * 0.15, time, 0.4,
                             noiseOctaves, noisePersistence, vec3(50.0)) + noise2 * 0.5;
    return vec2(nX, nY);
}
