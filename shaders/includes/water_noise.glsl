// Shared water noise helpers.
// Requires `includes/perlin.glsl` to be included first.

float waterFbmNoise(vec3 xyz, float spatialScale, float time, float timeScale,
                    int octaves, float persistence, float lacunarity, vec3 offset) {
    return fbm(vec4((xyz + offset) * spatialScale, time * timeScale), octaves, persistence, lacunarity);
}

float waterWaveDisplacement(vec3 xyz, float time,
                             float foamNoiseScale, int foamNoiseOctaves, float foamNoisePersistence,
                             float foamNoiseLacunarity,
                             float bumpAmp, float waveScale) {
    float baseNoise = waterFbmNoise(xyz, foamNoiseScale * 0.15, time, 0.15,
                                    foamNoiseOctaves, foamNoisePersistence, foamNoiseLacunarity, vec3(0.0));
    float baseNoise2 = waterFbmNoise(xyz, foamNoiseScale * 0.07, time, 0.12,
                                     max(foamNoiseOctaves - 1, 1), foamNoisePersistence, foamNoiseLacunarity, vec3(50.0));
    float signedWave = baseNoise + baseNoise2 * 0.5;
    float wave01 = clamp(signedWave * 0.5 + 0.5, 0.0, 1.0);
    return wave01 * bumpAmp * waveScale;
}

// Gradient-aware variant of waterWaveDisplacement.  Returns
// vec4(height, dHeight/dx, dHeight/dy, dHeight/dz) from a SINGLE noise
// evaluation (two FBMs) instead of the 5 evaluations the central-difference
// scheme needs.  The analytic spatial gradient is propagated through the FBM
// octave chain and through the final clamp() (whose slope is 0.5 inside the
// active range and 0 at the saturating ends).
vec4 waterFbmNoiseGrad(vec3 xyz, float spatialScale, float time, float timeScale,
                       int octaves, float persistence, float lacunarity, vec3 offset) {
    vec4 r = fbmGrad4D(vec4((xyz + offset) * spatialScale, time * timeScale),
                       octaves, persistence, lacunarity);
    r.yzw *= spatialScale; // chain rule through the spatial scaling
    return r;
}

vec4 waterWaveSample(vec3 xyz, float time,
                     float foamNoiseScale, int foamNoiseOctaves, float foamNoisePersistence,
                     float foamNoiseLacunarity,
                     float bumpAmp, float waveScale) {
    vec4 baseNoise  = waterFbmNoiseGrad(xyz, foamNoiseScale * 0.15, time, 0.15,
                                        foamNoiseOctaves, foamNoisePersistence, foamNoiseLacunarity, vec3(0.0));
    vec4 baseNoise2 = waterFbmNoiseGrad(xyz, foamNoiseScale * 0.07, time, 0.12,
                                        max(foamNoiseOctaves - 1, 1), foamNoisePersistence, foamNoiseLacunarity, vec3(50.0));
    float signedWave = baseNoise.x + baseNoise2.x * 0.5;
    vec3  gWave = baseNoise.yzw + baseNoise2.yzw * 0.5;

    // Chain rule through clamp(): slope 0.5 inside [-1,1], 0 outside.
    float clampSlope = (signedWave > -1.0 && signedWave < 1.0) ? 0.5 : 0.0;
    float wave01 = clamp(signedWave * 0.5 + 0.5, 0.0, 1.0);

    float height = wave01 * bumpAmp * waveScale;
    vec3  grad = gWave * clampSlope * bumpAmp * waveScale;
    return vec4(height, grad);
}

vec2 waterRefractionNoise(vec3 xyz, float noiseScale, float time,
                          int noiseOctaves, float noisePersistence, float noiseLacunarity) {
    float noise1 = waterFbmNoise(xyz, noiseScale * 0.15, time, 0.4,
                                 noiseOctaves, noisePersistence, noiseLacunarity, vec3(0.0));
    float noise2 = waterFbmNoise(xyz, noiseScale * 0.08, time, 0.25,
                                 max(noiseOctaves - 1, 1), noisePersistence, noiseLacunarity, vec3(100.0));
    float noise3 = waterFbmNoise(xyz, noiseScale * 0.30, time, 0.6,
                                 max(noiseOctaves - 2, 1), noisePersistence, noiseLacunarity, vec3(0.0));

    float nX = noise1 + noise2 * 0.5 + noise3 * 0.25;
    float nY = waterFbmNoise(xyz, noiseScale * 0.15, time, 0.4,
                             noiseOctaves, noisePersistence, noiseLacunarity, vec3(50.0)) + noise2 * 0.5;
    return vec2(nX, nY);
}
