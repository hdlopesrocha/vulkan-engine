#include "PerlinDistortDistanceEffect.hpp"
#include <stb/stb_perlin.h>

PerlinDistortDistanceEffect::PerlinDistortDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float frequency_, glm::vec3 offset_, float brightness_, float contrast_, const Transformation &model, float bias) : SignedDistanceEffect(function_, model, bias + amplitude_ * 1.97f), amplitude(amplitude_), frequency(frequency_), offset(offset_), brightness(brightness_), contrast(contrast_) {
}

glm::vec3 PerlinDistortDistanceEffect::distortPerlinFractal(const glm::vec3 &p, float frequency, int octaves, float lacunarity, float gain) {
    glm::vec3 totalNoise(0.0f);
    float freq = frequency;
    float amp = 1.0f;

    // Per-axis offsets decorrelate the three noise fields.
    glm::vec3 offsetX = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 offsetY = glm::vec3(100.0f, 100.0f, 100.0f);
    glm::vec3 offsetZ = glm::vec3(200.0f, 200.0f, 200.0f);

    for (int i = 0; i < octaves; ++i) {
        float nx = stb_perlin_noise3(
            p.x * freq + offsetX.x,
            p.y * freq + offsetX.y,
            p.z * freq + offsetX.z,
            0, 0, 0
        );
        float ny = stb_perlin_noise3(
            p.x * freq + offsetY.x,
            p.y * freq + offsetY.y,
            p.z * freq + offsetY.z,
            0, 0, 0
        );
        float nz = stb_perlin_noise3(
            p.x * freq + offsetZ.x,
            p.y * freq + offsetZ.y,
            p.z * freq + offsetZ.z,
            0, 0, 0
        );

        totalNoise += amp * glm::vec3(nx, ny, nz);

        freq *= lacunarity;
        amp *= gain;
    }

    return totalNoise;
}

float PerlinDistortDistanceEffect::distance(const glm::vec3 &p) const {
    glm::vec3 localP = p - m_model.translate;
    glm::vec3 noise = distortPerlinFractal(localP + offset, frequency, 6, 2.0f, 0.5f);
    noise.x = Math::brightnessAndContrast(noise.x, brightness, contrast);
    noise.y = Math::brightnessAndContrast(noise.y, brightness, contrast);
    noise.z = Math::brightnessAndContrast(noise.z, brightness, contrast);

    glm::vec3 newLocalPos = localP + amplitude * noise;
    float d = function.distance(newLocalPos + m_model.translate);

    float maxJacobian = 18.0f * amplitude * frequency;
    float L = 1.0f + maxJacobian;
    return d / L;
}
