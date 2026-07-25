#include "PerlinDistortDistanceEffect.hpp"

PerlinDistortDistanceEffect::PerlinDistortDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float frequency_, glm::vec3 offset_, float brightness_, float contrast_, const Transformation &model, float bias) : SignedDistanceEffect(function_, model, bias + amplitude_ * 1.97f), amplitude(amplitude_), frequency(frequency_), offset(offset_), brightness(brightness_), contrast(contrast_) {
}

PerlinDistortDistanceEffect::~PerlinDistortDistanceEffect() {
}

const char* PerlinDistortDistanceEffect::getLabel() const {
    return "Perlin Distort";
}

float PerlinDistortDistanceEffect::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 localP = p - model.translate;
    glm::vec3 noise = SDF::distortPerlinFractal(localP + offset, frequency, 6, 2.0f, 0.5f);
    noise.x = Math::brightnessAndContrast(noise.x, brightness, contrast);
    noise.y = Math::brightnessAndContrast(noise.y, brightness, contrast);
    noise.z = Math::brightnessAndContrast(noise.z, brightness, contrast);

    glm::vec3 newLocalPos = localP + amplitude * noise;
    float d = function.distance(newLocalPos + model.translate, model);

    float maxJacobian = 18.0f * amplitude * frequency;
    float L = 1.0f + maxJacobian;
    return d / L;
}
