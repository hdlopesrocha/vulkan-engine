#include "PerlinCarveDistanceEffect.hpp"

PerlinCarveDistanceEffect::PerlinCarveDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float frequency_, float threshold_, glm::vec3 offset_, float brightness_, float contrast_, const Transformation &model, float bias) : SignedDistanceEffect(function_, model, bias + amplitude_ * 1.97f), amplitude(amplitude_), frequency(frequency_), threshold(threshold_), offset(offset_), brightness(brightness_), contrast(contrast_) {

}

PerlinCarveDistanceEffect::~PerlinCarveDistanceEffect() {
}

const char* PerlinCarveDistanceEffect::getLabel() const {
    return "Perlin Carve";
}

float PerlinCarveDistanceEffect::distance(const glm::vec3 &p) const {
    float d = function.distance(p);
    glm::vec3 localP = p - m_model.translate;
    float noise = SDF::distortedCarveFractalSDF(localP + offset, threshold, frequency, 6, 2.0f, 0.5f);
    noise = Math::brightnessAndContrast(noise, brightness, contrast);

    float carved = d + noise * amplitude;

    float maxGrad = amplitude * frequency * 6.0f;
    float L = 1.0f + maxGrad;
    return carved / L;
}
