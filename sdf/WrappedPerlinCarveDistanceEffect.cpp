#include "WrappedPerlinCarveDistanceEffect.hpp"

WrappedPerlinCarveDistanceEffect::WrappedPerlinCarveDistanceEffect(WrappedSignedDistanceFunction * function, float amplitude, float frequency, float threshold, glm::vec3 offset, float brightness, float contrast) : WrappedSignedDistanceEffect(function), amplitude(amplitude), frequency(frequency), threshold(threshold), offset(offset), brightness(brightness), contrast(contrast) {

}

WrappedPerlinCarveDistanceEffect::~WrappedPerlinCarveDistanceEffect() {

}

const char* WrappedPerlinCarveDistanceEffect::getLabel() const {
    return "Perlin Carve";
}

float WrappedPerlinCarveDistanceEffect::distance(const glm::vec3 &p, const Transformation &model) {
    float d = function->distance(p, model);
    glm::vec3 localP = p - model.translate;
    float noise = SDF::distortedCarveFractalSDF(localP + offset, threshold, frequency, 6, 2.0f, 0.5f);
    noise = Math::brightnessAndContrast(noise, brightness, contrast);

    float carved = d + noise * amplitude;

    float maxGrad = amplitude * frequency * 6.0f;
    float L = 1.0f + maxGrad;
    return carved / L;
}

ContainmentType WrappedPerlinCarveDistanceEffect::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    return WrappedSignedDistanceEffect::check(cube, model, bias + amplitude * 1.97f);
};