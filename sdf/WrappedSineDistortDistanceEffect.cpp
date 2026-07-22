#include "WrappedSineDistortDistanceEffect.hpp"

WrappedSineDistortDistanceEffect::WrappedSineDistortDistanceEffect(WrappedSignedDistanceFunction * function_, float amplitude_, float frequency_, glm::vec3 offset_) : WrappedSignedDistanceEffect(NULL), amplitude(amplitude_), frequency(frequency_), offset(offset_) {
    this->setFunction(function_);
}

WrappedSineDistortDistanceEffect::~WrappedSineDistortDistanceEffect() {

}

const char* WrappedSineDistortDistanceEffect::getLabel() const {
    return "Sine Distort";
}

float WrappedSineDistortDistanceEffect::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 localP = p - model.translate;
    glm::vec3 pp = localP + offset;

    float dx = sin(pp.x * frequency) * cos(pp.y * frequency) * sin(pp.z * frequency);
    float dy = cos(pp.x * frequency) * sin(pp.y * frequency) * cos(pp.z * frequency);
    float dz = sin(pp.x * frequency) * sin(pp.y * frequency) * cos(pp.z * frequency);

    const float norm = 2.0f;
    glm::vec3 newLocalPos = localP + amplitude / norm * glm::vec3(dx, dy, dz);

    float d = function->distance(newLocalPos + model.translate, model);

    float maxJacobian = 1.5f * amplitude * frequency;
    float L = 1.0f + maxJacobian;
    return d / L;
}

ContainmentType WrappedSineDistortDistanceEffect::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    return WrappedSignedDistanceEffect::check(cube, model, bias + amplitude * 0.5f);
};
