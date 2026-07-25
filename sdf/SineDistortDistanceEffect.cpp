#include "SineDistortDistanceEffect.hpp"

SineDistortDistanceEffect::SineDistortDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float frequency_, glm::vec3 offset_, const Transformation &model, float bias) : SignedDistanceEffect(function_, model, bias + amplitude_ * 0.5f), amplitude(amplitude_), frequency(frequency_), offset(offset_) {
}

SineDistortDistanceEffect::~SineDistortDistanceEffect() {
}

const char* SineDistortDistanceEffect::getLabel() const {
    return "Sine Distort";
}

float SineDistortDistanceEffect::distance(const glm::vec3 &p) const {
    glm::vec3 localP = p - m_model.translate;
    glm::vec3 pp = localP + offset;

    float dx = sin(pp.x * frequency) * cos(pp.y * frequency) * sin(pp.z * frequency);
    float dy = cos(pp.x * frequency) * sin(pp.y * frequency) * cos(pp.z * frequency);
    float dz = sin(pp.x * frequency) * sin(pp.y * frequency) * cos(pp.z * frequency);

    const float norm = 2.0f;
    glm::vec3 newLocalPos = localP + amplitude / norm * glm::vec3(dx, dy, dz);

    float d = function.distance(newLocalPos + m_model.translate);

    float maxJacobian = 1.5f * amplitude * frequency;
    float L = 1.0f + maxJacobian;
    return d / L;
}
