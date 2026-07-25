#pragma once
#include "SignedDistanceEffect.hpp"
#include <glm/glm.hpp>

class SineDistortDistanceEffect : public SignedDistanceEffect {
    public:
    float amplitude;
    float frequency;
    glm::vec3 offset;
    SineDistortDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float frequency_, glm::vec3 offset_, const Transformation &model, float bias);
    ~SineDistortDistanceEffect();
    const char* getLabel() const override;
    float distance(const glm::vec3 &p) const override;
    SdfType getType() const override { return SdfType::DISTORT_SINE; }
};
