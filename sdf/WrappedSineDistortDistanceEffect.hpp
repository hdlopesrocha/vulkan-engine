#pragma once
#include "WrappedSignedDistanceEffect.hpp"
#include <glm/glm.hpp>

class WrappedSineDistortDistanceEffect : public WrappedSignedDistanceEffect {
    public:
    float amplitude;
    float frequency;
    glm::vec3 offset;
    WrappedSineDistortDistanceEffect(WrappedSignedDistanceFunction * function_, float amplitude_, float frequency_, glm::vec3 offset_, const Transformation &model = Transformation(), float bias = 0.0f);
    ~WrappedSineDistortDistanceEffect();
    const char* getLabel() const override;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override { return SdfType::DISTORT_SINE; }
};

 
