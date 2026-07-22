#pragma once
#include "WrappedSignedDistanceEffect.hpp"
#include <glm/glm.hpp>

class WrappedSineDistortDistanceEffect : public WrappedSignedDistanceEffect {
    public:
    float amplitude;
    float frequency;
    glm::vec3 offset;
    WrappedSineDistortDistanceEffect(WrappedSignedDistanceFunction * function_, float amplitude_, float frequency_, glm::vec3 offset_);
    ~WrappedSineDistortDistanceEffect();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    const char* getLabel() const override;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    SdfType getType() const override { return SdfType::DISTORT_SINE; }
};

 
