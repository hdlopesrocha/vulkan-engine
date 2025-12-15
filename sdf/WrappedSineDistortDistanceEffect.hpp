#ifndef SDF_WRAPPED_SINE_DISTORT_EFFECT_HPP
#define SDF_WRAPPED_SINE_DISTORT_EFFECT_HPP

#include "WrappedSignedDistanceEffect.hpp"
#include <glm/glm.hpp>

class WrappedSineDistortDistanceEffect : public WrappedSignedDistanceEffect {
    public:
    float amplitude;
    float frequency;
    glm::vec3 offset;
    WrappedSineDistortDistanceEffect(WrappedSignedDistanceFunction * function, float amplitude, float frequency, glm::vec3 offset);
    ~WrappedSineDistortDistanceEffect();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    const char* getLabel() const override;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override { return SdfType::DISTORT_SINE; }
};

#endif
