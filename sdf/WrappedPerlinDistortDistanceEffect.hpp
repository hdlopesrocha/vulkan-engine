#ifndef SDF_WRAPPED_PERLIN_DISTORT_EFFECT_HPP
#define SDF_WRAPPED_PERLIN_DISTORT_EFFECT_HPP

#include "WrappedSignedDistanceEffect.hpp"
#include <glm/glm.hpp>
#include "SDF.hpp"
#include "../math/Math.hpp"

class WrappedPerlinDistortDistanceEffect : public WrappedSignedDistanceEffect {
    public:
    float amplitude;
    float frequency;
    glm::vec3 offset;
    float brightness;
    float contrast;
    WrappedPerlinDistortDistanceEffect(WrappedSignedDistanceFunction * function, float amplitude, float frequency, glm::vec3 offset, float brightness, float contrast);
    ~WrappedPerlinDistortDistanceEffect();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    const char* getLabel() const override;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    SdfType getType() const override { return SdfType::DISTORT_PERLIN; }
};

#endif
