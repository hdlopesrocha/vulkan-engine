#ifndef SDF_WRAPPED_PERLIN_CARVE_EFFECT_HPP
#define SDF_WRAPPED_PERLIN_CARVE_EFFECT_HPP

#include "WrappedSignedDistanceEffect.hpp"
#include <glm/glm.hpp>

class WrappedPerlinCarveDistanceEffect : public WrappedSignedDistanceEffect {
    public:
    float amplitude;
    float frequency;
    float threshold;
    glm::vec3 offset;
    float brightness;
    float contrast;
    WrappedPerlinCarveDistanceEffect(WrappedSignedDistanceFunction * function, float amplitude, float frequency, float threshold, glm::vec3 offset, float brightness, float contrast);
    ~WrappedPerlinCarveDistanceEffect();
    const char* getLabel() const override;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    SdfType getType() const override { return SdfType::CARVE_PERLIN; }
};

#endif
