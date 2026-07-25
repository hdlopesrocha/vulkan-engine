#pragma once
#include "SignedDistanceEffect.hpp"
#include <glm/glm.hpp>
#include "SDF.hpp"
#include "../math/Math.hpp"

class PerlinCarveDistanceEffect : public SignedDistanceEffect {
    public:
    float amplitude;
    float frequency;
    float threshold;
    glm::vec3 offset;
    float brightness;
    float contrast;
    PerlinCarveDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float frequency_, float threshold_, glm::vec3 offset_, float brightness_, float contrast_, const Transformation &model, float bias);
    ~PerlinCarveDistanceEffect();
    const char* getLabel() const override;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override { return SdfType::CARVE_PERLIN; }
};
