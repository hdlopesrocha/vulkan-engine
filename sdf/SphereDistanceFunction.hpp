#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>

class SphereDistanceFunction : public SignedDistanceFunction {
public:
    SphereDistanceFunction();
    virtual ~SphereDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override;
    glm::vec3 getCenter(const Transformation &model) const override;
};

 
