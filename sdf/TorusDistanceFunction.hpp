#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "../math/BoundingSphere.hpp"
#include "SDF.hpp"

class TorusDistanceFunction : public SignedDistanceFunction {
public:
    glm::vec2 radius;
private:
    BoundingSphere sphere;
public:
    TorusDistanceFunction(glm::vec2 radius_, const Transformation &model, float bias);
    virtual ~TorusDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};
