#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "../math/BoundingSphere.hpp"
#include "SDF.hpp"

class CapsuleDistanceFunction : public SignedDistanceFunction {
public:
    glm::vec3 a;
    glm::vec3 b;
    float radius;
private:
    BoundingSphere sphere;
public:
    CapsuleDistanceFunction(glm::vec3 a_, glm::vec3 b_, float r, const Transformation &model, float bias);
    virtual ~CapsuleDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    glm::vec3 getCenter(const Transformation &model) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};
