#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "../math/BoundingSphere.hpp"
#include "SDF.hpp"

class PyramidDistanceFunction : public SignedDistanceFunction {
private:
    BoundingSphere sphere;
public:
    PyramidDistanceFunction(const Transformation &model, float bias);
    virtual ~PyramidDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) const override;
    float boundingSphereRadius(float width, float depth, float height) const;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};
