#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/geometry/Transformation.hpp"
#include "../math/geometry/BoundingSphere.hpp"
#include "SDF.hpp"

class BoxDistanceFunction : public SignedDistanceFunction {
public:
    BoxDistanceFunction(const Transformation &model, float bias);
    virtual ~BoxDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
};
