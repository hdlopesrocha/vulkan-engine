#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "../math/BoundingSphere.hpp"
#include "SDF.hpp"

class OctahedronDistanceFunction : public SignedDistanceFunction {
public:
    OctahedronDistanceFunction(const Transformation &model, float bias);
    virtual ~OctahedronDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
};
