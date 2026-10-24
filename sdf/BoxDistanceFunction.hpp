#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/geometry/Transformation.hpp"
#include "../math/geometry/BoundingSphere.hpp"

class BoxDistanceFunction : public SignedDistanceFunction {
public:
    BoxDistanceFunction(const Transformation &model, float bias);
    virtual ~BoxDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;

    // Axis-aligned box distance in the primitive's local space. Shared with
    // HeightMapDistanceFunction / OctreeDifferenceFunction, which clamp their
    // fields with the same box.
    static float boxDistance(const glm::vec3 &p, const glm::vec3 &len);
};
