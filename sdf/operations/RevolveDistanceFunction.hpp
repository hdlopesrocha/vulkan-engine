#pragma once

#include <memory>

#include "../core/SignedDistanceFunction.hpp"
#include "../core/SignedDistance2D.hpp"
#include "../../math/geometry/Transformation.hpp"

// Revolve a normalized 2D profile 360 degrees around the model's local Y axis.
//
// Profile coordinates: x = radial distance, y = axial height. The wrapper model
// maps them to world units (scale.x = radial half extent, scale.y = axial half
// extent). The profile is expected to stay in the x >= 0 half-plane.
class RevolveDistanceFunction : public SignedDistanceFunction {
public:
    RevolveDistanceFunction(std::shared_ptr<const SignedDistance2D> profile,
                            const Transformation &model, float bias);
    ~RevolveDistanceFunction() override = default;

    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    glm::vec3 getCenter() const override { return m_center; }

private:
    std::shared_ptr<const SignedDistance2D> m_profile;
    BoundingSphere m_sphere;
};

