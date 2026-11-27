#pragma once

#include <memory>

#include "../core/SignedDistanceFunction.hpp"
#include "../core/SignedDistance2D.hpp"
#include "../../math/geometry/Transformation.hpp"

// Extrude a normalized 2D profile along the model's local Z axis.
//
// The wrapper model supplies the half extents: scale.xy scale the profile,
// scale.z is the extrusion half-thickness. The profile itself is expected to
// be centred at the origin and have maxRadius() about 1.
class ExtrudeDistanceFunction : public SignedDistanceFunction {
public:
    ExtrudeDistanceFunction(std::shared_ptr<const SignedDistance2D> profile,
                            const Transformation &model, float bias);
    ~ExtrudeDistanceFunction() override = default;

    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    glm::vec3 getCenter() const override { return m_center; }

private:
    std::shared_ptr<const SignedDistance2D> m_profile;
    BoundingSphere m_sphere;
};

