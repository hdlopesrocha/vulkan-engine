#pragma once
#include "../core/SignedDistanceFunction.hpp"
#include "../../math/geometry/Transformation.hpp"
#include "../../math/geometry/BoundingSphere.hpp"

// Linear interpolation between two signed distance fields:
//   distance(p) = mix(source.distance(p), target.distance(p), factor).
// The operands are referenced, not owned: the caller must keep them alive for
// the lifetime of this wrapper (the brush apply path builds both inside the
// same synchronous call and applies immediately).
//
// The mixed field is a conservative 1-Lipschitz SDF (a convex combination of
// two 1-Lipschitz fields), not an exact Euclidean distance to either surface.
// Its bounding sphere is the union of both operand bounds, so Octree::expand
// and check() never clip a target placed away from the source.
class InterpolatedDistanceFunction : public SignedDistanceFunction {
public:
    InterpolatedDistanceFunction(const SignedDistanceFunction &source,
                                 const SignedDistanceFunction &target,
                                 float factor, const Transformation &model,
                                 float bias);

    float distance(const glm::vec3 &p) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;

private:
    const SignedDistanceFunction &source_;
    const SignedDistanceFunction &target_;
    float factor_ = 0.5f;
};
