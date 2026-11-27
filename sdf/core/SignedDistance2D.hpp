#pragma once

#include <glm/glm.hpp>

// Minimal 2D signed-distance interface used by the extrude/revolve solids.
// Profiles are defined in normalized local units (typically within a unit
// radius); ExtrudeDistanceFunction / RevolveDistanceFunction scale them by the
// wrapper model. No model transform is carried here: a profile is a shape, not
// a placed primitive.
class SignedDistance2D {
public:
    virtual ~SignedDistance2D() = default;
    virtual float distance(const glm::vec2 &p) const = 0;
    // Conservative distance from the profile origin to its farthest point.
    virtual float maxRadius() const = 0;
};
