#pragma once

#include "../core/SignedDistance2D.hpp"

// Normalized 2D circle profile (origin, radius). Used by the extrude/revolve
// catalogue examples; the wrapper model supplies the world size.
class Circle2DDistanceFunction : public SignedDistance2D {
public:
    explicit Circle2DDistanceFunction(float radius = 1.0f)
        : m_radius(radius > 0.0f ? radius : 1.0f) {}

    float distance(const glm::vec2 &p) const override {
        return glm::length(p) - m_radius;
    }
    float maxRadius() const override { return m_radius; }

private:
    float m_radius = 1.0f;
};
