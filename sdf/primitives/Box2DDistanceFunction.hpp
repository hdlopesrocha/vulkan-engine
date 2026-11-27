#pragma once

#include "../core/SignedDistance2D.hpp"

// Normalized 2D box profile. `distance()` is the exact axis-aligned box SDF.
// Used by the extrude/revolve catalogue examples; the wrapper model supplies
// the world size.
class Box2DDistanceFunction : public SignedDistance2D {
public:
    Box2DDistanceFunction(const glm::vec2 &center, const glm::vec2 &halfExtents)
        : m_center(center), m_half(glm::max(halfExtents, glm::vec2(1.0e-4f))) {}

    float distance(const glm::vec2 &p) const override {
        glm::vec2 q = glm::abs(p - m_center) - m_half;
        return glm::length(glm::max(q, glm::vec2(0.0f))) +
               glm::min(glm::max(q.x, q.y), 0.0f);
    }
    float maxRadius() const override {
        return glm::length(m_center) + glm::length(m_half);
    }

private:
    glm::vec2 m_center{0.0f};
    glm::vec2 m_half{1.0f};
};
