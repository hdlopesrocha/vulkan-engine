#pragma once

#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "SignedDistanceFunction.hpp"
#include "SvgShape.hpp"

// SVG profile extruded along the model's local Y axis and intersected with the
// brush's box ("cookie-cutter" SDF brush).
//
// SvgShape normalizes the drawing into [-1,1]^2; here it is mapped onto the
// model's local XZ plane with the model's half extents (model.scale) and
// extruded along the local Y axis, so the resulting solid is the SVG
// silhouette clipped to the box [-scale, +scale] — i.e. the intersection of
// the extrusion with that box. distance() is a real (approximate but
// Lipschitz-safe) distance in world units, computed from the scaled polygon
// edges with the sign taken from the even-odd fill rule.
//
// Local profile coordinates (used by SvgBrush for per-contour colours) are
// p_local = (R^-1 (p - center)) / scale.
class SvgPolygonDistanceFunction : public SignedDistanceFunction {
public:
    SvgPolygonDistanceFunction(std::shared_ptr<const SvgShape> shape,
                               const Transformation &model, float bias);
    ~SvgPolygonDistanceFunction() override = default;

    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;

    const std::shared_ptr<const SvgShape> &shape() const { return m_shape; }
    glm::vec3 halfExtents() const { return m_half; }
    glm::vec3 localPoint(const glm::vec3 &p) const;

private:
    float polygonDistance(const glm::vec2 &p) const;

    std::shared_ptr<const SvgShape> m_shape;
    // Contours scaled into world units (x*half.x, z*half.z).
    std::vector<std::vector<glm::vec2>> m_scaled;
    std::vector<char> m_closed;
    glm::vec3 m_half = glm::vec3(1.0f);
};
