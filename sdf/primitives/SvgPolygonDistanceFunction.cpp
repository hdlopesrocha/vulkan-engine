#include "SvgPolygonDistanceFunction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// Distance from p to the segment [a, b].
inline float segmentDistance(const glm::vec2 &p, const glm::vec2 &a, const glm::vec2 &b) {
    const glm::vec2 ab = b - a;
    const float denom = glm::dot(ab, ab);
    const float t = (denom > 1e-12f) ? glm::clamp(glm::dot(p - a, ab) / denom, 0.0f, 1.0f) : 0.0f;
    return glm::length(p - (a + ab * t));
}

}  // namespace

SvgPolygonDistanceFunction::SvgPolygonDistanceFunction(std::shared_ptr<const SvgShape> shape,
                                                       const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model), m_shape(std::move(shape)) {
    m_half = model.scale;
    m_sphere = getSphere(model, bias);

    if (m_shape) {
        m_scaled.reserve(m_shape->contours.size());
        m_closed.reserve(m_shape->contours.size());
        for (const SvgContour &contour : m_shape->contours) {
            if (contour.points.size() < 2) continue;
            std::vector<glm::vec2> scaled;
            scaled.reserve(contour.points.size());
            for (const glm::vec2 &p : contour.points) {
                scaled.emplace_back(p.x * m_half.x, p.y * m_half.z);
            }
            m_scaled.push_back(std::move(scaled));
            m_closed.push_back(contour.closed && contour.points.size() >= 3 ? 1 : 0);
        }
    }
}

float SvgPolygonDistanceFunction::polygonDistance(const glm::vec2 &p) const {
    float best = std::numeric_limits<float>::max();
    bool inside = false;

    for (size_t c = 0; c < m_scaled.size(); ++c) {
        const std::vector<glm::vec2> &pts = m_scaled[c];
        const size_t n = pts.size();
        const size_t last = m_closed[c] ? n : n - 1;
        for (size_t i = 0; i < last; ++i) {
            const glm::vec2 &a = pts[i];
            const glm::vec2 &b = pts[(i + 1) % n];
            best = std::min(best, segmentDistance(p, a, b));
        }
        if (!m_closed[c]) continue;
        // Even-odd crossing test (holes fall out naturally).
        bool cross = false;
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            const glm::vec2 &a = pts[i];
            const glm::vec2 &b = pts[j];
            if (((a.y > p.y) != (b.y > p.y)) &&
                (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)) {
                cross = !cross;
            }
        }
        if (cross) inside = !inside;
    }
    if (best == std::numeric_limits<float>::max()) return 1.0f;  // no geometry
    return inside ? -best : best;
}

float SvgPolygonDistanceFunction::distance(const glm::vec3 &p) const {
    const glm::vec3 local = localPoint(p);

    // 2D profile distance in world units + the box slab distance along local Y.
    const float d2 = polygonDistance(glm::vec2(local.x, local.z));
    const float dy = std::fabs(local.y) - std::fabs(m_half.y);

    // Box-intersection combine: exact for box ∩ (extruded profile).
    const float outside = std::max(std::max(d2, dy), 0.0f);
    const float inside = std::min(std::max(d2, dy), 0.0f);
    return outside + inside;
}

glm::vec3 SvgPolygonDistanceFunction::localPoint(const glm::vec3 &p) const {
    return glm::inverse(m_model.quaternion) * (p - getCenter());
}

BoundingSphere SvgPolygonDistanceFunction::getSphere(const Transformation &model,
                                                     float bias) const {
    return BoundingSphere(getCenter(), glm::length(model.scale) + bias);
}
