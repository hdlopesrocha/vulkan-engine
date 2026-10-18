#include "SvgBrush.hpp"

#include <cmath>

SvgBrush::SvgBrush(std::shared_ptr<const SvgShape> shape, const Transformation &model,
                   int materialIndex, glm::vec3 fallbackHsv, bool tintFromFill)
    : m_shape(std::move(shape))
    , m_model(model)
    , m_material(materialIndex)
    , m_fallbackHsv(fallbackHsv)
    , m_tintFromFill(tintFromFill) {}

int SvgBrush::paint(const Vertex &vertex) const {
    (void)vertex;
    return m_material;
}

glm::vec3 SvgBrush::paintHSV(const Vertex &vertex) const {
    if (!m_tintFromFill || !m_shape) return m_fallbackHsv;

    const glm::vec3 local =
        glm::inverse(m_model.quaternion) * (vertex.position - m_model.translate);
    const float sx = (std::fabs(m_model.scale.x) > 1e-6f) ? m_model.scale.x : 1.0f;
    const float sz = (std::fabs(m_model.scale.z) > 1e-6f) ? m_model.scale.z : 1.0f;
    const int contour = m_shape->containingContour(glm::vec2(local.x / sx, local.z / sz));
    if (contour < 0 || static_cast<size_t>(contour) >= m_shape->contours.size()) {
        return m_fallbackHsv;
    }

    const glm::vec3 hsv = SvgShape::rgbToHsvDegrees(m_shape->contours[contour].rgb);
    return glm::vec3(hsv.x, hsv.y * 0.5f, hsv.z * 0.5f);
}
