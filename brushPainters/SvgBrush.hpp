#pragma once

#include <memory>

#include <glm/glm.hpp>

#include "../math/BrushPainter.hpp"
#include "../math/geometry/Transformation.hpp"
#include "../sdf/SvgShape.hpp"

// Per-vertex painter for the SVG brush: the world position is mapped into the
// SVG's normalized profile space with the same transform the SDF uses
// (SvgPolygonDistanceFunction), and the vertex takes the fill colour of the
// contour that contains it as its HSV tint. The material index is the brush's
// configured texture layer for every vertex.
//
// Engine HSV convention: the solid shader treats (0, 0.5, 0.5) as neutral and
// multiplies saturation/value by 2, so the exported tint stores half of the
// parsed saturation/value to reproduce the drawing's colours exactly.
class SvgBrush : public BrushPainter {
public:
    SvgBrush(std::shared_ptr<const SvgShape> shape, const Transformation &model,
             int materialIndex, glm::vec3 fallbackHsv = glm::vec3(0.0f, 0.5f, 0.5f),
             bool tintFromFill = true);

    int paint(const OctreeNode &o) const override;
    glm::vec3 paintHSV(const OctreeNode &o) const override;

private:
    std::shared_ptr<const SvgShape> m_shape;
    Transformation m_model;
    int m_material = 0;
    glm::vec3 m_fallbackHsv{0.0f, 0.5f, 0.5f};
    bool m_tintFromFill = true;
};
