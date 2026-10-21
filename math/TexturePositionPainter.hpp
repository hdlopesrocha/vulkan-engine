#pragma once
#include "geometry/Vertex.hpp"
#include "TextureType.hpp"
#include <glm/glm.hpp>

// Paints per-vertex surface attributes while the octree is built. The painter
// decides (per node/vertex) which material the surface uses (paint) and its HSV
// tint (paintHSV). The texture-coordinate technique and UV are a per-shape
// decision and come from the TextureCoordPainter passed to Octree::apply().
class TexturePositionPainter {
public:
    virtual int paint(const Vertex &v) const = 0;
    virtual glm::vec3 paintHSV(const Vertex &v) const { return glm::vec3(0.0f, 0.5f, 0.5f); }
};
