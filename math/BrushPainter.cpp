#include "BrushPainter.hpp"

glm::vec3 BrushPainter::paintHSV(const OctreeNode& o) const {
    (void)o;
    return glm::vec3(0.0f, 0.5f, 0.5f);
}

bool BrushPainter::needsPosition() const {
    return true;
}
