#include "LandBrush.hpp"

#include "../math/BrushMode.hpp"

int LandBrush::paint(const OctreeNode& o) const {
    (void)o;
    return LAND_BRUSH_INDEX;
}

glm::vec3 LandBrush::paintHSV(const OctreeNode& o) const {
    (void)o;
    return glm::vec3(0.0f, 0.5f, 0.5f);
}

bool LandBrush::needsPosition() const {
    return false;
}
