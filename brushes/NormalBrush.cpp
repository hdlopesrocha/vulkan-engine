#include "NormalBrush.hpp"

NormalBrush::NormalBrush(int brush_) {
    this->brush = brush_;
    this->hsv = glm::vec3(0.0f, 0.5f, 0.5f);
    this->normal = glm::vec3(0.0f, 1.0f, 0.0f);
}

NormalBrush::NormalBrush(int brush_, glm::vec3 hsv_, glm::vec3 normal_) {
    this->brush = brush_;
    this->hsv = hsv_;
    this->normal = normal_;
}

int NormalBrush::paint(const OctreeNode &o) const {
    (void)o;
    // The surface normal is no longer available to the painter (the octree
    // build no longer computes SDF::getNormalFromPosition), so the node's
    // material is painted unconditionally.
    return brush;
}

glm::vec3 NormalBrush::paintHSV(const OctreeNode &o) const {
    return hsv;
}
