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
    // The surface normal is only computed on the GPU (the octree build never
    // calls SDF::getNormalFromPosition). Tag the brush; OctreeTriangles.comp
    // computes dot(surfaceNormal, +Y) and discards the nodes that fail it.
    return NORMAL_BRUSH_INDEX;
}

glm::vec3 NormalBrush::paintHSV(const OctreeNode &o) const {
    return hsv;
}
