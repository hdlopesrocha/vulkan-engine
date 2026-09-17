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

int NormalBrush::paint(const Vertex &vertex) const {
    return glm::dot(vertex.normal, normal) > 0.0f ? brush : DISCARD_BRUSH_INDEX;
}

glm::vec3 NormalBrush::paintHSV(const Vertex &vertex) const {
    return hsv;
}
