#include "SimpleBrush.hpp"

 
SimpleBrush::SimpleBrush(int brush_, glm::vec3 hsv_) {
    this->brush = brush_;
    this->hsv = hsv_;
}

int SimpleBrush::paint(const OctreeNode &o) const {
    return brush;
}

glm::vec3 SimpleBrush::paintHSV(const OctreeNode &o) const {
    return hsv;
}
