#include "SimpleBrush.hpp"

 
SimpleBrush::SimpleBrush(int brush_, glm::vec3 hsv_){
    this->brush = brush_;
    this->hsv = hsv_;
}

int SimpleBrush::paint(const Vertex &vertex) const {
    return brush;
}

glm::vec3 SimpleBrush::paintHSV(const Vertex &vertex) const {
    return hsv;
}
