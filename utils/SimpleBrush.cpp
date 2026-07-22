#include "SimpleBrush.hpp"

 
SimpleBrush::SimpleBrush(int brush_){
    this->brush = brush_;
}

int SimpleBrush::paint(const Vertex &vertex, glm::vec4 translate, glm::vec4 scale) const {
    return brush;
}
