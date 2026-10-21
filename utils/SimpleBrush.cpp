#include "SimpleBrush.hpp"

 
SimpleBrush::SimpleBrush(int brush_, glm::vec3 hsv_, TextureType textureType_){
    this->brush = brush_;
    this->hsv = hsv_;
    this->textureType = textureType_;
}

int SimpleBrush::paint(const Vertex &vertex) const {
    return brush;
}

glm::vec3 SimpleBrush::paintHSV(const Vertex &vertex) const {
    return hsv;
}

TextureType SimpleBrush::paintTextureType(const Vertex &vertex) const {
    return textureType;
}
