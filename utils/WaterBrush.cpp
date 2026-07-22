#include "WaterBrush.hpp"

WaterBrush::WaterBrush(int water_){
    this->water = water_;
}

int WaterBrush::paint(const Vertex &vertex, glm::vec4 translate, glm::vec4 scale) const {    
    int brushIndex;
    glm::vec3 n = glm::normalize(vertex.normal);
    if (glm::dot(glm::vec3(0.0f,1.0f,0.0f), n ) < 0.5 ){
        brushIndex= water;
    } else {
        brushIndex = water;
    } 
    return brushIndex;
}

