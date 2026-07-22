#pragma once
#include <glm/glm.hpp>

struct TileDraw {
public:
    glm::vec2 size;
    glm::vec2 offset;
    glm::vec2 pivot;
    float angle;
    uint index;
    TileDraw(uint index_, glm::vec2 size_, glm::vec2 offset_, glm::vec2 pivot_, float angle_);
};

 
