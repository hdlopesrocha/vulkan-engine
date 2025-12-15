#ifndef TILEDRAW_HPP
#define TILEDRAW_HPP

#include <glm/glm.hpp>

struct TileDraw {
public:
    glm::vec2 size;
    glm::vec2 offset;
    glm::vec2 pivot;
    float angle;
    uint index;
    TileDraw(uint index,glm::vec2 size, glm::vec2 offset, glm::vec2 pivot, float angle);
};

#endif
