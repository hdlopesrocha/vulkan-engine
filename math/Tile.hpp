#ifndef TILE_HPP
#define TILE_HPP

#include <glm/glm.hpp>

struct Tile {
public:
    glm::vec2 size;
    glm::vec2 offset;
    Tile(glm::vec2 size, glm::vec2 offset);
};

#endif
