#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include "Vertex.hpp"

const std::array<glm::ivec3, 8> CUBE_CORNERS = {{
    {0, 0, 0},
    {0, 0, 1},
    {0, 1, 0},
    {0, 1, 1},
    {1, 0, 0},
    {1, 0, 1},
    {1, 1, 0},
    {1, 1, 1}
}};

#include "ContainmentType.hpp"
#include "SpaceType.hpp"


 
