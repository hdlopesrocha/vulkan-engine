#pragma once

#include <glm/glm.hpp>
#include <cstdint>
#include "../math/BoundingCube.hpp"
#include "../math/Vertex.hpp"
#include "../math/BrushMode.hpp"

typedef unsigned int uint;

struct alignas(16) OctreeNodeCubeSerialized {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    int brushIndex;
    uint children[8];
    glm::vec3 min;
    uint bits;
    glm::vec3 length;
    uint level;

    OctreeNodeCubeSerialized();
    OctreeNodeCubeSerialized(float * sdf, BoundingCube cube, Vertex vertex, uint bits, uint level);
};
