#pragma once

#include <cstdint>
#include <glm/glm.hpp>
typedef unsigned int uint;

#pragma pack(16)
struct OctreeNodeSerialized {
    public:
    float sdf[8];
    uint children[8] = {0,0,0,0,0,0,0,0};
    int brushIndex;
    glm::vec3 hsv;
    uint8_t bits;
    uint8_t lod = 0;
    uint8_t chunkLod = 0;
};
#pragma pack()
