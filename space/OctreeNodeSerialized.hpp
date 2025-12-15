// Auto-generated wrapper header for OctreeNodeSerialized
#pragma once

#include <cstdint>
typedef unsigned int uint;

#pragma pack(16)
struct OctreeNodeSerialized {
    public:
    float sdf[8];
    uint children[8] = {0,0,0,0,0,0,0,0};
    int brushIndex;
    uint8_t bits;
};
#pragma pack()
