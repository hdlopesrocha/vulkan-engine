#pragma once

#include <cstddef>

struct Vertex {
    float pos[3];
    float color[3];
    float uv[2];
    float normal[3];
    float texIndex;
    float tangent[4];
};
