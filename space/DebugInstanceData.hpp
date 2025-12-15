// Auto-generated wrapper header for DebugInstanceData
#pragma once
#include <glm/glm.hpp>

struct DebugInstanceData {
public:
    glm::vec4 sdf1;
    glm::vec4 sdf2;
    glm::mat4 matrix;
    int brushIndex;
    DebugInstanceData(const glm::mat4 &matrix, float sdf[8], int brushIndex);
};