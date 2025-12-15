#include "DebugInstanceData.hpp"

DebugInstanceData::DebugInstanceData(const glm::mat4 &matrix, float sdf[8], int brushIndex)
    : matrix(matrix), brushIndex(brushIndex)
{
    sdf1 = glm::vec4(sdf[0], sdf[1], sdf[2], sdf[3]);
    sdf2 = glm::vec4(sdf[4], sdf[5], sdf[6], sdf[7]);
}