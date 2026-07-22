#include <cstring>
#include "InstanceData.hpp"

InstanceData::InstanceData()
    : matrix(1.0f), shift(0.0f), animation(0)
{
}

InstanceData::InstanceData(uint animation_, const glm::mat4 &matrix_, float shift_)
    : matrix(matrix_), shift(shift_), animation(animation_)
{
}




