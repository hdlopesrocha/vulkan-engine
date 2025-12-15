#include <cstring>
#include "InstanceData.hpp"

InstanceData::InstanceData()
    : matrix(1.0f), shift(0.0f), animation(0)
{
}

InstanceData::InstanceData(uint animation, const glm::mat4 &matrix, float shift)
    : matrix(matrix), shift(shift), animation(animation)
{
}




