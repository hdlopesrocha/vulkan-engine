#pragma once
#include <glm/glm.hpp>

struct MaterialGPU {
    glm::vec4 materialFlags;   // .z = ambientFactor
    glm::vec4 mappingParams;   // x = mappingEnabled (0/1), y = tessLevel, z = invertHeight (0/1), w = tessHeightScale
    glm::vec4 specularParams;  // x = specularStrength, y = shininess
    glm::vec4 triplanarParams; // x = scaleU, y = scaleV, z = triplanarEnabled (0/1)
    glm::vec4 normalParams;   // x = flipNormalY (0/1), y = swapNormalXZ (0/1), z/w = reserved
};
