#pragma once
#include <glm/glm.hpp>

// GPU-side water uniform buffer
struct WaterUBO {
    glm::mat4 viewProjection;
    glm::mat4 invViewProjection;
    glm::vec4 viewPos;
    glm::vec4 screenSize;    // width, height, 1/width, 1/height
    float brushAlpha;        // brush overlay opacity (0-1)
    float _pad[3];           // std140 alignment padding to vec4
};
