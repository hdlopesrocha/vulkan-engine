#pragma once
#include <glm/glm.hpp>

// GPU-side water uniform buffer
struct WaterUBO {
    glm::mat4 viewProjection;
    glm::mat4 invViewProjection;
    glm::vec4 viewPos;
    glm::vec4 screenSize;    // width, height, 1/width, 1/height
    float brushAlpha;        // brush overlay opacity (0-1)
    float brushMode;         // 0=overlay, 2=PAINT (replaces solid texture)
    float waterBlurScale;    // depth-guided water blur radius (pixels per meter)
    float waterBlurMax;      // water blur radius clamp (pixels)
};
