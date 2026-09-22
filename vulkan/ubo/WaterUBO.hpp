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
    // H4: 1 when at least one water layer needs the final-pass blur, i.e. the
    // water pass wrote the body/column aux attachments; 0 lets the composite
    // skip both fetches entirely (was part of the std140 padding).
    float waterBlurEnabled;
    float _pad;              // std140 alignment padding to vec4
};
