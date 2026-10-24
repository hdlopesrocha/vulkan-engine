#pragma once

#include <vector>

#include <glm/glm.hpp>

// One normalized SVG contour (polyline) plus its parsed fill colour.
struct SvgContour {
    std::vector<glm::vec2> points;  // normalized [-1,1] range (may exceed for open paths)
    glm::vec3 rgb = glm::vec3(1.0f); // parsed fill colour
    bool closed = true;
};
