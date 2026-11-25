#pragma once

#include <vector>

#include <glm/glm.hpp>

// One normalized SVG contour (polyline) plus its parsed fill colour.
struct SvgContour {
    std::vector<glm::vec2> points;  // normalized [-1,1] range (may exceed for open paths)
    bool closed = true;
};
