#pragma once

#include <glm/glm.hpp>

// Separate UBO dedicated to sky parameters (binding 6)
struct SkyUniform {
    glm::vec4 skyHorizon; // rgb = horizon color, a = unused
    glm::vec4 skyZenith;  // rgb = zenith color, a = unused
    glm::vec4 skyParams;  // x = warmth, y = exponent, z = sunFlare, w = unused
    glm::vec4 nightHorizon; // rgb = night horizon color
    glm::vec4 nightZenith;  // rgb = night zenith color
    glm::vec4 nightParams;  // x = night intensity, y = starIntensity, z/w unused
};
