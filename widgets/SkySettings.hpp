#pragma once

#include <glm/glm.hpp>

// Plain-old-data settings for sky appearance. Intentionally public for easy UI binding.
struct SkySettings {
    // Mode
    enum class Mode { Gradient = 0, Grid = 1 } mode = Mode::Gradient;

    // Day colors
    glm::vec3 horizonColor = glm::vec3(0.6f, 0.7f, 0.9f);
    glm::vec3 zenithColor = glm::vec3(0.05f, 0.15f, 0.4f);

    // Day controls
    float warmth = 0.0f; // how warm the horizon becomes when sun is low
    float exponent = 1.0f; // gradient power
    float sunFlare = 1.0f; // sun flare intensity

    // Night parameters
    glm::vec3 nightHorizon = glm::vec3(0.02f, 0.03f, 0.06f);
    glm::vec3 nightZenith = glm::vec3(0.0f, 0.02f, 0.08f);
    float nightIntensity = 1.0f; // how dark/strong the night colors are
    float starIntensity = 0.5f; // brightness/amount of stars
};
