#pragma once
#include <glm/glm.hpp>

// GPU uniform buffer for vegetation wind parameters.
// Accessed at set=2, binding=0 in vegetation/impostor shaders.
// Updated once per frame (or less frequently for invariant fields).
struct WindParamsUBO {
    glm::vec4 windDirAndStrength;   // xz = wind direction (normalized), w = strength
    glm::vec4 windNoise;            // x = baseFrequency, y = speed, z = gustFrequency, w = gustStrength
    glm::vec4 windShape;            // x = skewAmount, y = trunkStiffness, z = noiseScale, w = verticalFlutter
    glm::vec4 windTurbulence;       // x = turbulence
    glm::vec4 densityParams;        // x = enabled, y = nearDistance, z = farDistance, w = minFactor
    glm::vec4 cameraPosAndFalloff;  // xyz = main camera position, w = density falloff
};
static_assert(sizeof(WindParamsUBO) == 96, "WindParamsUBO expected 96 bytes");
