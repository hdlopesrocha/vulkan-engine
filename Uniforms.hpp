#pragma once

#include <glm/glm.hpp>
#include "utils/MaterialProperties.hpp"
#include <cstddef>
#include <iostream>

struct UniformObject {
    glm::mat4 viewProjection;        // offset 0, size 64
    glm::vec4 viewPos;               // offset 64, size 16
    glm::vec4 lightDir;              // offset 80, size 16
    glm::vec4 lightColor;            // offset 96, size 16
    glm::vec4 materialFlags;         // offset 112, size 16
    glm::mat4 lightSpaceMatrix;      // offset 128, size 64
    glm::vec4 shadowEffects;         // offset 192, size 16
    glm::vec4 debugParams;           // offset 208, size 16
    glm::vec4 triplanarSettings;     // offset 224, size 16
    glm::vec4 tessParams;            // offset 240, size 16
    glm::vec4 passParams;            // offset 256, size 16  <-- THIS IS WHERE noiseScale IS
    
    // Total size: 272 bytes

    // Note: sky-related data moved to SkyUniform

    void setMaterial(const MaterialProperties& mat) {
        materialFlags = glm::vec4(0.0f, 0.0f, mat.ambientFactor, 0.0f);
        // Per-material values are provided by the Materials SSBO; do not duplicate here.
    }
    
    // Debug: print passParams
    void printPassParams() const {
        std::cout << "[UBO Debug] passParams at offset 256: "
                  << "x=" << passParams.x << " y=" << passParams.y 
                  << " z=" << passParams.z << " w=" << passParams.w << std::endl;
    }
};

// Separate UBO dedicated to sky parameters (binding 6)
struct SkyUniform {
    glm::vec4 skyHorizon; // rgb = horizon color, a = unused
    glm::vec4 skyZenith;  // rgb = zenith color, a = unused
    glm::vec4 skyParams;  // x = warmth, y = exponent, z = sunFlare, w = unused
    glm::vec4 nightHorizon; // rgb = night horizon color
    glm::vec4 nightZenith;  // rgb = night zenith color
    glm::vec4 nightParams;  // x = night intensity, y = starIntensity, z/w unused
};
