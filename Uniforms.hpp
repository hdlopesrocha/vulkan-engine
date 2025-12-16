#pragma once

#include <glm/glm.hpp>
#include "utils/MaterialProperties.hpp"

struct UniformObject {
    glm::mat4 mvp;
    glm::mat4 model;
    glm::vec4 viewPos; // world-space camera position
    glm::vec4 lightDir; // xyz = direction, w = unused (padding)
    glm::vec4 lightColor; // rgb = color, w = intensity
    // Material flags
    glm::vec4 materialFlags; // x=unused, y=unused, z=ambient, w=unused
    // Per-material properties are read from the GPU-side Materials SSBO; keep only per-pass flags here.
    glm::mat4 lightSpaceMatrix; // for shadow mapping
    glm::vec4 shadowEffects; // x=enableSelfShadow, y=enableShadowDisplacement, z=selfShadowQuality, w=unused
    glm::vec4 debugParams; // x=debugMode (0=normal,1=normalVec,2=normalMap,3=uv,4=tangent,5=bitangent)
    glm::vec4 tessParams; // x = tessNearDist, y = tessFarDist, z = tessMinLevel, w = tessMaxLevel
    glm::vec4 passParams;   // x = isShadowPass (1.0 for shadow pass, 0.0 for main pass)

    // Note: sky-related data moved to SkyUniform

    void setMaterial(const MaterialProperties& mat) {
        materialFlags = glm::vec4(0.0f, 0.0f, mat.ambientFactor, 0.0f);
        // Per-material values are provided by the Materials SSBO; do not duplicate here.
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
