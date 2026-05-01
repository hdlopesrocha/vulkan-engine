#pragma once

#include <glm/glm.hpp>
#include "../../utils/MaterialProperties.hpp"
#include <cstddef>
#include <iostream>

static constexpr int SHADOW_CASCADE_COUNT = 3;

struct UniformObject {
    glm::mat4 viewProjection;        // offset 0, size 64
    glm::vec4 viewPos;               // offset 64, size 16
    glm::vec4 lightDir;              // offset 80, size 16
    glm::vec4 lightColor;            // offset 96, size 16
    glm::vec4 materialFlags;         // offset 112, size 16
    glm::mat4 lightSpaceMatrix;      // offset 128, size 64  (cascade 0)
    glm::vec4 shadowEffects;         // offset 192, size 16
    glm::vec4 debugParams;           // offset 208, size 16
    glm::vec4 triplanarSettings;     // offset 224, size 16
    glm::vec4 tessParams;            // offset 240, size 16
    glm::vec4 passParams;            // offset 256, size 16  x=isShadowPass, y=tessEnabled, z=nearPlane, w=farPlane
    glm::mat4 lightSpaceMatrix1;     // offset 272, size 64  (cascade 1 = 4x ortho0)
    glm::mat4 lightSpaceMatrix2;     // offset 336, size 64  (cascade 2 = 16x ortho0)
    
    // Total size: 400 bytes

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
