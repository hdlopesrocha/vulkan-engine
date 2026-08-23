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
    glm::vec4 shadowEffects;         // offset 128, size 16
    glm::vec4 debugParams;           // offset 208, size 16
    glm::vec4 triplanarSettings;     // offset 224, size 16
    glm::vec4 tessParams;            // offset 240, size 16
    glm::vec4 passParams;            // x=isShadowPass, y=tessEnabled, z=nearPlane, w=farPlane
    glm::mat4 invViewProjection;     // inverse of viewProjection (camera-constant)
    glm::vec4 brushParams;           // offset 464, size 16  x=brushTextureIndex, y=brushMode (0=overlay, 2=PAINT)
    glm::vec4 brushHSV;              // offset 480, size 16  x=H(0..360), y=S(0..1), z=V(0..1), w=unused

    // Total size: 496 bytes

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
