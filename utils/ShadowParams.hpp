#pragma once

// Light.hpp defines GLM_FORCE_DEPTH_ZERO_TO_ONE — include first
#include "../math/Light.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include "../vulkan/ubo/UniformObject.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <cfloat>

struct ShadowParams {
    float orthoSize = 1024.0f;
    uint32_t shadowMapSize = 8192;
    glm::mat4 lightSpaceMatrix[SHADOW_CASCADE_COUNT];

    void update(const glm::vec3& camPos, Light& light,
                const glm::mat4& cameraViewProj,
                float nearPlane, float farPlane) {
        glm::vec3 lightDir = glm::normalize(light.getDirection());

        glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
        if (std::abs(glm::dot(lightDir, worldUp)) > 0.9f)
            worldUp = glm::vec3(1.0f, 0.0f, 0.0f);

        // ---- 1. Camera frustum corners in world space ----
        glm::mat4 invViewProj = glm::inverse(cameraViewProj);
        std::array<glm::vec3, 8> corners;
        int idx = 0;
        for (int z = 0; z < 2; ++z) {
            for (int y = 0; y < 2; ++y) {
                for (int x = 0; x < 2; ++x) {
                    glm::vec4 clip(x == 0 ? -1.0f : 1.0f,
                                   y == 0 ? -1.0f : 1.0f,
                                   z == 0 ? 0.0f : 1.0f, 1.0f);
                    glm::vec4 w = invViewProj * clip;
                    corners[idx++] = glm::vec3(w) / w.w;
                }
            }
        }

        // ---- 2. Practical cascade splits ----
        float splits[SHADOW_CASCADE_COUNT + 1];
        splits[0] = nearPlane;
        for (int i = 1; i <= SHADOW_CASCADE_COUNT; ++i) {
            float ci = (float)i / (float)SHADOW_CASCADE_COUNT;
            float logSplit = nearPlane * std::pow(farPlane / nearPlane, ci);
            float uniformSplit = nearPlane + (farPlane - nearPlane) * ci;
            splits[i] = glm::mix(logSplit, uniformSplit, 0.5f);
        }

        // ---- 3. Shared light view matrix ----
        // Use the frustum center as the look-at target for consistent orientation.
        glm::vec3 frustumCenter(0.0f);
        for (const auto& c : corners) frustumCenter += c;
        frustumCenter /= 8.0f;

        glm::vec3 lightPos = frustumCenter - lightDir * farPlane * 2.0f;
        glm::mat4 lightView = glm::lookAt(lightPos, frustumCenter, worldUp);
        light.setViewMatrix(lightView);

        // ---- 4. Per-cascade AABB in light space ----
        // The standard CSM approach computes a tight orthographic frustum for
        // each cascade by projecting the view frustum slice into light space,
        // then extending the Z-range toward the light to catch all potential
        // shadow casters.
        const float padFraction = 0.15f;

        for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
            // Interpolation factors for this cascade's near and far slice
            float tNear = (splits[i] - nearPlane) / (farPlane - nearPlane);
            float tFar  = (splits[i + 1] - nearPlane) / (farPlane - nearPlane);

            // Compute the 8 corners of the frustum slice
            std::array<glm::vec3, 8> sliceCorners;
            for (int j = 0; j < 4; ++j) {
                sliceCorners[j]     = glm::mix(corners[j], corners[j + 4], tNear);
                sliceCorners[j + 4] = glm::mix(corners[j], corners[j + 4], tFar);
            }

            // Transform to light space and compute AABB
            glm::vec3 minLS( FLT_MAX);
            glm::vec3 maxLS(-FLT_MAX);
            for (const auto& c : sliceCorners) {
                glm::vec4 ls = lightView * glm::vec4(c, 1.0f);
                minLS = glm::min(minLS, glm::vec3(ls));
                maxLS = glm::max(maxLS, glm::vec3(ls));
            }

            // Extend Z-range toward the light to capture shadow casters between
            // the light and the visible frustum slice.  Without this extension,
            // objects closer to the light than the view frustum (e.g. geometry
            // behind the camera or above the camera) are missed and can't cast
            // shadows.  Clamp near plane to at least 1 unit from the light for
            // depth precision.
            maxLS.z = std::max(maxLS.z, -1.0f);

            // Pad XY bounds by a fraction of the Z-range to catch shadow
            // casters that are outside the view frustum in XY but can still
            // cast shadows into it.
            float zRange = maxLS.z - minLS.z;
            float padX = zRange * 0.1f;
            float padY = zRange * 0.1f;
            minLS.x -= padX;
            maxLS.x += padX;
            minLS.y -= padY;
            maxLS.y += padY;

            // ---- 5. Texel snap ----
            float snapX = (maxLS.x - minLS.x) / (float)shadowMapSize;
            float snapY = (maxLS.y - minLS.y) / (float)shadowMapSize;
            minLS.x = std::floor(minLS.x / snapX) * snapX;
            minLS.y = std::floor(minLS.y / snapY) * snapY;
            maxLS.x = minLS.x + snapX * (float)shadowMapSize;
            maxLS.y = minLS.y + snapY * (float)shadowMapSize;

            // ---- 6. Orthographic projection ----
            // glm::ortho with GLM_FORCE_DEPTH_ZERO_TO_ONE maps:
            //   Z_view = -near  → depth 0
            //   Z_view = -far   → depth 1
            // minLS.z is the farthest (most negative), maxLS.z is the nearest.
            float nearVal = -maxLS.z;
            float farVal  = -minLS.z;
            glm::mat4 proj = glm::ortho(minLS.x, maxLS.x, minLS.y, maxLS.y, nearVal, farVal);

            light.setProjection(proj);
            lightSpaceMatrix[i] = light.getViewProjectionMatrix();
        }
    }
};
