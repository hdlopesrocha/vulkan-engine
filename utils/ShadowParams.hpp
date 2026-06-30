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
    // Per-cascade shadow map resolution set by ShadowRenderer.
    // Defaults here are overwritten at init from ShadowRenderer.
    uint32_t shadowMapSizes[SHADOW_CASCADE_COUNT] = {2048, 1024, 512};
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

        // ---- 3. Stable light view matrix ----
        // Use a fixed light view anchored to world origin so the light-space
        // coordinate system never changes between frames.  This makes texel
        // snapping trivially stable and prevents shadow swimming.
        glm::mat4 lightView = glm::lookAt(
            -lightDir * farPlane * 2.0f,
            glm::vec3(0.0f),
            worldUp);
        light.setViewMatrix(lightView);

        // ---- 4. Per-cascade orthographic bounds in stable light space ----
        for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
            float res = (float)shadowMapSizes[i];

            // Frustum-slice corners for this cascade
            float tNear = (splits[i] - nearPlane) / (farPlane - nearPlane);
            float tFar  = (splits[i + 1] - nearPlane) / (farPlane - nearPlane);

            std::array<glm::vec3, 8> sliceCorners;
            for (int j = 0; j < 4; ++j) {
                sliceCorners[j]     = glm::mix(corners[j], corners[j + 4], tNear);
                sliceCorners[j + 4] = glm::mix(corners[j], corners[j + 4], tFar);
            }

            // AABB in light space  (minLS.z = farthest, maxLS.z = nearest)
            glm::vec3 minLS( FLT_MAX), maxLS(-FLT_MAX);
            for (const auto& c : sliceCorners) {
                glm::vec4 ls = lightView * glm::vec4(c, 1.0f);
                minLS = glm::min(minLS, glm::vec3(ls));
                maxLS = glm::max(maxLS, glm::vec3(ls));
            }

            // Extend Z toward the light (more positive Z = toward the light)
            // to ensure casters between the light and the visible slice are
            // captured.
            maxLS.z = std::max(maxLS.z, -1.0f);

            // Pad XY by a fraction of the Z-range to catch shadow casters
            // that are outside the view frustum in XY.
            float zRange = maxLS.z - minLS.z;
            float pad = zRange * 0.1f;
            minLS.x -= pad;
            minLS.y -= pad;
            maxLS.x += pad;
            maxLS.y += pad;

            // ---- 5. Texel-snap the AABB edges outward ----
            // Snap minLS down (floor) and maxLS up (ceil) to texel boundaries
            // so the orthographic frustum always covers the original AABB.
            // The size becomes a multiple of snapX/snapY, keeping the
            // projection stable across frames while never dropping coverage.
            float snapX = (maxLS.x - minLS.x) / res;
            float snapY = (maxLS.y - minLS.y) / res;
            minLS.x = std::floor(minLS.x / snapX) * snapX;
            minLS.y = std::floor(minLS.y / snapY) * snapY;
            maxLS.x = std::ceil(maxLS.x / snapX) * snapX;
            maxLS.y = std::ceil(maxLS.y / snapY) * snapY;

            // ---- 6. Orthographic projection from the snapped AABB ----
            float nearVal = -maxLS.z;
            float farVal  = -minLS.z;
            glm::mat4 proj = glm::ortho(
                minLS.x, maxLS.x,
                minLS.y, maxLS.y,
                nearVal, farVal);

            light.setProjection(proj);
            lightSpaceMatrix[i] = light.getViewProjectionMatrix();
        }
    }
};
