#pragma once
#include <glm/glm.hpp>

// GPU-side water render UBO
struct WaterRenderUBO {
    glm::vec4 timeParams; // x = waterTime, y = water refraction allowed,
                          // z = water reflection allowed,
                          // w = water blur allowed (Settings::blurEnabled)
    // x = solidSceneDepthTex holds THIS frame's solid depth (1) or a stale /
    //     unrelated one (0). Gates the shader-side solid-occlusion rejection
    //     (perf report 20 C5): the offscreen water pass runs after the solid
    //     pass, so it can discard fragments the terrain already covers; the
    //     water-in-main variant binds the PREVIOUS frame's depth for its
    //     in-trace lookups, where the same discard would punch holes under
    //     camera motion. yzw = unused.
    glm::vec4 depthParams;
};
