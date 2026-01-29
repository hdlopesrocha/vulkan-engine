#pragma once

#include "Widget.hpp"
#include "../vulkan/ShadowRenderer.hpp"
#include "../vulkan/ShadowParams.hpp"
#include <imgui.h>
#include <glm/glm.hpp>

class ShadowMapWidget : public Widget {
public:
    ShadowMapWidget(ShadowRenderer* shadowMapper, ShadowParams* shadowParams);
    
    void render() override;
    
private:
    ShadowRenderer* shadowMapper;
    ShadowParams* shadowParams;
    float displaySize = 512.0f;
};
