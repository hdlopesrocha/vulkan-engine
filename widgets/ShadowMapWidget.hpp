#pragma once

#include "Widget.hpp"
#include "../vulkan/ShadowMapper.hpp"
#include "../vulkan/ShadowParams.hpp"
#include <imgui.h>
#include <glm/glm.hpp>

class ShadowMapWidget : public Widget {
public:
    ShadowMapWidget(ShadowMapper* shadowMapper, ShadowParams* shadowParams);
    
    void render() override;
    
private:
    ShadowMapper* shadowMapper;
    ShadowParams* shadowParams;
    float displaySize = 512.0f;
};
