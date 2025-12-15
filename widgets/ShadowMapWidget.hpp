#pragma once

#include "Widget.hpp"
#include "../vulkan/ShadowMapper.hpp"
#include <imgui.h>
#include <glm/glm.hpp>

class ShadowMapWidget : public Widget {
public:
    ShadowMapWidget(ShadowMapper* shadowMapper);
    
    void render() override;
    
private:
    ShadowMapper* shadowMapper;
    float displaySize = 512.0f;
};
