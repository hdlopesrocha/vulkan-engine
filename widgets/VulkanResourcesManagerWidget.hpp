#pragma once

#include "Widget.hpp"
#include "../vulkan/VulkanResourceManager.hpp"
#include <memory>

class VulkanApp;

class VulkanResourcesManagerWidget : public Widget {
public:
    VulkanResourcesManagerWidget(VulkanResourceManager* mgr, VulkanApp* app = nullptr);
    void render() override;

private:
    VulkanResourceManager* mgr;
    VulkanApp* app;
    bool showHex = true;
};
