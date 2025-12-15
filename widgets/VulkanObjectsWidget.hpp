#pragma once

#include "Widget.hpp"
#include <memory>

class VulkanApp;

class VulkanObjectsWidget : public Widget {
public:
    VulkanObjectsWidget(VulkanApp* app);
    void render() override;

private:
    VulkanApp* app;
    bool showHex = true;
};
