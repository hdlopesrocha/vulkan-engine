#pragma once

#include "Widget.hpp"
#include "../services/ImpostorService.hpp"
#include <memory>

class VulkanApp;
class VegetationRenderer;

// Dedicated widget for capturing and inspecting Fibonacci-sphere impostor views
// of billboard vegetation textures.
//
// Usage:
//   1. Construct with shared ImpostorService.
//   2. The service handles Vulkan lifecycle; just call render() for the UI.
class ImpostorWidget : public Widget {
public:
    ImpostorWidget(std::shared_ptr<ImpostorService> impostorService);

    void setVegetationRenderer(VegetationRenderer* renderer)
        { impostorService->setVegetationRenderer(renderer); }

    void render() override;

private:
    std::shared_ptr<ImpostorService> impostorService;

    // Active selection in the combo box.
    int  selectedBillboard  = 0;

    // Scale used for the capture
    float captureScale      = 10.0f;

    // Preview orbit state
    float previewYaw        = 0.0f;
    float previewPitch      = 0.3f;
    int   previewMode       = 0; // 0=albedo, 1=normal, 2=depth
};
