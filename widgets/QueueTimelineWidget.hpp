#pragma once

#include "Widget.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "imgui.h"
#include "../vulkan/VulkanApp.hpp"

class QueueTimelineWidget : public Widget {
public:
    explicit QueueTimelineWidget(VulkanApp* app);
    // Per-frame refresh (does NOT store VulkanApp* persistently)
    void updateWithApp(class VulkanApp* app);
    void render() override;

private:
    struct QueueRow {
        VkQueue  handle = VK_NULL_HANDLE;
        std::string name;
        ImVec4   color = {1, 1, 1, 1};
    };

    // Cached queue rows, rebuilt on first updateWithApp (handles are stable).
    std::vector<QueueRow> rows_;
    bool rowsBuilt_ = false;

    // Snapshot of queue-timeline segments copied from VulkanApp each frame.
    std::vector<VulkanApp::QueueSegment> segments_;
    uint64_t latestFrame_ = 0;

    // UI state
    bool perFrameMode_ = true;   // true: slot grid for the latest frame; false: rolling Gantt
    int  slots_ = 72;            // time slots across a frame
    float windowMs_ = 50.0f;     // rolling window length (ms)

    static uint64_t nowNs();
};
