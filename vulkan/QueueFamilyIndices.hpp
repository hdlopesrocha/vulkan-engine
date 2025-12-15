// Auto-generated wrapper header for QueueFamilyIndices
#pragma once

#include "vulkan.hpp"

// Queue family indices helper
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};