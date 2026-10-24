#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <vector>

#include "../resources/Buffer.hpp"

// Three host-visible descriptor buffers (one per frame-in-flight) holding the
// full set-0 image/buffer descriptors, written via vkGetDescriptorEXT
// (DescriptorBuffer wrapper) instead of vkUpdateDescriptorSets.
struct Set0DescriptorBuffers {
    std::vector<Buffer> buffers;                 // one per frame
    std::vector<VkDeviceAddress> addresses;      // device address per frame
    VkDeviceSize setSize = 0;                    // aligned set-0 byte size
    std::array<VkDeviceSize, 29> bindingOffsets{}; // driver binding offsets
    bool ready = false;
};
