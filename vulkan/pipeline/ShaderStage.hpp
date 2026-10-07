#pragma once

#include <vulkan/vulkan.h>

struct ShaderStage {
    VkPipelineShaderStageCreateInfo info{};
    ShaderStage() = default;
    ShaderStage(VkShaderModule module, VkShaderStageFlagBits stage);
};
