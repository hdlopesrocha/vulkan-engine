#pragma once

#include <vulkan/vulkan.h>

struct ShaderStage {
    VkPipelineShaderStageCreateInfo info{};
    ShaderStage() = default;
    ShaderStage(VkShaderModule module, VkShaderStageFlagBits stage) {
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage = stage;
        info.module = module;
        info.pName = "main";
    }
};
