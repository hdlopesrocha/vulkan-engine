#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

// Per-command-buffer state tracker that eliminates redundant vkCmdBindPipeline
// calls.  Designed to be reset at the start of each frame or when beginning a
// new command buffer.
//
// Descriptor set caching is intentionally omitted: the same VkDescriptorSet
// handle can be bound with different VkPipelineLayout values across renderers,
// and the pre-existing codebase has layout-compatibility issues that make
// caching fragile.  Pipeline handles uniquely identify their creation state
// (including layout), so skipping redundant pipeline binds is always safe.
//
// Thread-safety: not thread-safe.  Must not be accessed concurrently from
// multiple threads.  Renderers used by the async back-face task keep
// cmdState=nullptr to avoid data races.
struct CommandBufferState {
    VkPipeline lastGraphicsPipeline = VK_NULL_HANDLE;
    VkPipeline lastComputePipeline  = VK_NULL_HANDLE;

    void reset() {
        lastGraphicsPipeline = VK_NULL_HANDLE;
        lastComputePipeline  = VK_NULL_HANDLE;
    }

    void bindGraphicsPipeline(VkCommandBuffer cmd, VkPipeline pipeline) {
        if (pipeline == lastGraphicsPipeline) return;
        lastGraphicsPipeline = pipeline;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    }

    void bindComputePipeline(VkCommandBuffer cmd, VkPipeline pipeline) {
        if (pipeline == lastComputePipeline) return;
        lastComputePipeline = pipeline;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    }

    void bindGraphicsDescriptorSets(VkCommandBuffer cmd, VkPipelineLayout layout,
                                     uint32_t firstSet, uint32_t setCount,
                                     const VkDescriptorSet* sets,
                                     uint32_t dynamicOffsetCount = 0,
                                     const uint32_t* dynamicOffsets = nullptr) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                                firstSet, setCount, sets, dynamicOffsetCount, dynamicOffsets);
    }

    void bindComputeDescriptorSets(VkCommandBuffer cmd, VkPipelineLayout layout,
                                    uint32_t firstSet, uint32_t setCount,
                                    const VkDescriptorSet* sets,
                                    uint32_t dynamicOffsetCount = 0,
                                    const uint32_t* dynamicOffsets = nullptr) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                                firstSet, setCount, sets, dynamicOffsetCount, dynamicOffsets);
    }

    void bindGraphicsPipelineUnchecked(VkCommandBuffer cmd, VkPipeline pipeline) {
        lastGraphicsPipeline = pipeline;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    }

    void bindComputePipelineUnchecked(VkCommandBuffer cmd, VkPipeline pipeline) {
        lastComputePipeline = pipeline;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    }
};
