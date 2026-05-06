#pragma once

#include "../VulkanApp.hpp"
#include "IndirectRenderer.hpp"
#include "../../math/Vertex.hpp"
#include <vulkan/vulkan.h>

// Lightweight renderer that creates wireframe pipelines against arbitrary render
// passes and draws geometry from an external IndirectRenderer.  Owned by
// SceneRenderer so wireframe mode works uniformly for solid and water passes.
class WireframeRenderer {
public:
    WireframeRenderer() = default;
    ~WireframeRenderer() = default;

    // Create a wireframe pipeline using dynamic rendering.
    // `colorFormats` must list the color attachment formats (matches the pass).
    // `setLayouts` are the descriptor set layouts the pipeline needs.
    // `hasTessellation` enables patch-list topology + tess shaders.
    void createPipeline(VulkanApp* app,
                        const std::vector<VkFormat>& colorFormats,
                        const std::vector<VkDescriptorSetLayout>& setLayouts,
                        const char* vertPath,
                        const char* fragPath,
                        const char* tescPath,
                        const char* tesePath,
                        const char* label);

    // Draw wireframe using the given indirect renderer's prepared buffers.
    // Must be called inside a compatible render pass.
    // `descriptorSets` are bound consecutively starting at set 0.
    void draw(VkCommandBuffer cmd,
              VulkanApp* app,
              const std::vector<VkDescriptorSet>& descriptorSets,
              IndirectRenderer& indirectRenderer);

    VkPipeline getPipeline() const { return wireframePipeline; }
    VkPipelineLayout getPipelineLayout() const { return wireframePipelineLayout; }

    void cleanup();

private:
    VkPipeline wireframePipeline = VK_NULL_HANDLE;
    VkPipelineLayout wireframePipelineLayout = VK_NULL_HANDLE;
};
