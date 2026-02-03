#pragma once
#include "VulkanApp.hpp"
#include "TextureArrayManager.hpp"
#include "../math/Vertex.hpp"
#include "BillboardManager.hpp"
#include "../utils/Scene.hpp" // for NodeID
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>

// Per-chunk vegetation instance buffer and renderer
class VegetationRenderer {
public:
    float billboardScale = 1.0f;
    explicit VegetationRenderer(VulkanApp* app_ = nullptr);
    ~VegetationRenderer();

    void setTextureArrayManager(TextureArrayManager* mgr);
    void onTextureArraysReallocated();
    void init(VulkanApp* app_);
    void cleanup();
    void createPipelines(VkRenderPass renderPassOverride = VK_NULL_HANDLE);

    // Register per-chunk vegetation instances
    void setChunkInstances(NodeID chunkId, const std::vector<glm::vec3>& positions);
    // New: Set chunk instance buffer directly from GPU buffer (compute shader output)
    void setChunkInstanceBuffer(NodeID chunkId, VkBuffer buffer, uint32_t count);
    void clearAllInstances();

    // Draw all visible vegetation chunks (frustum culling is per-chunk, matching geometry)
    void draw(VkCommandBuffer& commandBuffer, VkDescriptorSet vegetationDescriptorSet, const glm::mat4& viewProj);

    // Stats helpers
    size_t getChunkCount() const { return chunkInstanceCounts.size(); }
    size_t getInstanceTotal() const;

private:
    VulkanApp* app = nullptr;
    VkPipeline vegetationPipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    TextureArrayManager* vegetationTextureArrayManager = nullptr;

    // Descriptor set allocated from the app's descriptor pool and re-created when the texture arrays are (re)allocated
    VkDescriptorSet vegDescriptorSet = VK_NULL_HANDLE;
    uint32_t vegDescriptorVersion = 0;
    bool ensureVegDescriptorSet();
    // Listener id returned from TextureArrayManager::addAllocationListener(), -1 if none
    int vegTextureListenerId = -1;

    struct InstanceBuffer {
        VkBuffer buffer = VK_NULL_HANDLE; // instance data
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkBuffer indirectBuffer = VK_NULL_HANDLE; // indirect draw command
        VkDeviceMemory indirectMemory = VK_NULL_HANDLE;
        size_t count = 0;
    };
    std::unordered_map<NodeID, InstanceBuffer> chunkBuffers;
    std::unordered_map<NodeID, size_t> chunkInstanceCounts;
    void createInstanceBuffer(NodeID chunkId, const std::vector<glm::vec3>& positions);
    void destroyInstanceBuffer(NodeID chunkId);
};
