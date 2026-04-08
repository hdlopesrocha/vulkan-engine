#pragma once
#include "VulkanApp.hpp"
#include "TextureArrayManager.hpp"
#include "../math/Vertex.hpp"
#include "BillboardManager.hpp"
#include "VertexBufferObject.hpp"
#include "../utils/Scene.hpp" // for NodeID
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>

// Per-chunk vegetation instance buffer and renderer
class VegetationRenderer {
public:
    float billboardScale = 1.0f;
    explicit VegetationRenderer();
    ~VegetationRenderer();

    void setTextureArrayManager(TextureArrayManager* mgr, VulkanApp* app);
    void onTextureArraysReallocated(VulkanApp* app);
    void init();
    void cleanup();
    void init(VulkanApp* app, VkRenderPass renderPassOverride = VK_NULL_HANDLE);
    // Generate per-chunk vegetation instances from mesh geometry using the
    // compute shader. This is the only supported instancing path now.
    void generateChunkInstances(NodeID chunkId,
                                VkBuffer vertexBuffer, uint32_t vertexCount,
                                VkBuffer indexBuffer, uint32_t indexCount,
                                uint32_t instancesPerTriangle, VulkanApp* app,
                                uint32_t seed = 1337);
    void clearAllInstances();

    // Draw all visible vegetation chunks (frustum culling is per-chunk, matching geometry)
    void draw(VulkanApp* app, VkCommandBuffer& commandBuffer, VkDescriptorSet vegetationDescriptorSet, const glm::mat4& viewProj);

    // Stats helpers
    size_t getChunkCount() const { return chunkInstanceCounts.size(); }
    size_t getInstanceTotal() const;

private:
    
    VkPipeline vegetationPipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    TextureArrayManager* vegetationTextureArrayManager = nullptr;

    // Descriptor set allocated from the app's descriptor pool and re-created when the texture arrays are (re)allocated
    VkDescriptorSet vegDescriptorSet = VK_NULL_HANDLE;
    uint32_t vegDescriptorVersion = 0;
    bool ensureVegDescriptorSet(VulkanApp* app);
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
    void destroyInstanceBuffer(NodeID chunkId);
    // If the renderer was initialized with an app, this will be set and
    // allows immediate compute-based generation calls to run against the
    // provided `VulkanApp` instance.
    VulkanApp* appPtr = nullptr;
    // Simple VBO that provides the per-vertex 'base' used by the vegetation
    // pipeline. We use a single base vertex and expand in the shader via
    // the instance data.
    VertexBufferObject billboardVBO;
};
