#pragma once

#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "../VertexBufferObject.hpp"
#include "../../math/BoundingBox.hpp"
#include <vector>
#include <glm/glm.hpp>
#include "CommandBufferState.hpp"

// Renders debug wireframe cubes for octree visualization
class DebugCubeRenderer {
public:
    struct CubeWithColor {
        BoundingBox cube;
        glm::vec3 color;
    };

    explicit DebugCubeRenderer();
    ~DebugCubeRenderer();

    // Initialize pipeline and load grid texture
    void init(VulkanApp* app);

    // Set which cubes to render this frame
    void setCubes(const std::vector<CubeWithColor>& cubes);

    // Render all registered cubes
    void render(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet descriptorSet);

    void cleanup();

private:
    // VulkanApp is not stored; pass `app` into methods that need it.
    TrackedHandle<VkPipeline> pipeline;
    TrackedHandle<VkPipelineLayout> pipelineLayout;
    TrackedHandle<VkShaderModule> vertModule;
    TrackedHandle<VkShaderModule> fragModule;
    
    // Cube line geometry VBO (shared for all cubes, transformed via push constants)
    VertexBufferObject cubeVBO;
    
    // Grid texture for wireframe look
    VkImage gridTextureImage = VK_NULL_HANDLE;
    VmaAllocation gridTextureAllocation = VK_NULL_HANDLE;
    VkDeviceMemory gridTextureMemory = VK_NULL_HANDLE;
    VkImageView gridTextureView = VK_NULL_HANDLE;
    TrackedHandle<VkSampler> gridTextureSampler;
    TrackedHandle<VkDescriptorSet> gridDescriptorSet;
    TrackedHandle<VkDescriptorSetLayout> gridDescriptorSetLayout;
    TrackedHandle<VkDescriptorPool> gridDescriptorPool;
    
    // Instance data buffer (model matrix + color per cube)
    Buffer instanceBuffer;
    uint32_t instanceBufferCapacity = 0;
    
    // Cubes to render this frame
    std::vector<CubeWithColor> activeCubes;
    CommandBufferState* cmdState = nullptr;
public:
    void setCmdState(CommandBufferState* state) { cmdState = state; }
private:
    void createCubeVBO(VulkanApp* app);
    void loadGridTexture(VulkanApp* app);
    void createGridDescriptorSet(VulkanApp* app);
    void updateInstanceBuffer(VulkanApp* app);
};
