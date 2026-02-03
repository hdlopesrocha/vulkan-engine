#pragma once

#include "VulkanApp.hpp"
#include "VertexBufferObject.hpp"
#include "../math/BoundingBox.hpp"
#include <vector>
#include <glm/glm.hpp>

// Renders debug wireframe cubes for octree visualization
class DebugCubeRenderer {
public:
    struct CubeWithColor {
        BoundingBox cube;
        glm::vec3 color;
    };

    explicit DebugCubeRenderer(VulkanApp* app);
    ~DebugCubeRenderer();

    // Initialize pipeline and load grid texture
    void init(VkRenderPass renderPassOverride = VK_NULL_HANDLE);

    // Set which cubes to render this frame
    void setCubes(const std::vector<CubeWithColor>& cubes);

    // Render all registered cubes
    void render(VkCommandBuffer& cmd, VkDescriptorSet descriptorSet);

    void cleanup();

private:
    VulkanApp* app = nullptr;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;
    
    // Cube line geometry VBO (shared for all cubes, transformed via push constants)
    VertexBufferObject cubeVBO;
    
    // Grid texture for wireframe look
    VkImage gridTextureImage = VK_NULL_HANDLE;
    VkDeviceMemory gridTextureMemory = VK_NULL_HANDLE;
    VkImageView gridTextureView = VK_NULL_HANDLE;
    VkSampler gridTextureSampler = VK_NULL_HANDLE;
    VkDescriptorSet gridDescriptorSet = VK_NULL_HANDLE;
    VkDescriptorSetLayout gridDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool gridDescriptorPool = VK_NULL_HANDLE;
    
    // Instance data buffer (model matrix + color per cube)
    Buffer instanceBuffer;
    uint32_t instanceBufferCapacity = 0;
    
    // Cubes to render this frame
    std::vector<CubeWithColor> activeCubes;
    
    void createCubeVBO();
    void loadGridTexture();
    void createGridDescriptorSet();
    void updateInstanceBuffer();
};
