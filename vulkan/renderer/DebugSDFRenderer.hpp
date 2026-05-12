#pragma once

#include "../Buffer.hpp"
#include "../VulkanApp.hpp"
#include "../../math/BoundingCube.hpp"
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

// Renders leaf-node cube faces colored by their SDF sign and magnitude.
class DebugSDFRenderer {
public:
    struct CubeSDF {
        BoundingCube cube;
        std::array<float, 8> sdf;
        int brushIndex;
    };

    DebugSDFRenderer();
    ~DebugSDFRenderer();

    void init(VulkanApp* app);
    void setCubes(const std::vector<CubeSDF>& cubes);
    void render(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet descriptorSet);
    void cleanup();

private:
    struct CubeVertex {
        glm::vec3 position;
        uint32_t cornerIndex;
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;

    Buffer vertexBuffer;
    Buffer indexBuffer;
    uint32_t indexCount = 0;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    Buffer instanceBuffer;
    uint32_t instanceBufferCapacity = 0;

    std::vector<CubeSDF> activeCubes;

    void createCubeBuffers(VulkanApp* app);
    void createDescriptorSet(VulkanApp* app);
    void updateInstanceBuffer(VulkanApp* app);
};
