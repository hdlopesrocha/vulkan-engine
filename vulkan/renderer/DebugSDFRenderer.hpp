#pragma once

#include "../Buffer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "../../math/BoundingCube.hpp"
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
#include "CommandBufferState.hpp"

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

    TrackedHandle<VkPipeline> pipeline;
    TrackedHandle<VkPipelineLayout> pipelineLayout;
    TrackedHandle<VkShaderModule> vertModule;
    TrackedHandle<VkShaderModule> fragModule;

    Buffer vertexBuffer;
    Buffer indexBuffer;
    uint32_t indexCount = 0;

    TrackedHandle<VkDescriptorSetLayout> descriptorSetLayout;
    TrackedHandle<VkDescriptorPool> descriptorPool;
    TrackedHandle<VkDescriptorSet> descriptorSet;
    Buffer instanceBuffer;
    uint32_t instanceBufferCapacity = 0;

    std::vector<CubeSDF> activeCubes;
    CommandBufferState* cmdState = nullptr;
public:
    void setCmdState(CommandBufferState* state) { cmdState = state; }
private:
    void createCubeBuffers(VulkanApp* app);
    void createDescriptorSet(VulkanApp* app);
    void updateInstanceBuffer(VulkanApp* app);
};
