#pragma once
#include "../VulkanApp.hpp"
#include "../TextureArrayManager.hpp"
#include "../EditableTexture.hpp"
#include "../../math/Vertex.hpp"
#include "DebugCubeRenderer.hpp"
#include "../../utils/BillboardManager.hpp"
#include "../VertexBufferObject.hpp"
#include "../../utils/Scene.hpp" // for NodeID
#include <vector>
#include <unordered_map>
#include <array>
#include <glm/glm.hpp>

// Per-chunk vegetation instance buffer and renderer
class VegetationRenderer {
public:
    struct WindSettings {
        bool enabled = true;
        glm::vec2 direction = glm::vec2(1.0f, 0.0f);
        float strength = 4.0f;
        float baseFrequency = 0.003f;
        float speed = 0.75f;
        float gustFrequency = 0.012f;
        float gustStrength = 0.45f;
        float skewAmount = 1.75f;
        float trunkStiffness = 0.70f;
        float noiseScale = 1.0f;
        float verticalFlutter = 0.20f;
        float turbulence = 0.60f;
    };

    struct WindPushConstants {
        float billboardScale = 1.0f;
        float windEnabled = 1.0f;
        float windTime = 0.0f;
        float pad0 = 0.0f;
        glm::vec4 windDirAndStrength = glm::vec4(1.0f, 0.0f, 0.0f, 4.0f);
        glm::vec4 windNoise = glm::vec4(0.003f, 0.75f, 0.012f, 0.45f);
        glm::vec4 windShape = glm::vec4(1.75f, 0.70f, 1.0f, 0.20f);
        glm::vec4 windTurbulence = glm::vec4(0.60f, 0.0f, 0.0f, 0.0f);
        glm::vec4 densityParams = glm::vec4(1.0f, 512.0f, 4096.0f, 0.10f);
        glm::vec4 cameraPosAndFalloff = glm::vec4(0.0f);
    };

    struct DistanceDensitySettings {
        bool enabled = true;
        float fullDensityDistance = 512.0f;
        float minDensityDistance = 4096.0f;
        float minDensityFactor = 0.10f;
    };

    float billboardScale = 10.0f;
    uint32_t billboardCount = 3; // number of billboard texture variants (3 = foliage/grass/wild)
    explicit VegetationRenderer();
    ~VegetationRenderer();

    void setTextureArrayManager(TextureArrayManager* mgr, VulkanApp* app);
    void setBillboardArrayTextures(VkImageView albedoView, VkImageView normalView, VkImageView opacityView, VkSampler sampler, VulkanApp* app);
    void onTextureArraysReallocated(VulkanApp* app);
    void init();
    void cleanup();
    void init(VulkanApp* app, VkRenderPass renderPassOverride = VK_NULL_HANDLE);
    // Generate per-chunk vegetation instances from mesh geometry using the
    // compute shader. This is the only supported instancing path now.
    // vertexBuffer/indexBuffer are device-local buffers created by the caller.
    // We accept `Buffer` objects so the renderer can defer destruction until GPU work completes.
    void generateChunkInstances(NodeID chunkId,
                                Buffer vertexBuffer, uint32_t vertexCount,
                                Buffer indexBuffer, uint32_t indexCount,
                                const glm::vec3& chunkCenter,
                                uint32_t instancesPerTriangle, VulkanApp* app,
                                uint32_t seed = 1337);
    void clearAllInstances();

    // Draw all visible vegetation chunks (frustum culling is per-chunk, matching geometry)
    void draw(VulkanApp* app, VkCommandBuffer& commandBuffer, VkDescriptorSet vegetationDescriptorSet, const glm::mat4& viewProj, const glm::vec3& cameraPos);

    // Stats helpers
    size_t getChunkCount() const { return chunkInstanceCounts.size(); }
    size_t getInstanceTotal() const;

    WindSettings& getWindSettings() { return windSettings; }
    const WindSettings& getWindSettings() const { return windSettings; }
    DistanceDensitySettings& getDistanceDensitySettings() { return distanceDensitySettings; }
    const DistanceDensitySettings& getDistanceDensitySettings() const { return distanceDensitySettings; }
    void setWindTime(float timeSeconds) { windTimeSeconds = timeSeconds; }
    float computeDensityFactor(float distanceToCamera) const;
    std::vector<DebugCubeRenderer::CubeWithColor> getDensityDebugCubes(const glm::vec3& cameraPos) const;
    float getAverageDensityFactor(const glm::vec3& cameraPos) const;

private:
    
    VkPipeline vegetationPipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    TextureArrayManager* vegetationTextureArrayManager = nullptr;
    VkImageView billboardAlbedoView   = VK_NULL_HANDLE;
    VkImageView billboardNormalView   = VK_NULL_HANDLE;
    VkImageView billboardOpacityView  = VK_NULL_HANDLE;
    VkSampler   billboardArraySampler = VK_NULL_HANDLE;

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
        glm::vec3 center = glm::vec3(0.0f);
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

    WindSettings windSettings;
    DistanceDensitySettings distanceDensitySettings;
    float windTimeSeconds = 0.0f;
};
