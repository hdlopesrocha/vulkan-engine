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
#include <deque>
#include <unordered_map>
#include <array>
#include <mutex>
#include <glm/glm.hpp>
#include <glm/gtc/round.hpp>

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
        float impostorDistance = 0.0f; // formerly pad0; 0 = impostor rendering disabled
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
    void init(VulkanApp* app);
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
    // CPU-side instance generation — avoids GPUVM faults on RADV iGPUs where
    // the Texture Cache/Pipe cannot read from device-local or host-visible
    // storage buffers.  Enqueues the chunk and processes up to maxPerFrame
    // chunks each frame via processPendingChunks().
    void generateChunkInstancesCPU(NodeID chunkId,
                                   const std::vector<glm::vec3>& positions,
                                   const std::vector<uint32_t>& grassIndices,
                                   const glm::vec3& chunkCenter,
                                   uint32_t instancesPerTriangle, VulkanApp* app,
                                   uint32_t seed = 1337);
    // Drain up to maxChunks from the pending queue.  Call every frame from
    // draw() so chunks trickle in at a controlled rate.
    void processPendingChunks(uint32_t maxChunks);
    // Number of chunks still waiting in the queue.
    size_t pendingChunkCount() const;
    void clearAllInstances();

    // Draw all visible vegetation chunks with GPU frustum culling.
    // If queryPool != VK_NULL_HANDLE, writes GPU timestamps:
    //   queryRealIndex .. queryRealIndex+1  = real billboard passes (depth prepass + shading)
    //   queryImpostorIndex .. queryImpostorIndex+1 = impostor passes (impostor depth + color)
    void draw(VulkanApp* app, VkCommandBuffer& commandBuffer, VkDescriptorSet vegetationDescriptorSet,
              const glm::mat4& viewProj, const glm::vec3& cameraPos,
              VkQueryPool queryPool = VK_NULL_HANDLE,
              uint32_t queryRealIndex = 0,
              uint32_t queryImpostorIndex = 0);
    void recordReadBarriers(VkCommandBuffer& commandBuffer);
    
    // Draw vegetation to shadow map using light-space matrix in the bound UBO.
    // Camera position is used for distance-based LOD; viewProj is the camera's
    // view-projection for GPU frustum culling (matching solid shadow culling).
    void drawShadow(VulkanApp* app, VkCommandBuffer& commandBuffer, VkDescriptorSet shadowDescriptorSet, const glm::mat4& viewProj, const glm::vec3& cameraPos);

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

    // Impostor rendering.  Call after init() once impostor views have been captured.
    // albedoArray60 and normalArray60 must be VkImageView covering 60 layers
    // (3 billboard types × 20 Fibonacci views).
    // depthArray60 is the captured device Z array (R32_SFLOAT, 60 layers) for depth reprojection.
    // captureInvVPBuf is a storage buffer containing per-layer inverse VP matrices.
    void setImpostorData(VulkanApp* app,
                         VkImageView albedoArray60,
                         VkImageView normalArray60,
                         VkSampler sampler,
                         VkImageView depthArray60 = VK_NULL_HANDLE,
                         VkBuffer   captureInvVPBuf = VK_NULL_HANDLE);

    // Distance beyond which vegetation instances are replaced by impostor quads.
    // Set to 0 (default) to disable impostor rendering entirely.
    void setImpostorDistance(float dist) { impostorDistance = dist; }

    // Build the concatenated instance buffer and per-chunk metadata for GPU
    // frustum culling. Must be called once after all chunks are generated.
    // Uses a temporary command buffer (synchronous, one-time cost).
    void consolidateChunks(VulkanApp* app);

    // GPU frustum culling: dispatch compute shader that culls chunks against
    // viewProj and compacts visible draw commands. Must be called OUTSIDE any
    // render pass (compute dispatches are illegal inside dynamic rendering).
    void prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj);

private:
    
    VkPipeline vegetationPipeline = VK_NULL_HANDLE;          // shading pass (depthWrite=false, EQUAL)
    VkPipeline vegetationDepthPipeline = VK_NULL_HANDLE;     // depth-only prepass
    VkPipelineLayout vegetationDepthPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline vegetationShadowPipeline = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
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
    void destroyInstanceBuffer(NodeID chunkId, VulkanApp* app = nullptr, VkFence completionFence = VK_NULL_HANDLE);

    // Pending CPU-generation queue — chunks are enqueued by the scene loader
    // and drained 10-per-frame by draw().
    struct PendingChunk {
        NodeID chunkId;
        std::vector<glm::vec3> positions;
        std::vector<uint32_t> grassIndices;
        glm::vec3 chunkCenter;
        uint32_t instancesPerTriangle;
        uint32_t seed;
    };
    std::deque<PendingChunk> pendingChunks;
    mutable std::mutex pendingChunksMutex;
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

    // Impostor pipeline resources (populated via setImpostorData).
    VkPipeline            impostorPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      impostorPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout impostorDescSetLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      impostorDescPool       = VK_NULL_HANDLE;
    VkDescriptorSet       impostorDescSet        = VK_NULL_HANDLE;

    // Impostor depth pipeline (shadow map depth-only pass).
    VkPipeline            impostorDepthPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      impostorDepthPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout impostorDepthDescSetLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      impostorDepthDescPool       = VK_NULL_HANDLE;
    VkDescriptorSet       impostorDepthDescSet        = VK_NULL_HANDLE;

    float                 impostorDistance       = 0.0f;
    VkRenderPass          storedSolidRenderPass  = VK_NULL_HANDLE;

    // ── GPU frustum culling (indirection via concatenated instance buffer) ──────
    struct ChunkMeta {
        glm::vec3 aabbMin;
        float pad0;
        glm::vec3 aabbMax;
        float pad1;
        uint32_t instanceOffset;
        uint32_t instanceCount;
    };

    Buffer concatenatedInstanceBuffer;  // all instances concatenated (vec4 per element)
    Buffer chunkMetaBuffer;             // ChunkMeta[] on GPU
    Buffer compactedCmdBuffer;          // output VkDrawIndirectCommand[] (compacted visible chunks)
    Buffer visibleCountBuffer;          // atomic counter (uint32_t)
    uint32_t* visibleCountMapped = nullptr;

    // Culling compute pipeline
    VkPipeline            vegCullPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      vegCullPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout vegCullDescSetLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      vegCullDescPool       = VK_NULL_HANDLE;
    VkDescriptorSet       vegCullDescSet        = VK_NULL_HANDLE;

    uint32_t vegNumChunks = 0;             // number of chunks in the consolidated metadata
    bool vegConsolidationDirty = true;     // rebuild concatenated buffer + metadata

    void initCulling(VulkanApp* app);
    void destroyCulling();
};
