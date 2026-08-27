#pragma once

#include "Renderer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "../VertexBufferObject.hpp"
#include "../../math/BoundingBox.hpp"
#include "../../space/Octree.hpp" // for NodeID
#include <vector>
#include <unordered_map>
#include <mutex>
#include <glm/glm.hpp>
#include "CommandBufferState.hpp"

class IndirectRenderer; // forward declaration: the solid IndirectRenderer owns the merged bbox cull dispatch

// Renders debug wireframe cubes for octree visualization
class DebugCubeRenderer : public Renderer {
public:
    struct CubeWithColor {
        BoundingBox cube;
        glm::vec3 color;
        // LoD meta carried so registerBoundingBoxesToIndirect can build the
        // IndirectRenderer::BBox (with cellSize/level/base) and the shader's
        // clipmap band gate keeps exactly one rung per region.
        float cellSize = 0.0f;
        int level = 0;
        glm::vec3 base = glm::vec3(0.0f);
    };

    explicit DebugCubeRenderer();
    ~DebugCubeRenderer();

    // Initialize pipeline and load grid texture
    void init(VulkanApp* app);

    // Set which cubes to render this frame
    void setCubes(const std::vector<CubeWithColor>& cubes);

    // The solid IndirectRenderer performs the bounding-box frustum cull in its OWN
    // indirect.comp dispatch (folded into the terrain cull). Point this renderer at
    // it so render() draws from the terrain IR's bbox output buffers.
    void setIndirectRenderer(IndirectRenderer* ir) { terrainIR_ = ir; }
    void setCullFrame(uint32_t frame) { currentCullFrame = frame % 3; }

    // Render all registered cubes
    void render(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet descriptorSet);

    void cleanup(VulkanApp* app) override;

    // ── Per-node debug cube tracking (moved from SceneRenderer) ──
    // Populated by the space-change handlers (worker threads) after geometry
    // generation; consumed as a copy on the render thread.
    void addCubeForNode(NodeID id, const CubeWithColor& cube);
    void removeCubeForNode(NodeID id);
    void clearCubes();
    // Render the octree node debug cubes merged with the supplied widget cubes.
    // Reads the per-node cache internally — no external getCubes. The widget cubes
    // (octree explorer overlay) are passed in; the per-node cache is merged here.
    void renderOverlay(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet ds,
                       const std::vector<CubeWithColor>& widgetCubes);

    // ── Per-chunk mesh bounding-box accumulation (processNodeLayer) ──
    // Replaces the chunk's cached box list; consumed flattened on the render
    // thread. Mirrors the SDF-cube path's chunk-keyed caching.
    void setBoundingBoxesForChunk(NodeID id, const std::vector<CubeWithColor>& cubes);
    void clearBoundingBoxes();
    // Register the cached per-chunk mesh bounding boxes with the solid
    // IndirectRenderer's merged cull dispatch (folds box AABBs into the terrain
    // frustum cull) and upload the instance payload. Reads the cache internally.
    // Call OUTSIDE a render pass, before the solid IR's prepareCull.
    void registerBoundingBoxesToIndirect();
    // Drop the active bounding-box inputs (overlay toggled off).
    void clearBoundingBoxesToIndirect();

private:
    // Per-node debug cubes keyed by octree node id
    std::unordered_map<NodeID, CubeWithColor> nodeDebugCubes;
    // Per-chunk mesh bounding boxes keyed by chunk node id
    std::unordered_map<NodeID, std::vector<CubeWithColor>> chunkBBoxCubes;
    mutable std::recursive_mutex cubesMutex;

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
    // Number of instances actually uploaded/drawn this frame (capped to the
    // device's maxStorageBufferRange so the storage buffer binding stays valid).
    uint32_t drawInstanceCount = 0;

    // Cubes to render this frame
    std::vector<CubeWithColor> activeCubes;

    // ── Per-frame GPU cull integration ──
    // The bounding-box frustum cull is performed by the solid IndirectRenderer's
    // merged indirect.comp dispatch; render() reads the surviving DrawCmd stream
    // + count from terrainIR_ (frame-matched via currentCullFrame). The instance
    // payload (model + color) is still uploaded here, indexed by gl_InstanceIndex.
    IndirectRenderer* terrainIR_ = nullptr;
    uint32_t currentCullFrame = 0;
    PFN_vkCmdDrawIndexedIndirectCountKHR cmdDrawIndexedIndirectCount = nullptr;
private:
    void createCubeVBO(VulkanApp* app);
    void loadGridTexture(VulkanApp* app);
    void createGridDescriptorSet(VulkanApp* app);
    void updateInstanceBuffer(VulkanApp* app);
};
