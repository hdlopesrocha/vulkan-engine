#include "SolidRenderer.hpp"
#include "VegetationRenderer.hpp"
#include "WaterRenderer.hpp"
#include "PostProcessRenderer.hpp"
#include "SkyRenderer.hpp"
#include "ShadowRenderer.hpp"
#include "DebugCubeRenderer.hpp"
#include "WireframeRenderer.hpp"
#include "WaterBackFaceRenderer.hpp"
#include "Solid360Renderer.hpp"
#pragma once

// Forward declarations for change handler types
class SolidSpaceChangeHandler;
class LiquidSpaceChangeHandler;

#include <vulkan/vulkan.h>
#include "../VulkanApp.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include "../TextureArrayManager.hpp"
#include "../TextureTriple.hpp"
#include "../MaterialManager.hpp"
#include "../ShaderStage.hpp"
#include "../../utils/FileReader.hpp"
#include "../../math/Vertex.hpp"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <deque>
#include <vector>
#include "../../utils/Model3DVersion.hpp"
#include "SkyRenderer.hpp"
#include "SolidRenderer.hpp"
#include "IndirectRenderer.hpp"
#include "ShadowRenderer.hpp"
#include "WaterRenderer.hpp"
#include "../../utils/UniqueOctreeChangeHandler.hpp"

#include "../ubo/PassUBO.hpp"

class SceneRenderer {
    void addDebugCubeForGeometry(Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom);
public:
    // UBOs for main, shadow, and water passes
    PassUBO<UniformObject> mainPassUBO;
    PassUBO<UniformObject> shadowPassUBO;
    PassUBO<WaterParamsGPU> waterPassUBO;

    // Main uniform buffer
    Buffer mainUniformBuffer;
    
    // Materials SSBO
    Buffer materialsBuffer;
    MaterialManager* materialManagerPtr = nullptr;

    // Water params UBO (binding 7) — stored for descriptor rebinding
    Buffer waterParamsBuffer_;

    // Water render time UBO (binding 10)
    Buffer waterRenderUBOBuffer_;

    // Shadow-specific descriptor set: same as main but binding 4 points to a
    // dummy 1×1 depth image (avoids layout mismatch when shadow map is being written)
    VkDescriptorSet shadowDescriptorSet = VK_NULL_HANDLE;

    std::unique_ptr<SkyRenderer> skyRenderer;
    std::unique_ptr<ShadowRenderer> shadowMapper;
    std::unique_ptr<WaterRenderer> waterRenderer;
    // Scene-owned sub-renderers for water (moved from WaterRenderer)
    std::unique_ptr<WaterBackFaceRenderer> backFaceRenderer;
    std::unique_ptr<Solid360Renderer> solid360Renderer;
    std::unique_ptr<PostProcessRenderer> postProcessRenderer;
    std::unique_ptr<SolidRenderer> solidRenderer;
    std::unique_ptr<VegetationRenderer> vegetationRenderer;
    std::unique_ptr<DebugCubeRenderer> debugCubeRenderer;
    std::unique_ptr<DebugCubeRenderer> boundingBoxRenderer;
    std::unique_ptr<WireframeRenderer> solidWireframe;
    std::unique_ptr<WireframeRenderer> waterWireframe;
    // Sky settings owned by this renderer
    std::unique_ptr<SkySettings> skySettings;
    SkySettings& getSkySettings() { return *skySettings; }

    SceneRenderer();
    ~SceneRenderer();

    // Cleanup and resource destruction (accepts app for Vulkan operations)
    void cleanup(VulkanApp* app);
   
    // Pending change queues (thread-safe)
    struct PendingNode {
        Layer layer;
        OctreeNodeData node;
    };

    // Mutex protecting all chunk maps (solid, transparent, brush) and mesh operations
    std::recursive_mutex chunksMutex;

    // Track model ids for transparent/water meshes so we can remove them if erased/updated
    std::unordered_map<NodeID, Model3DVersion> transparentChunks;

    // Track model ids for opaque/solid meshes (moved from SolidRenderer)
    std::unordered_map<NodeID, Model3DVersion> solidChunks;

    // Brush scene chunk tracking (separate from main scene)
    std::unordered_map<NodeID, Model3DVersion> brushSolidChunks;
    std::unordered_map<NodeID, Model3DVersion> brushTransparentChunks;

    // Debug cubes for nodes (populated by change handlers after geometry generation)
    std::unordered_map<NodeID, DebugCubeRenderer::CubeWithColor> nodeDebugCubes;
    void addDebugCubeForNode(NodeID id, const DebugCubeRenderer::CubeWithColor& cube) { nodeDebugCubes[id] = cube; }
    void removeDebugCubeForNode(NodeID id) { nodeDebugCubes.erase(id); }
    std::vector<DebugCubeRenderer::CubeWithColor> getDebugNodeCubes() const {
        std::vector<DebugCubeRenderer::CubeWithColor> out;
        out.reserve(nodeDebugCubes.size());
        for (const auto &p : nodeDebugCubes) out.push_back(p.second);
        return out;
    }

    // Register/inspect opaque model versions (moved from SolidRenderer)
    void registerModelVersion(NodeID id, const Model3DVersion& ver) { solidChunks[id] = ver; }
    size_t getRegisteredModelCount() const { return solidChunks.size(); }
    const std::unordered_map<NodeID, Model3DVersion>& getNodeModelVersions() const { return solidChunks; }

    // Remove all registered opaque meshes via IndirectRenderer and clear the map
    void removeAllRegisteredMeshes() {
        if (!solidRenderer) return;
        for (auto &entry : solidChunks) {
            if (entry.second.meshId != UINT32_MAX) solidRenderer->getIndirectRenderer().removeMesh(entry.second.meshId);
        }
        solidChunks.clear();
    }

    // Remove all registered transparent/water meshes and clear the map
    void removeAllTransparentMeshes() {
        if (!waterRenderer) return;
        for (auto &entry : transparentChunks) {
            if (entry.second.meshId != UINT32_MAX) waterRenderer->getIndirectRenderer().removeMesh(entry.second.meshId);
        }
        transparentChunks.clear();
    }

    void shadowPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkDescriptorSet mainDescriptorSet, Buffer &mainUniformBuffer, const UniformObject &uboStatic, bool shadowsEnabled, bool vegetationEnabled);
    void skyPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, const UniformObject &uboStatic, const glm::mat4 &viewProj);
    void mainPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &mainPassInfo, uint32_t frameIdx, bool hasWater, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, bool wireframeEnabled, const glm::mat4 &viewProj,
                  const UniformObject &uboStatic, bool normalMappingEnabled, bool tessellationEnabled, bool shadowsEnabled, int debugMode, float triplanarThreshold, float triplanarExponent);
    void waterPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo, uint32_t frameIdx, VkDescriptorSet perTextureDescriptorSet, bool wireframeEnabled, float waterTime, bool skipBackFace = false, VkImageView skyView = VK_NULL_HANDLE, VkImageView cubeReflectionView = VK_NULL_HANDLE);
    void init(VulkanApp* app_, TextureArrayManager* textureArrayManager, MaterialManager* materialManager, const std::vector<WaterParams>& waterParams);
    // Re-update main descriptor set when texture arrays are (re)allocated
    void updateTextureDescriptorSet(VulkanApp* app, TextureArrayManager * textureArrayManager);
    // cleanup declared above (accepts VulkanApp*)

    // Process pending node change queues on the main thread
    void updateMeshForNode(VulkanApp* app, Layer layer, NodeID nid, const OctreeNodeData &nd, const Geometry &geom);

    // Process nodes from a generic per-layer NodeID->OctreeNodeData map
    // Process nodes for a single Layer (nodeMap maps NodeID->OctreeNodeData)
    void processNodeLayer(Scene& scene, Layer layer, NodeID nid, OctreeNodeData& nodeData, const std::function<void(Layer, NodeID, const OctreeNodeData&, const Geometry&)>& onGeometry);

    // Runtime introspection helpers for UI/debug
    size_t getTransparentModelCount();

    // Query whether a model for the given node is already registered
    bool hasModelForNode(Layer layer, NodeID nid) const;

    // Create change handlers pre-bound to this renderer
    SolidSpaceChangeHandler makeSolidSpaceChangeHandler(Scene* scene, VulkanApp* app);
    LiquidSpaceChangeHandler makeLiquidSpaceChangeHandler(Scene* scene, VulkanApp* app);

    // Create change handlers for brush scene (uses brush chunk maps)
    SolidSpaceChangeHandler makeBrushSolidSpaceChangeHandler(Scene* scene, VulkanApp* app);
    LiquidSpaceChangeHandler makeBrushLiquidSpaceChangeHandler(Scene* scene, VulkanApp* app);

    // Remove all brush meshes from GPU and clear brush chunk maps
    void clearBrushMeshes();

    // Resize offscreen resources when the swapchain changes
    void onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height);

    // ── Parallel scene loading ─────────────────────────────────────────────────
    // CPU mesh-generation results are pushed here from the background loading
    // thread; the main (render) thread drains the queue each frame and performs
    // the actual Vulkan uploads.
    struct PendingMeshData {
        Layer          layer;
        NodeID         nid;
        OctreeNodeData nodeData;
        Geometry       geom;
    };
    // Drain the pending mesh queue on the main (render) thread.
    // Call once per frame from update() before recording command buffers.
    void processPendingMeshes(VulkanApp* app, glm::vec3 cameraPos);

private:
    // Callbacks stored here so handler references remain valid
    NodeDataCallback solidNodeEventCallback;
    NodeDataCallback solidNodeEraseCallback;
    NodeDataCallback liquidNodeEventCallback;
    NodeDataCallback liquidNodeEraseCallback;

    // Brush scene callbacks (kept alive so handler references remain valid)
    NodeDataCallback brushSolidNodeEventCallback;
    NodeDataCallback brushSolidNodeEraseCallback;
    NodeDataCallback brushLiquidNodeEventCallback;
    NodeDataCallback brushLiquidNodeEraseCallback;

    // Thread-safe queue fed by the background loading thread
    mutable std::mutex pendingMeshMutex;
    std::deque<PendingMeshData> pendingMeshQueue;
};

// ...existing code...

