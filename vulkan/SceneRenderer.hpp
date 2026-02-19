#include "SolidRenderer.hpp"
#include "VegetationRenderer.hpp"
#include "WaterRenderer.hpp"
#include "SkyRenderer.hpp"
#include "ShadowRenderer.hpp"
#include "DebugCubeRenderer.hpp"
#pragma once

// Forward declarations for change handler types
class SolidSpaceChangeHandler;
class LiquidSpaceChangeHandler;

#include <vulkan/vulkan.h>
#include "VulkanApp.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include "TextureArrayManager.hpp"
#include "TextureTriple.hpp"
#include "MaterialManager.hpp"
#include "ShaderStage.hpp"
#include "../utils/FileReader.hpp"
#include "../math/Vertex.hpp"
#include <unordered_map>
#include <memory>
#include "Model3DVersion.hpp"
#include "SkyRenderer.hpp"
#include "SolidRenderer.hpp"
#include "IndirectRenderer.hpp"
#include "ShadowRenderer.hpp"
#include "WaterRenderer.hpp"
#include "../utils/UniqueOctreeChangeHandler.hpp"

#include "PassUBO.hpp"

class SceneRenderer {
    void addDebugCubeForGeometry(Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom);
public:
    // UBOs for main, shadow, and water passes
    PassUBO<UniformObject> mainPassUBO;
    PassUBO<UniformObject> shadowPassUBO;
    PassUBO<WaterParamsGPU> waterPassUBO;

    // No stored VulkanApp*: methods accept VulkanApp* when needed

    // Texture array manager for albedo/normal/bump arrays (owned by application)
    TextureArrayManager* textureArrayManager = nullptr;
    MaterialManager* materialManager = nullptr;

    // Main uniform buffer
    Buffer mainUniformBuffer;
    
    // Materials SSBO
    Buffer materialsBuffer;

    std::unique_ptr<SkyRenderer> skyRenderer;
    std::unique_ptr<ShadowRenderer> shadowMapper;
    std::unique_ptr<WaterRenderer> waterRenderer;
    std::unique_ptr<SolidRenderer> solidRenderer;
    std::unique_ptr<VegetationRenderer> vegetationRenderer;
    std::unique_ptr<DebugCubeRenderer> debugCubeRenderer;
    std::unique_ptr<DebugCubeRenderer> boundingBoxRenderer;
    // Sky settings owned by this renderer
    std::unique_ptr<SkySettings> skySettings;
    SkySettings& getSkySettings() { return *skySettings; }

    SceneRenderer(TextureArrayManager* textureArrayManager_, MaterialManager* materialManager_);
    ~SceneRenderer();

    void createDescriptorSets(MaterialManager &materialManager, TextureArrayManager &textureArrayManager, VkDescriptorSet &outDescriptorSet, VkDescriptorSet &outShadowPassDescriptorSet, size_t tripleCount);

    // Cleanup and resource destruction (accepts app for Vulkan operations)
    void cleanup(VulkanApp* app);

   
    // Pending change queues (thread-safe)
    struct PendingNode {
        Layer layer;
        OctreeNodeData node;
    };

    // Track model ids for transparent/water meshes so we can remove them if erased/updated
    std::unordered_map<NodeID, Model3DVersion> transparentChunks;

    // Track model ids for opaque/solid meshes (moved from SolidRenderer)
    std::unordered_map<NodeID, Model3DVersion> solidChunks;

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

    void shadowPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkQueryPool queryPool, VkDescriptorSet shadowPassDescriptorSet, const UniformObject &uboStatic, bool shadowsEnabled, bool shadowTessellationEnabled);
    void depthPrePass(VkCommandBuffer &commandBuffer, VkQueryPool queryPool);
    void skyPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, const UniformObject &uboStatic, const glm::mat4 &viewProj);
    void mainPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &mainPassInfo, uint32_t frameIdx, bool hasWater, bool vegetationEnabled, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, bool wireframeEnabled, bool profilingEnabled, VkQueryPool queryPool, const glm::mat4 &viewProj,
                  const UniformObject &uboStatic, const WaterParams &waterParams, float waterTime, bool normalMappingEnabled, bool tessellationEnabled, bool shadowsEnabled, int debugMode, float triplanarThreshold, float triplanarExponent);
    void waterPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo, uint32_t frameIdx, VkDescriptorSet perTextureDescriptorSet, bool profilingEnabled, VkQueryPool queryPool, const WaterParams &waterParams, float waterTime);
    void init(VulkanApp* app_);
    // cleanup declared above (accepts VulkanApp*)

    // Incremental change handling (called from SolidSpaceChangeHandler callbacks)
    void onNodeCreated(Layer layer, const OctreeNodeData &node);
    void onNodeUpdated(Layer layer, const OctreeNodeData &node);
    void onNodeErased(Layer layer, const OctreeNodeData &node);

    // Process pending node change queues on the main thread
    void updateMeshForNode(VulkanApp* app, Layer layer, NodeID nid, const OctreeNodeData &nd, const Geometry &geom);

    // Process nodes from a generic per-layer NodeID->OctreeNodeData map
    // Process nodes for a single Layer (nodeMap maps NodeID->OctreeNodeData)
    void processNodeLayer(Scene& scene, Layer layer, NodeID nid, OctreeNodeData& nodeData, const std::function<void(Layer, NodeID, const OctreeNodeData&, const Geometry&)>& onGeometry);

    // Overload: accept raw pending node vector and coalesce inside (per-layer dispatch)
    void processNodes(Scene& scene, const std::vector<PendingNode>& pendingNodes, const std::function<void(Layer, NodeID, const OctreeNodeData&, const Geometry&)>& onGeometry);

    // Process erased node id set for a single Layer
    void processErasedNodeSet(Layer layer, const std::unordered_set<NodeID>& nodeSet, const std::function<void(Layer, NodeID)>& onErased);

    // Overload: accept raw pending erased vector and coalesce inside (per-layer dispatch)
    void processErasedNodeSet(const std::vector<PendingNode>& pendingNodes, const std::function<void(Layer, NodeID)>& onErased);

    // Runtime introspection helpers for UI/debug
    size_t getTransparentModelCount();

    // Query whether a model for the given node is already registered
    bool hasModelForNode(Layer layer, NodeID nid) const;

    // Create change handlers pre-bound to this renderer
    SolidSpaceChangeHandler makeSolidSpaceChangeHandler(Scene* scene, VulkanApp* app);
    LiquidSpaceChangeHandler makeLiquidSpaceChangeHandler(Scene* scene, VulkanApp* app);

    // Resize offscreen resources when the swapchain changes
    void onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height);

private:
    // Callbacks stored here so handler references remain valid
    NodeDataCallback solidNodeEventCallback;
    NodeDataCallback solidNodeEraseCallback;
    NodeDataCallback liquidNodeEventCallback;
    NodeDataCallback liquidNodeEraseCallback;
};

// ...existing code...

