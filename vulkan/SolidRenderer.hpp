#pragma once

#include "VulkanApp.hpp"
#include "IndirectRenderer.hpp"
#include "Model3DVersion.hpp"
#include "ShaderStage.hpp"
#include "../math/Vertex.hpp"
#include "../utils/Scene.hpp"
#include <unordered_map>
#include <vector>

class SolidRenderer {
public:
    explicit SolidRenderer(VulkanApp* app_ = nullptr);
    ~SolidRenderer();

    void init(VulkanApp* app_);
    void createPipelines();
    void cleanup();

    // Depth pre-pass that binds depthPrePassPipeline and draws indirect only
    void depthPrePass(VkCommandBuffer &commandBuffer, VkQueryPool queryPool);

    // Draw main solid geometry: bind pipeline (wireframe or normal) and draw
    void draw(VkCommandBuffer &commandBuffer, VulkanApp* app, VkDescriptorSet perTextureDescriptorSet, bool wireframeEnabled);

    // Access for adding meshes
    IndirectRenderer& getIndirectRenderer() { return indirectRenderer; }

    // Register model version map managed here
    void registerModelVersion(NodeID id, const Model3DVersion& ver) { nodeModelVersions[id] = ver; }

    // Remove all registered meshes and clear map
    void removeAllRegisteredMeshes() {
        for (auto &entry : nodeModelVersions) {
            if (entry.second.meshId != UINT32_MAX) indirectRenderer.removeMesh(entry.second.meshId);
        }
        nodeModelVersions.clear();
    }

    // Get count of registered models
    size_t getRegisteredModelCount() const { return nodeModelVersions.size(); }

private:
    VulkanApp* app = nullptr;
    IndirectRenderer indirectRenderer;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkPipeline graphicsPipelineWire = VK_NULL_HANDLE;
    VkPipeline depthPrePassPipeline = VK_NULL_HANDLE;

    std::unordered_map<NodeID, Model3DVersion> nodeModelVersions;
};
