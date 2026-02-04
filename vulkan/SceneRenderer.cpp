#include "SceneRenderer.hpp"

#include "../utils/SolidSpaceChangeHandler.hpp"
#include "../utils/LiquidSpaceChangeHandler.hpp"

void SceneRenderer::cleanup() {
    // Cleanup all sub-renderers to properly destroy GPU resources
    if (waterRenderer) {
        waterRenderer->cleanup();
    }
    if (solidRenderer) {
        solidRenderer->cleanup();
    }
    if (shadowMapper) {
        shadowMapper->cleanup();
    }
    if (skyRenderer) {
        skyRenderer->cleanup();
    }
    if (vegetationRenderer) {
        vegetationRenderer->cleanup();
    }
    if (debugCubeRenderer) {
        debugCubeRenderer->cleanup();
    }
    if (boundingBoxRenderer) {
        boundingBoxRenderer->cleanup();
    }
}

void SceneRenderer::onSwapchainResized(uint32_t width, uint32_t height) {
    // Recreate offscreen targets that depend on swapchain size
    if (solidRenderer) {
        solidRenderer->createRenderTargets(width, height);
    }
    if (waterRenderer) {
        waterRenderer->createRenderTargets(width, height);
    }
}

SceneRenderer::SceneRenderer(VulkanApp* app_, TextureArrayManager* textureArrayManager_, MaterialManager* materialManager_)
    : app(app_),
      textureArrayManager(textureArrayManager_),
      materialManager(materialManager_),
      shadowMapper(std::make_unique<ShadowRenderer>(app_, 8192)),
      waterRenderer(std::make_unique<WaterRenderer>(app_)),
      skyRenderer(std::make_unique<SkyRenderer>(app_)),
      solidRenderer(std::make_unique<SolidRenderer>(app_)),
      vegetationRenderer(std::make_unique<VegetationRenderer>(app_)),
      debugCubeRenderer(std::make_unique<DebugCubeRenderer>(app_)),
      boundingBoxRenderer(std::make_unique<DebugCubeRenderer>(app_)),
      skySettings()
{
    // All renderer members are now properly instantiated and internal SkySettings constructed

    // Initialize callbacks bound to this renderer so they outlive handler instances
    solidNodeEventCallback = [this](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        if (hasModelForNode(LAYER_OPAQUE, nid)) {
            onNodeUpdated(LAYER_OPAQUE, nd);
        } else {
            onNodeCreated(LAYER_OPAQUE, nd);
        }
    };
    solidNodeEraseCallback = [this](const OctreeNodeData& nd) {
        onNodeErased(LAYER_OPAQUE, nd);
    };

    liquidNodeEventCallback = [this](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        if (hasModelForNode(LAYER_TRANSPARENT, nid)) {
            onNodeUpdated(LAYER_TRANSPARENT, nd);
        } else {
            onNodeCreated(LAYER_TRANSPARENT, nd);
        }
    };
    liquidNodeEraseCallback = [this](const OctreeNodeData& nd) {
        onNodeErased(LAYER_TRANSPARENT, nd);
    };
}

SceneRenderer::~SceneRenderer() {
    cleanup();
}

void SceneRenderer::shadowPass(VkCommandBuffer &commandBuffer, VkQueryPool queryPool, VkDescriptorSet shadowPassDescriptorSet, const UniformObject &uboStatic, bool shadowsEnabled, bool shadowTessellationEnabled) {
    if (commandBuffer == VK_NULL_HANDLE) {
        fprintf(stderr, "[SceneRenderer::shadowPass] commandBuffer is VK_NULL_HANDLE, skipping.\n");
        return;
    }
    //fprintf(stderr, "[SceneRenderer::shadowPass] Entered.\n");
    glm::mat4 lightSpaceMatrix = glm::mat4(1.0f); // TODO: Replace with actual light space matrix
    // Record shadow pass on a temporary command buffer to avoid nested render passes
    if (!shadowsEnabled) return;

    VkCommandBuffer cmd = app->beginSingleTimeCommands();
    shadowMapper->beginShadowPass(cmd, lightSpaceMatrix);
    // TODO: Render objects to shadow map using shadowMapper->renderObject(...)
    // End and transition
    shadowMapper->endShadowPass(cmd);
    app->endSingleTimeCommands(cmd);

}

void SceneRenderer::mainPass(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &mainPassInfo, uint32_t frameIdx, bool hasWater, bool vegetationEnabled, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, bool wireframeEnabled, bool profilingEnabled, VkQueryPool queryPool, const glm::mat4 &viewProj,
                  const UniformObject &uboStatic, const WaterParams &waterParams, float waterTime, bool normalMappingEnabled, bool tessellationEnabled, bool shadowsEnabled, int debugMode, float triplanarThreshold, float triplanarExponent) {
    if (commandBuffer == VK_NULL_HANDLE) {
        fprintf(stderr, "[SceneRenderer::mainPass] commandBuffer is VK_NULL_HANDLE, skipping.\n");
        return;
    }
    (void)hasWater;
    static bool printedOnce = false;
    if (!printedOnce) {
        fprintf(stderr, "[SceneRenderer::mainPass] Entered. solidRenderer=%p skyRenderer=%p\n", (void*)solidRenderer.get(), (void*)skyRenderer.get());
        printedOnce = true;
    }
    if (!solidRenderer) {
        fprintf(stderr, "[SceneRenderer::mainPass] solidRenderer is nullptr, skipping.\n");
        return;
    }
    solidRenderer->draw(commandBuffer, app, perTextureDescriptorSet, wireframeEnabled);
    //fprintf(stderr, "[SceneRenderer::mainPass] After solidRenderer->draw.\n");

    // Vegetation pass
    if (!vegetationRenderer) {
        fprintf(stderr, "[SceneRenderer::mainPass] vegetationRenderer is nullptr, skipping.\n");
    } else if (vegetationEnabled) {
        try {
            vegetationRenderer->draw(commandBuffer, perTextureDescriptorSet, viewProj);
            //fprintf(stderr, "[SceneRenderer::mainPass] After vegetationRenderer->draw.\n");
        } catch (...) {
            fprintf(stderr, "[SceneRenderer::mainPass] Exception in vegetationRenderer->draw.\n");
        }
    }
}

void SceneRenderer::skyPass(VkCommandBuffer &commandBuffer, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, const UniformObject &uboStatic, const glm::mat4 &viewProj) {
    if (!skyRenderer) {
        fprintf(stderr, "[SceneRenderer::skyPass] skyRenderer is nullptr, skipping.\n");
        return;
    }
    try {
        // skySettings owned by this renderer
        SkySettings::Mode mode = skySettings.mode;
        skyRenderer->render(commandBuffer, perTextureDescriptorSet, mainUniformBuffer, uboStatic, viewProj, mode);
    } catch (...) {
        fprintf(stderr, "[SceneRenderer::skyPass] Exception in skyRenderer->render.\n");
    }
}

void SceneRenderer::waterPass(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo, uint32_t frameIdx, VkDescriptorSet perTextureDescriptorSet, bool profilingEnabled, VkQueryPool queryPool, const WaterParams &waterParams, float waterTime) {
    static int frameCount = 0;
    if (frameCount++ == 0) {
        printf("[DEBUG] WaterRenderer::waterPass called for the first time\n");
    }
    
    if (commandBuffer == VK_NULL_HANDLE) {
        fprintf(stderr, "[SceneRenderer::waterPass] commandBuffer is VK_NULL_HANDLE, skipping.\n");
        return;
    }

    // Ensure per-frame scene textures are bound to the water descriptor sets
    // Update the descriptor sets after the main scene pass so images are in SHADER_READ_ONLY layout
    // Bind the solid offscreen outputs so water uses the solid pass for refraction/foam
    VkImageView sceneColorView = solidRenderer ? solidRenderer->getColorView(frameIdx) : VK_NULL_HANDLE;
    VkImageView sceneDepthView = solidRenderer ? solidRenderer->getDepthView(frameIdx) : VK_NULL_HANDLE;
    waterRenderer->updateSceneTexturesBinding(sceneColorView, sceneDepthView, frameIdx);

    // Run water geometry pass offscreen on a temporary command buffer to avoid nested render passes
    VkCommandBuffer cmd = app->beginSingleTimeCommands();
    waterRenderer->beginWaterGeometryPass(cmd, frameIdx);
    // TODO: Render water geometry here if needed (use waterRenderer APIs)
    waterRenderer->endWaterGeometryPass(cmd);
    app->endSingleTimeCommands(cmd);

    // Post-processing should run inside the active main render pass; caller (e.g. MyApp::draw) should invoke
    // `waterRenderer->renderWaterPostProcess` with valid scene color/depth views when available. Keep this function focused
    // on executing offscreen geometry and returning control to the main pass.
}


void SceneRenderer::createPipelines() {
    // Solid and vegetation have public createPipelines
    if (solidRenderer) solidRenderer->createPipelines();

    // WaterRenderer initialization (requires a Buffer for params and render targets) is performed elsewhere via WaterRenderer::init(Buffer&)

    // SkyRenderer pipelines must match the solid render pass
    if (skyRenderer) skyRenderer->init(solidRenderer ? solidRenderer->getRenderPass() : VK_NULL_HANDLE);



    // Water pipeline creation requires initialization with buffers/targets and is handled by WaterRenderer::init()/createRenderTargets elsewhere
    // Shadow pipeline creation is performed during ShadowRenderer::init()
    shadowMapper->init();

    if (vegetationRenderer) vegetationRenderer->createPipelines(solidRenderer ? solidRenderer->getRenderPass() : VK_NULL_HANDLE);
}

void SceneRenderer::init(VulkanApp* app_, VkDescriptorSet descriptorSet) {
    if (!app_) {
        fprintf(stderr, "[SceneRenderer::init] app_ is nullptr!\n");
        return;
    }
    app = app_;
    // skySettingsRef was initialized at construction and must be valid
    
    // Bind external texture arrays if provided; allocation/initialization should be done by the application
    if (vegetationRenderer) {
        if (textureArrayManager) {
            vegetationRenderer->setTextureArrayManager(textureArrayManager);
            vegetationRenderer->init(app);
        } else {
            fprintf(stderr, "[SceneRenderer::init] No TextureArrayManager provided — vegetation renderer initialization deferred\n");
        }
    }
    
    if (solidRenderer) {
        solidRenderer->init(app);
        solidRenderer->createRenderTargets(app->getWidth(), app->getHeight());
    }

    // Create pipelines for all renderers (solid renderer now has its render pass ready)
    createPipelines();
    
    // Initialize debug cube renderer
    if (debugCubeRenderer) {
        debugCubeRenderer->init(solidRenderer ? solidRenderer->getRenderPass() : VK_NULL_HANDLE);
    }
    // Initialize bounding box renderer (reuses cube wireframe pipeline)
    if (boundingBoxRenderer) {
        boundingBoxRenderer->init(solidRenderer ? solidRenderer->getRenderPass() : VK_NULL_HANDLE);
    }
    
    // Create main uniform buffer
    mainUniformBuffer = app->createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    VkDescriptorSet mainDs = app->getMainDescriptorSet();
    printf("[SceneRenderer::init] mainDescriptorSet = 0x%llx\n", (unsigned long long)mainDs);

    // Initialize sky renderer with our owned settings now that descriptor sets are ready
    if (skyRenderer) {
        skyRenderer->initSky(skySettings, mainDs);
    }
    
    // Bind main uniform buffer into the app's main descriptor set (binding 0)
    VkDescriptorBufferInfo mainBufInfo{ mainUniformBuffer.buffer, 0, sizeof(UniformObject) };
    VkWriteDescriptorSet uboWrite{};
    uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    uboWrite.dstSet = mainDs;
    uboWrite.dstBinding = 0;
    uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboWrite.descriptorCount = 1;
    uboWrite.pBufferInfo = &mainBufInfo;
    
    printf("[SceneRenderer::init] Binding UBO: buffer=%p, binding=0, descriptorSet=%p\n", 
           (void*)mainUniformBuffer.buffer, (void*)mainDs);
    
    // Bind texture arrays (bindings 1, 2, 3)
    // Prepare descriptor writes dynamically: only include image samplers that have valid image views
    std::vector<VkWriteDescriptorSet> writes;

    // UBO write (always present)
    writes.push_back(uboWrite);

    // Helper to add image write if valid
    auto addImageWrite = [&](uint32_t binding, VkSampler sampler, VkImageView view, VkImageLayout layout) {
        if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) {
            fprintf(stderr, "[SceneRenderer::init] Skipping descriptor binding %u: imageView=%p sampler=%p\n", binding, (void*)view, (void*)sampler);
            return;
        }
        VkDescriptorImageInfo* info = new VkDescriptorImageInfo();
        info->sampler = sampler;
        info->imageView = view;
        info->imageLayout = layout;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = mainDs;
        w.dstBinding = binding;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo = info;
        writes.push_back(w);
    };

    if (textureArrayManager) {
        addImageWrite(1, textureArrayManager->albedoSampler, textureArrayManager->albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addImageWrite(2, textureArrayManager->normalSampler, textureArrayManager->normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addImageWrite(3, textureArrayManager->bumpSampler, textureArrayManager->bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else {
        fprintf(stderr, "[SceneRenderer::init] No TextureArrayManager set — skipping texture array descriptor writes\n");
    }
    addImageWrite(4, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

    // Create and bind Materials SSBO at binding 5. Prefer external MaterialManager if provided.
    Buffer materialsBuf;
    bool materialsOwnedLocally = false;
    if (materialManager) {
        materialsBuf = materialManager->getBuffer();
    } else {
        materialsBuf = app->createBuffer(sizeof(MaterialGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        materialsOwnedLocally = true;
    }
    VkDescriptorBufferInfo materialsInfo{};
    materialsInfo.buffer = materialsBuf.buffer;
    materialsInfo.offset = 0;
    materialsInfo.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet materialsWrite{};
    materialsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    materialsWrite.dstSet = mainDs;
    materialsWrite.dstBinding = 5;
    materialsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    materialsWrite.descriptorCount = 1;
    materialsWrite.pBufferInfo = &materialsInfo;
    writes.push_back(materialsWrite);
    // Note: if we allocated materialsBuf locally above we should free it in cleanup; that plumbing is omitted for brevity.

    // Perform descriptor update (clean up temporary image infos afterwards)
    app->updateDescriptorSet(mainDs, writes);
    for (auto &w : writes) {
        if (w.pImageInfo) delete w.pImageInfo;
    }

    // Initialize WaterRenderer
    Buffer waterParamsBuffer = app->createBuffer(sizeof(WaterUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    waterRenderer->init(waterParamsBuffer);
    waterRenderer->createRenderTargets(app->getWidth(), app->getHeight());
    
    // Initialize sky renderer with sphere VBO now that descriptor sets are ready
    if (skyRenderer) {
        skyRenderer->initSky(skySettings, mainDs);
    }
    
    printf("[SceneRenderer::init] Initialization complete\n");
}

#include "../utils/LocalScene.hpp"

#include <mutex>


void SceneRenderer::onNodeCreated(Layer layer, const OctreeNodeData &node) {
    std::lock_guard<std::mutex> lock(pendingMutex);
    pendingCreated.push_back(PendingNode{layer, node});
}

void SceneRenderer::onNodeUpdated(Layer layer, const OctreeNodeData &node) {
    std::lock_guard<std::mutex> lock(pendingMutex);
    pendingUpdated.push_back(PendingNode{layer, node});
}

void SceneRenderer::onNodeErased(Layer layer, const OctreeNodeData &node) {
    std::lock_guard<std::mutex> lock(pendingMutex);
    pendingErased.push_back(PendingNode{layer, node});
}

void SceneRenderer::processNodeLayer(Scene& scene, Layer layer, NodeID nid, OctreeNodeData& nodeData, const std::function<void(Layer, NodeID, const OctreeNodeData&, const Geometry&)>& onGeometry) {

    // Make a local copy of the node data so the callback may safely outlive this stack frame
    OctreeNodeData nodeCopy = nodeData;
    // Capture nodeCopy by value so async callbacks receive a safe copy
    scene.requestModel3D(layer, nodeCopy, [this, layer, nid, nodeCopy, &onGeometry](const Geometry &geom) {
        //std::cout << "[SceneRenderer::processNodeLayer] Received geometry for layer=" << static_cast<int>(layer) << " nid=" << nid << " with " << geom.vertices.size() << " vertices and " << geom.indices.size() << " indices.\n";
        onGeometry(layer, nid, nodeCopy, geom);
    });
    
}

// Return Solid/Liquid change handlers that reference the callbacks stored on this object
SolidSpaceChangeHandler SceneRenderer::makeSolidSpaceChangeHandler() const {
    return SolidSpaceChangeHandler(solidNodeEventCallback, solidNodeEraseCallback);
}

LiquidSpaceChangeHandler SceneRenderer::makeLiquidSpaceChangeHandler() const {
    return LiquidSpaceChangeHandler(liquidNodeEventCallback, liquidNodeEraseCallback);
}

// Accept pending nodes and coalesce into per-layer maps, then call the map-based processor
void SceneRenderer::processNodes(Scene& scene, const std::vector<PendingNode>& pendingNodes, const std::function<void(Layer, NodeID, const OctreeNodeData&, const Geometry&)>& onGeometry) {
    //std::cout << "[SceneRenderer::processNodes] Processing " << pendingNodes.size() << " pending nodes.\n";
    std::unordered_map<NodeID, std::pair<Layer, OctreeNodeData>> uniqueMap;
    for (const auto &p : pendingNodes) {
        NodeID nid = reinterpret_cast<NodeID>(p.node.node);
        uniqueMap[nid] = std::make_pair(p.layer, p.node); // last write wins
    }

    size_t total = 0;
    for (auto &layerPair : uniqueMap) {
        Layer layer = layerPair.second.first;
        NodeID nid = layerPair.first;
        OctreeNodeData& nodeData = layerPair.second.second;
        //fprintf(stdout, "[SceneRenderer::processNodes] unique layer=%d count=1\n", static_cast<int>(layer));
        ++total;
//void SceneRenderer::processNodeLayer(Scene* scene, Layer layer, NodeID nid, OctreeNodeData& nodeData, const std::function<void(Layer, NodeID, const OctreeNodeData&, const Geometry&)>& onGeometry) {;
        
        processNodeLayer(scene, layer, nid, nodeData, onGeometry);
    }
    //fprintf(stdout, "[SceneRenderer::processNodes] totals: unique=%zu transparentTracked=%zu\n", total, transparentChunks.size());
}

// Accept pending erased nodes, coalesce per-layer NodeID set, then call map-based processor
void SceneRenderer::processErasedNodeSet(const std::vector<PendingNode>& pendingNodes, const std::function<void(Layer, NodeID)>& onErased) {
    std::unordered_map<Layer, std::unordered_set<NodeID>> uniqueSet;
    for (const auto &p : pendingNodes) {
        NodeID nid = reinterpret_cast<NodeID>(p.node.node);
        uniqueSet[p.layer].insert(nid);
    }

    size_t total = 0;
    for (auto &layerPair : uniqueSet) {
        Layer layer = layerPair.first;
        //fprintf(stdout, "[SceneRenderer::processErasedNodeSet] unique erased layer=%d count=%zu\n", static_cast<int>(layer), layerPair.second.size());
        total += layerPair.second.size();
        // Dispatch per-layer
        processErasedNodeSet(layer, layerPair.second, onErased);
    }
    //fprintf(stdout, "[SceneRenderer::processErasedNodeSet] totals: uniqueErased=%zu\n", total);
}

void SceneRenderer::processErasedNodeSet(Layer layer, const std::unordered_set<NodeID>& nodeSet, const std::function<void(Layer, NodeID)>& onErased) {
    for (const NodeID nid : nodeSet) {
        onErased(layer, nid);
    }
}

// Ensure mesh exists and is up-to-date for a node: insert or replace when needed
void SceneRenderer::updateMeshForNode(Layer layer, NodeID nid, const OctreeNodeData &nd, const Geometry &geom) {
    IndirectRenderer &renderer = layer == LAYER_OPAQUE ? solidRenderer->getIndirectRenderer() : waterRenderer->getIndirectRenderer();

    const auto &cur = layer == LAYER_OPAQUE ? solidChunks : transparentChunks;
    auto it = cur.find(nid);
    if (it != cur.end()) {
        if (it->second.version >= nd.node->version) {
            return; // already up-to-date
        }
        if (it->second.meshId != UINT32_MAX) {
            renderer.removeMesh(it->second.meshId);
        }
    }
    uint32_t meshId = renderer.addMesh(app, geom);
    Model3DVersion mv{meshId, nd.node->version};
    registerModelVersion(nid, mv);
    renderer.uploadMesh(app, meshId);

}


void SceneRenderer::processPendingNodeChanges(Scene& scene) {
    std::vector<PendingNode> created;
    std::vector<PendingNode> updated;
    std::vector<PendingNode> erased;
    {
        std::lock_guard<std::mutex> lock(pendingMutex);
        created.swap(pendingCreated);
        updated.swap(pendingUpdated);
        erased.swap(pendingErased);
    }
    size_t addedCount = 0;
    size_t removedCount = 0;


    // Delegate processing/coalescing: created/updated
    auto onGeometry = [&](Layer layer, NodeID nid, const OctreeNodeData &nd, const Geometry &geom) {
        updateMeshForNode(layer, nid, nd, geom);
        addDebugCubeForGeometry(layer, nid, nd, geom);
        ++addedCount;
    };

    // Process created/updated pending lists (coalescing happens inside overload)
    processNodes(scene, created, onGeometry);
    processNodes(scene, updated, onGeometry);

    // Process erased nodes (coalescing + handling)
    auto onErase = [&](Layer layer, NodeID nid) {
        auto chunks = layer == LAYER_OPAQUE ? &solidChunks : &transparentChunks;
        auto renderer = layer == LAYER_OPAQUE ? &solidRenderer->getIndirectRenderer() : &waterRenderer->getIndirectRenderer();

        auto it = chunks->find(nid);
        if (it != chunks->end() && it->second.meshId != UINT32_MAX) {
            renderer->removeMesh(it->second.meshId);
            chunks->erase(it);
            // Remove debug cube instance for this node
            removeDebugCubeForNode(nid);
            ++removedCount;
        }
    };

    processErasedNodeSet(erased, onErase);

    if (addedCount > 0 || removedCount > 0) {
        printf("[SceneRenderer::processPendingNodeChanges] added=%zu removed=%zu -> rebuilding\n", addedCount, removedCount);
        solidRenderer->getIndirectRenderer().rebuild(app);
        waterRenderer->getIndirectRenderer().rebuild(app);
    }
}

// Runtime introspection helpers for UI/debug
size_t SceneRenderer::getPendingCreatedCount() {
    std::lock_guard<std::mutex> lock(pendingMutex);
    return pendingCreated.size();
}
size_t SceneRenderer::getPendingUpdatedCount() {
    std::lock_guard<std::mutex> lock(pendingMutex);
    return pendingUpdated.size();
}
size_t SceneRenderer::getPendingErasedCount() {
    std::lock_guard<std::mutex> lock(pendingMutex);
    return pendingErased.size();
}
size_t SceneRenderer::getTransparentModelCount() {
    return transparentChunks.size();
}

bool SceneRenderer::hasModelForNode(Layer layer, NodeID nid) const {
    if (layer == LAYER_OPAQUE) {
        return solidChunks.find(nid) != solidChunks.end();
    } else {
        return transparentChunks.find(nid) != transparentChunks.end();
    }
}

void SceneRenderer::addDebugCubeForGeometry(Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
    if (!debugCubeRenderer) return;
    DebugCubeRenderer::CubeWithColor c;
    // Compute world-space AABB from geometry vertices using same model as used for mesh
    glm::vec3 minp(nd.cube.getMax()), maxp(nd.cube.getMin());
    for (const auto &v : geom.vertices) {
        minp = glm::min(minp, v.position);
        maxp = glm::max(maxp, v.position);
    }
    if (minp.x == FLT_MAX) {
        // Fallback to node cube if geometry empty
        minp = nd.cube.getMin();
        maxp = nd.cube.getMax();
    }
    c.cube = BoundingBox(minp, maxp);
    c.color = (layer == LAYER_OPAQUE) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(0.0f, 0.5f, 1.0f);
    addDebugCubeForNode(nid, c);
}
