// ...existing code...
#include "SceneRenderer.hpp"

#include "../utils/SolidSpaceChangeHandler.hpp"
#include "../utils/LiquidSpaceChangeHandler.hpp"
#include "../utils/LocalScene.hpp"
#include "../math/ContainmentType.hpp"
#include <mutex>

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
    // Destroy UBO and SSBO buffers owned by this renderer
    if (app) {
        VkDevice dev = app->getDevice();
        if (mainUniformBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(dev, mainUniformBuffer.buffer, nullptr);
            if (mainUniformBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(dev, mainUniformBuffer.memory, nullptr);
            mainUniformBuffer = {};
        }
        // Only destroy materialsBuffer if it is not the same buffer owned by the MaterialManager
        if (materialsBuffer.buffer != VK_NULL_HANDLE) {
            bool ownedByManager = false;
            if (materialManager) {
                const Buffer &mgrBuf = materialManager->getBuffer();
                if (mgrBuf.buffer == materialsBuffer.buffer && mgrBuf.memory == materialsBuffer.memory) ownedByManager = true;
            }
            if (!ownedByManager) {
                vkDestroyBuffer(dev, materialsBuffer.buffer, nullptr);
                if (materialsBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(dev, materialsBuffer.memory, nullptr);
                materialsBuffer = {};
            }
        }
        // Destroy Pass UBO buffers if allocated
        if (mainPassUBO.buffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(dev, mainPassUBO.buffer.buffer, nullptr);
            if (mainPassUBO.buffer.memory != VK_NULL_HANDLE) vkFreeMemory(dev, mainPassUBO.buffer.memory, nullptr);
            mainPassUBO.buffer = {};
        }
        if (shadowPassUBO.buffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(dev, shadowPassUBO.buffer.buffer, nullptr);
            if (shadowPassUBO.buffer.memory != VK_NULL_HANDLE) vkFreeMemory(dev, shadowPassUBO.buffer.memory, nullptr);
            shadowPassUBO.buffer = {};
        }
        if (waterPassUBO.buffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(dev, waterPassUBO.buffer.buffer, nullptr);
            if (waterPassUBO.buffer.memory != VK_NULL_HANDLE) vkFreeMemory(dev, waterPassUBO.buffer.memory, nullptr);
            waterPassUBO.buffer = {};
        }
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
            // If you have a vegetation buffer rebuild method, call it here (e.g., vegetationRenderer->rebuildBuffers(app);)
            // vegetationRenderer->rebuildBuffers(app); // Uncomment and implement if needed
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
    // Indirect rendering for water geometry (same as solid matter)
    // Use waterIndirectRenderer and match solidRenderer->draw signature
    auto &waterIndirectRenderer = waterRenderer->getIndirectRenderer();
    waterIndirectRenderer.drawPrepared(cmd, app);
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
    // Always create a valid materialsBuffer for descriptor binding 5
    if (materialManager) {
        materialsBuffer = materialManager->getBuffer();
        // If the MaterialManager exists but hasn't allocated its GPU buffer yet,
        // create a local fallback so the descriptor update does not receive a
        // VK_NULL_HANDLE buffer (which triggers validation errors).
        if (materialsBuffer.buffer == VK_NULL_HANDLE) {
            materialsBuffer = app->createBuffer(sizeof(MaterialGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
    } else {
        // Fallback: create a dummy buffer and keep it alive for the renderer lifetime
        materialsBuffer = app->createBuffer(sizeof(MaterialGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    VkDescriptorBufferInfo materialsInfo{};
    materialsInfo.buffer = materialsBuffer.buffer;
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
SolidSpaceChangeHandler SceneRenderer::makeSolidSpaceChangeHandler(Scene* scene) {
    solidNodeEventCallback = [this, scene](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        std::cout << "[SceneRenderer] Solid node event: nid=" << nid << " level=" << nd.level << " containment=" << static_cast<int>(nd.containmentType) << "\n";
        // Trigger solid mesh update for this node if needed
        if (nd.containmentType != ContainmentType::Disjoint) {
            OctreeNodeData nodeCopy = nd;
            this->processNodeLayer(*scene, LAYER_OPAQUE, nid, nodeCopy,
                [this](Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
                    this->updateMeshForNode(layer, nid, nd, geom);
                }
            );
        }
    };
    
    solidNodeEraseCallback = [this, scene](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        std::cout << "[SceneRenderer] Solid node erase: nid=" << nid << "\n";
        // Remove solid mesh for this node if it exists
        auto it = solidChunks.find(nid);
        if (it != solidChunks.end()) {
            if (it->second.meshId != UINT32_MAX) {
                solidRenderer->getIndirectRenderer().removeMesh(it->second.meshId);
            }
            solidChunks.erase(it);
            removeDebugCubeForNode(nid);
        }
    };
    
    return SolidSpaceChangeHandler(solidNodeEventCallback, solidNodeEraseCallback);
}

LiquidSpaceChangeHandler SceneRenderer::makeLiquidSpaceChangeHandler(Scene* scene) {

    liquidNodeEventCallback = [this, scene](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        std::cout << "[SceneRenderer] Liquid node event: nid=" << nid << " level=" << nd.level << " containment=" << static_cast<int>(nd.containmentType) << "\n";
        // Trigger water mesh update for this node if needed
        if (nd.containmentType != ContainmentType::Disjoint) {
            OctreeNodeData nodeCopy = nd;
            this->processNodeLayer(*scene, LAYER_TRANSPARENT, nid, nodeCopy,
                [this](Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
                    this->updateMeshForNode(layer, nid, nd, geom);
                }
            );
        }
    };

    liquidNodeEraseCallback = [this, scene](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        std::cout << "[SceneRenderer] Liquid node erase: nid=" << nid << "\n";
        // Remove water mesh for this node if it exists
        auto it = transparentChunks.find(nid);
        if (it != transparentChunks.end()) {
            if (it->second.meshId != UINT32_MAX) {
                waterRenderer->getIndirectRenderer().removeMesh(it->second.meshId);
            }
            transparentChunks.erase(it);
            removeDebugCubeForNode(nid);
        }
    };


    return LiquidSpaceChangeHandler(liquidNodeEventCallback, liquidNodeEraseCallback);
}


// Ensure mesh exists and is up-to-date for a node: insert or replace when needed
void SceneRenderer::updateMeshForNode(Layer layer, NodeID nid, const OctreeNodeData &nd, const Geometry &geom) {
    IndirectRenderer &renderer = layer == LAYER_OPAQUE ? solidRenderer->getIndirectRenderer() : waterRenderer->getIndirectRenderer();

    const auto &cur = layer == LAYER_OPAQUE ? solidChunks : transparentChunks;
    auto it = cur.find(nid);
    if (it != cur.end()) {
        if (it->second.version >= nd.node->version) {
            printf("[SceneRenderer::updateMeshForNode] Node %llu already up-to-date (version %u >= %u)\n", (unsigned long long)nid, it->second.version, nd.node->version);
            return; // already up-to-date
        }
        if (it->second.meshId != UINT32_MAX) {
            printf("[SceneRenderer::updateMeshForNode] Removing old mesh for node %llu (meshId=%u)\n", (unsigned long long)nid, it->second.meshId);
            renderer.removeMesh(it->second.meshId);
        }
    }
    printf("[SceneRenderer::updateMeshForNode] Adding mesh for node %llu (layer=%s): verts=%zu, indices=%zu\n", (unsigned long long)nid, (layer == LAYER_OPAQUE ? "OPAQUE" : "TRANSPARENT"), geom.vertices.size(), geom.indices.size());
    uint32_t meshId = renderer.addMesh(app, geom);
    printf("[SceneRenderer::updateMeshForNode] meshId=%u assigned to node %llu, version=%u\n", meshId, (unsigned long long)nid, nd.node->version);
    Model3DVersion mv{meshId, nd.node->version};
    registerModelVersion(nid, mv);
    renderer.uploadMesh(app, meshId);
    // After all mesh uploads, force a buffer rebuild if dirty
    if (renderer.isDirty()) {
        printf("[SceneRenderer::updateMeshForNode] Forcing buffer rebuild for %s renderer after mesh upload.\n", (layer == LAYER_OPAQUE ? "solid" : "water"));
        renderer.rebuild(app);
    }

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

