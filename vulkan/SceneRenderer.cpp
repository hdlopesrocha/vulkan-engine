// ...existing code...
#include "SceneRenderer.hpp"

#include "../utils/SolidSpaceChangeHandler.hpp"
#include "../utils/LiquidSpaceChangeHandler.hpp"
#include "../utils/LocalScene.hpp"
#include "../math/ContainmentType.hpp"
#include <mutex>

void SceneRenderer::cleanup(VulkanApp* app) {
    // Cleanup all sub-renderers to properly destroy GPU resources (app may be null)
    if (waterRenderer && app) {
        waterRenderer->cleanup(app);
    }
    if (solidRenderer && app) {
        solidRenderer->cleanup(app);
    }
    if (shadowMapper && app) {
        shadowMapper->cleanup(app);
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

    // Clear local CPU-side handles; Vulkan objects are destroyed via VulkanResourceManager
    if (mainUniformBuffer.buffer != VK_NULL_HANDLE) {
        mainUniformBuffer = {};
    }
    // Only clear materialsBuffer if it is not the same buffer owned by the MaterialManager
    if (materialsBuffer.buffer != VK_NULL_HANDLE) {
        bool ownedByManager = false;
        if (materialManager) {
            const Buffer &mgrBuf = materialManager->getBuffer();
            if (mgrBuf.buffer == materialsBuffer.buffer && mgrBuf.memory == materialsBuffer.memory) ownedByManager = true;
        }
        if (!ownedByManager) {
            materialsBuffer = {};
        }
    }
    if (mainPassUBO.buffer.buffer != VK_NULL_HANDLE) {
        mainPassUBO.buffer = {};
    }
    if (shadowPassUBO.buffer.buffer != VK_NULL_HANDLE) {
        shadowPassUBO.buffer = {};
    }
    if (waterPassUBO.buffer.buffer != VK_NULL_HANDLE) {
        waterPassUBO.buffer = {};
    }
}

void SceneRenderer::onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height) {
    // Recreate offscreen targets that depend on swapchain size
    if (solidRenderer) {
        solidRenderer->createRenderTargets(app, width, height);
    }
    if (waterRenderer) {
        waterRenderer->createRenderTargets(app, width, height);
    }
}

SceneRenderer::SceneRenderer(TextureArrayManager* textureArrayManager_, MaterialManager* materialManager_)
    :
      textureArrayManager(textureArrayManager_),
      materialManager(materialManager_),
    shadowMapper(std::make_unique<ShadowRenderer>(8192)),
    waterRenderer(std::make_unique<WaterRenderer>()),
    skyRenderer(std::make_unique<SkyRenderer>()),
    solidRenderer(std::make_unique<SolidRenderer>()),
    vegetationRenderer(std::make_unique<VegetationRenderer>()),
    debugCubeRenderer(std::make_unique<DebugCubeRenderer>()),
    boundingBoxRenderer(std::make_unique<DebugCubeRenderer>()),
      skySettings(std::make_unique<SkySettings>())
{

}

SceneRenderer::~SceneRenderer() {
    // Do not attempt Vulkan cleanup here (app is not available). The owner
    // (MyApp) must call `sceneRenderer->cleanup(app)` before destroying the
    // VulkanApp instance.
}

void SceneRenderer::shadowPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkQueryPool queryPool, VkDescriptorSet shadowPassDescriptorSet, const UniformObject &uboStatic, bool shadowsEnabled, bool shadowTessellationEnabled) {
    if (commandBuffer == VK_NULL_HANDLE) {
        fprintf(stderr, "[SceneRenderer::shadowPass] commandBuffer is VK_NULL_HANDLE, skipping.\n");
        return;
    }
    //fprintf(stderr, "[SceneRenderer::shadowPass] Entered.\n");
    glm::mat4 lightSpaceMatrix = glm::mat4(1.0f); // TODO: Replace with actual light space matrix
    // Record shadow pass on a temporary command buffer to avoid nested render passes
    if (!shadowsEnabled) return;

    VkCommandBuffer cmd = app->beginSingleTimeCommands();
    shadowMapper->beginShadowPass(app, cmd, lightSpaceMatrix);
    // TODO: Render objects to shadow map using shadowMapper->renderObject(...)
    // End and transition
    shadowMapper->endShadowPass(app, cmd);
    app->endSingleTimeCommands(cmd);

}

void SceneRenderer::mainPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &mainPassInfo, uint32_t frameIdx, bool hasWater, bool vegetationEnabled, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, bool wireframeEnabled, bool profilingEnabled, VkQueryPool queryPool, const glm::mat4 &viewProj,
                  const UniformObject &uboStatic, const WaterParams &waterParams, float waterTime, bool normalMappingEnabled, bool tessellationEnabled, bool shadowsEnabled, int debugMode, float triplanarThreshold, float triplanarExponent) {
    if (commandBuffer == VK_NULL_HANDLE) {
        fprintf(stderr, "[SceneRenderer::mainPass] commandBuffer is VK_NULL_HANDLE, skipping.\n");
        return;
    }
    static bool printedOnce = false;
    if (!printedOnce) {
        fprintf(stderr, "[SceneRenderer::mainPass] Entered. solidRenderer=%p skyRenderer=%p\n", (void*)solidRenderer.get(), (void*)skyRenderer.get());
        printedOnce = true;
    }
    solidRenderer->render(commandBuffer, app, perTextureDescriptorSet, wireframeEnabled);
    vegetationRenderer->draw(app, commandBuffer, perTextureDescriptorSet, viewProj);
}

void SceneRenderer::skyPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, const UniformObject &uboStatic, const glm::mat4 &viewProj) {
    if (!skyRenderer) {
        fprintf(stderr, "[SceneRenderer::skyPass] skyRenderer is nullptr, skipping.\n");
        return;
    }
    SkySettings::Mode mode = skySettings->mode;
    skyRenderer->render(app, commandBuffer, perTextureDescriptorSet, mainUniformBuffer, uboStatic, viewProj, mode);
}

void SceneRenderer::waterPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo, uint32_t frameIdx, VkDescriptorSet perTextureDescriptorSet, bool profilingEnabled, VkQueryPool queryPool, const WaterParams &waterParams, float waterTime) {
    static int frameCount = 0;
    if (frameCount++ == 0) {
        printf("[DEBUG] WaterRenderer::waterPass called for the first time\n");
    }
    
    if (commandBuffer == VK_NULL_HANDLE) {
        fprintf(stderr, "[SceneRenderer::waterPass] commandBuffer is VK_NULL_HANDLE, skipping.\n");
        return;
    }

    // Delegate water offscreen work to WaterRenderer::draw
    VkImageView sceneColorView = solidRenderer->getColorView(frameIdx);
    VkImageView sceneDepthView = solidRenderer->getDepthView(frameIdx);
    waterRenderer->render(app, frameIdx, sceneColorView, sceneDepthView);

    // Post-processing should run inside the active main render pass; caller (e.g. MyApp::draw) should invoke
    // `waterRenderer->renderWaterPostProcess` with valid scene color/depth views when available. Keep this function focused
    // on executing offscreen geometry and returning control to the main pass.
}

void SceneRenderer::init(VulkanApp* app) {
    if (!app) {
        fprintf(stderr, "[SceneRenderer::init] app is nullptr!\n");
        return;
    }
    // skySettingsRef was initialized at construction and must be valid
    
    // Bind external texture arrays if provided; allocation/initialization should be done by the application
    if (vegetationRenderer) {
        if (textureArrayManager) {
            vegetationRenderer->setTextureArrayManager(textureArrayManager, app);
            vegetationRenderer->init();
        } else {
            fprintf(stderr, "[SceneRenderer::init] No TextureArrayManager provided — vegetation renderer initialization deferred\n");
        }
    }
    
    solidRenderer->init();
    // Create render targets first so the solid renderer's renderPass is available
    solidRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    solidRenderer->createPipelines(app);

    // Create pipelines for all renderers (solid renderer now has its render pass ready)
    skyRenderer->init(app, solidRenderer->getRenderPass());
    shadowMapper->init(app);
    vegetationRenderer->init(app, solidRenderer->getRenderPass());

    // Initialize debug cube renderer
    if (debugCubeRenderer) {
        debugCubeRenderer->init(app, solidRenderer->getRenderPass());
    }
    // Initialize bounding box renderer (reuses cube wireframe pipeline)
    if (boundingBoxRenderer) {
        boundingBoxRenderer->init(app, solidRenderer->getRenderPass());
    }
    
    // Create main uniform buffer
    mainUniformBuffer = app->createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    VkDescriptorSet mainDs = app->getMainDescriptorSet();
    printf("[SceneRenderer::init] mainDescriptorSet = 0x%llx\n", (unsigned long long)mainDs);

    // Initialize sky renderer with our owned settings now that descriptor sets are ready
    if (skyRenderer) {
        skyRenderer->init(app, *skySettings, mainDs);
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
    app->updateDescriptorSet(writes);
    for (auto &w : writes) {
        if (w.pImageInfo) delete w.pImageInfo;
    }

    // Initialize WaterRenderer
    Buffer waterParamsBuffer = app->createBuffer(sizeof(WaterUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    waterRenderer->init(app, waterParamsBuffer);
    waterRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    
    // Initialize sky renderer with sphere VBO now that descriptor sets are ready
    skyRenderer->init(app, *skySettings, mainDs);
    

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
SolidSpaceChangeHandler SceneRenderer::makeSolidSpaceChangeHandler(Scene* scene, VulkanApp* app) {
    solidNodeEventCallback = [this, scene, app](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        std::cout << "[SceneRenderer] Solid node event: nid=" << nid << " level=" << nd.level << " containment=" << static_cast<int>(nd.containmentType) << "\n";
        // Trigger solid mesh update for this node if needed
        if (nd.containmentType != ContainmentType::Disjoint) {
            OctreeNodeData nodeCopy = nd;
            this->processNodeLayer(*scene, LAYER_OPAQUE, nid, nodeCopy,
                [this, app](Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
                    this->updateMeshForNode(app, layer, nid, nd, geom);
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

LiquidSpaceChangeHandler SceneRenderer::makeLiquidSpaceChangeHandler(Scene* scene, VulkanApp* app) {

    liquidNodeEventCallback = [this, scene, app](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        std::cout << "[SceneRenderer] Liquid node event: nid=" << nid << " level=" << nd.level << " containment=" << static_cast<int>(nd.containmentType) << "\n";
        // Trigger water mesh update for this node if needed
        if (nd.containmentType != ContainmentType::Disjoint) {
            OctreeNodeData nodeCopy = nd;
            this->processNodeLayer(*scene, LAYER_TRANSPARENT, nid, nodeCopy,
                [this, app](Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
                    this->updateMeshForNode(app, layer, nid, nd, geom);
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
void SceneRenderer::updateMeshForNode(VulkanApp* app, Layer layer, NodeID nid, const OctreeNodeData &nd, const Geometry &geom) {
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
    uint32_t meshId = renderer.addMesh(geom);
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

