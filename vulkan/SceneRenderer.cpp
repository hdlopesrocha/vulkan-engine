#include "SceneRenderer.hpp"

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

SceneRenderer::SceneRenderer(VulkanApp* app_)
    : app(app_),
      shadowMapper(std::make_unique<ShadowRenderer>(app_, 8192)),
      waterRenderer(std::make_unique<WaterRenderer>(app_)),
      skyRenderer(std::make_unique<SkyRenderer>(app_)),
      solidRenderer(std::make_unique<SolidRenderer>(app_)),
      vegetationRenderer(std::make_unique<VegetationRenderer>(app_)),
      debugCubeRenderer(std::make_unique<DebugCubeRenderer>(app_)),
      skySettings()
{
    // All renderer members are now properly instantiated and internal SkySettings constructed
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
    
    // Allocate and initialize texture arrays
    if (vegetationRenderer) {
        textureArrayManager.allocate(64, 256, 256);
        vegetationRenderer->setTextureArrayManager(&textureArrayManager);
        vegetationRenderer->init(app);
        textureArrayManager.initialize(app);
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

    addImageWrite(1, textureArrayManager.albedoSampler, textureArrayManager.albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    addImageWrite(2, textureArrayManager.normalSampler, textureArrayManager.normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    addImageWrite(3, textureArrayManager.bumpSampler, textureArrayManager.bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    addImageWrite(4, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

    // Create and bind Materials SSBO at binding 5
    materialsBuffer = app->createBuffer(sizeof(MaterialGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
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

#include "../utils/LocalScene.hpp"

void SceneRenderer::populateFromScene(Scene* scene, Layer layer) {
    if (!scene) return;
    if (!solidRenderer) return;

    printf("[SceneRenderer::populateFromScene] Populating meshes for layer=%d\n", layer);
    LocalScene* ls = dynamic_cast<LocalScene*>(scene);
    if (!ls) {
        fprintf(stderr, "[SceneRenderer::populateFromScene] Scene is not a LocalScene; skipping population\n");
        return;
    }

    ls->forEachChunkNode(layer, [&](std::vector<OctreeNodeData>& nodes) {
        size_t added = 0;
        for (auto &nd : nodes) {
            if (!nd.node) continue;
            NodeID nid = reinterpret_cast<NodeID>(nd.node);
            const auto &current = solidRenderer->getNodeModelVersions();
            auto it = current.find(nid);
            if (it != current.end() && it->second.version >= nd.node->version) {
                continue; // up-to-date
            }
            // Request geometry for this node (synchronous callback)
            scene->requestModel3D(layer, nd, [&](const Geometry &geom) {
                // Simple model: translate to node center (geometry is generated in object space)
                glm::mat4 model = glm::translate(glm::mat4(1.0f), nd.cube.getCenter());
                uint32_t meshId = solidRenderer->getIndirectRenderer().addMesh(app, geom, model);
                Model3DVersion mv; mv.meshId = meshId; mv.version = nd.node->version;
                solidRenderer->registerModelVersion(nid, mv);
                ++added;
            });
        }
        if (added > 0) {
            printf("[SceneRenderer::populateFromScene] Added %zu meshes for layer=%d\n", added, layer);
            solidRenderer->getIndirectRenderer().rebuild(app);
        }
    });
}
