// ...existing code...
#include "SceneRenderer.hpp"


#include <stdexcept>
#include "../../utils/SolidSpaceChangeHandler.hpp"
#include "../../utils/LiquidSpaceChangeHandler.hpp"
#include "../../utils/LocalScene.hpp"
#include "../../math/ContainmentType.hpp"
#include <algorithm>
#include <mutex>
#include <unordered_set>

void SceneRenderer::cleanup(VulkanApp* app) {
    // Cleanup all sub-renderers to properly destroy GPU resources (app may be null)
    if (postProcessRenderer && app) {
        postProcessRenderer->cleanup(app);
    }
    if (waterRenderer && app) {
        waterRenderer->cleanup(app);
    }
    // Cleanup scene-owned water sub-renderers
    if (backFaceRenderer && app) {
        backFaceRenderer->cleanup(app);
    }
    if (solid360Renderer && app) {
        solid360Renderer->cleanup(app);
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
    if (solidWireframe) {
        solidWireframe->cleanup();
    }
    if (waterWireframe) {
        waterWireframe->cleanup();
    }

    // Clear local CPU-side handles; Vulkan objects are destroyed via VulkanResourceManager
    if (mainUniformBuffer.buffer != VK_NULL_HANDLE) {
        mainUniformBuffer = {};
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
        // Recreate back-face and 360 reflection targets owned by SceneRenderer
        if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, width, height);
        if (solid360Renderer) {
            solid360Renderer->destroySolid360Targets(app);
            solid360Renderer->createSolid360Targets(app, solidRenderer->getRenderPass(), waterRenderer->getLinearSampler());
        }
    }
    if (postProcessRenderer) {
        postProcessRenderer->setRenderSize(width, height);
    }
    if (skyRenderer) {
        skyRenderer->createOffscreenTargets(app, width, height);
    }
}

SceneRenderer::SceneRenderer() :
    shadowMapper(std::make_unique<ShadowRenderer>(8192)),
    waterRenderer(std::make_unique<WaterRenderer>()),
    postProcessRenderer(std::make_unique<PostProcessRenderer>()),
    skyRenderer(std::make_unique<SkyRenderer>()),
    solidRenderer(std::make_unique<SolidRenderer>()),
    vegetationRenderer(std::make_unique<VegetationRenderer>()),
    debugCubeRenderer(std::make_unique<DebugCubeRenderer>()),
    boundingBoxRenderer(std::make_unique<DebugCubeRenderer>()),
    solidWireframe(std::make_unique<WireframeRenderer>()),
    waterWireframe(std::make_unique<WireframeRenderer>()),
      skySettings(std::make_unique<SkySettings>())
{

}

SceneRenderer::~SceneRenderer() {
    // Do not attempt Vulkan cleanup here (app is not available). The owner
    // (MyApp) must call `sceneRenderer->cleanup(app)` before destroying the
    // VulkanApp instance.
}

void SceneRenderer::shadowPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkDescriptorSet mainDescriptorSet, Buffer &mainUniformBuffer, const UniformObject &uboStatic, bool shadowsEnabled) {
    static bool firstCall = true;
    if (firstCall) {
        firstCall = false;
        std::cerr << "[SceneRenderer::shadowPass] FIRST CALL! shadowsEnabled=" << (int)shadowsEnabled
                  << " commandBuffer=" << (void*)commandBuffer
                  << " cascades=" << SHADOW_CASCADE_COUNT << std::endl;
    }
    if (commandBuffer == VK_NULL_HANDLE) return;
    if (!shadowsEnabled) return;

    // Render each cascade: upload light-space UBO, draw scene, restore UBO
    const glm::mat4 cascadeMatrices[SHADOW_CASCADE_COUNT] = {
        uboStatic.lightSpaceMatrix,
        uboStatic.lightSpaceMatrix1,
        uboStatic.lightSpaceMatrix2
    };

    for (int c = 0; c < SHADOW_CASCADE_COUNT; c++) {
        glm::mat4 lsMatrix = cascadeMatrices[c];

        // Upload a shadow-specific UBO: viewProjection = cascade lightSpaceMatrix, passParams.x = 1.0
        UniformObject shadowUBO = uboStatic;
        shadowUBO.viewProjection = lsMatrix;
        shadowUBO.passParams.x = 1.0f;

        vkCmdUpdateBuffer(commandBuffer, mainUniformBuffer.buffer, 0, sizeof(UniformObject), &shadowUBO);
        {
            VkMemoryBarrier memBarrier{};
            memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 1, &memBarrier, 0, nullptr, 0, nullptr);
        }

        shadowMapper->beginShadowPass(app, commandBuffer, c, lsMatrix);

        // Bind shadow descriptor set (uses dummy depth at bindings 4,8,9)
        VkPipelineLayout layout = shadowMapper->getShadowPipelineLayout();
        VkDescriptorSet ds = shadowDescriptorSet;
        if (layout != VK_NULL_HANDLE && ds != VK_NULL_HANDLE) {
            //printf("[BIND] SceneRenderer::mainPass (shadow bind): layout=%p firstSet=0 count=1 sets=%p\n", (void*)layout, (void*)ds);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &ds, 0, nullptr);
        }

        solidRenderer->getIndirectRenderer().drawAll(commandBuffer);

        shadowMapper->endShadowPass(app, commandBuffer, c);
    }

    // Restore the main UBO so subsequent passes see the original data
    vkCmdUpdateBuffer(commandBuffer, mainUniformBuffer.buffer, 0, sizeof(UniformObject), &uboStatic);
    {
        VkMemoryBarrier memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }
}

void SceneRenderer::mainPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &mainPassInfo, uint32_t frameIdx, bool hasWater, bool vegetationEnabled, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, bool wireframeEnabled, bool profilingEnabled, VkQueryPool queryPool, const glm::mat4 &viewProj,
                  const UniformObject &uboStatic, bool normalMappingEnabled, bool tessellationEnabled, bool shadowsEnabled, int debugMode, float triplanarThreshold, float triplanarExponent) {
    if (commandBuffer == VK_NULL_HANDLE) {
        std::cerr << "[SceneRenderer::mainPass] commandBuffer is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }
    static bool printedOnce = false;
    if (!printedOnce) {
        std::cerr << "[SceneRenderer::mainPass] Entered. solidRenderer=" << (void*)solidRenderer.get()
                  << " skyRenderer=" << (void*)skyRenderer.get() << std::endl;
        printedOnce = true;
    }
    if (wireframeEnabled && solidWireframe) {
        solidWireframe->draw(commandBuffer, app, {perTextureDescriptorSet}, solidRenderer->getIndirectRenderer());
    }else {
        solidRenderer->render(commandBuffer, app, perTextureDescriptorSet);
    }
    vegetationRenderer->draw(app, commandBuffer, perTextureDescriptorSet, viewProj);
}

void SceneRenderer::skyPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, const UniformObject &uboStatic, const glm::mat4 &viewProj) {
    if (!skyRenderer) {
        std::cerr << "[SceneRenderer::skyPass] skyRenderer is nullptr, skipping." << std::endl;
        return;
    }
    SkySettings::Mode mode = skySettings->mode;
    skyRenderer->render(app, commandBuffer, perTextureDescriptorSet, mainUniformBuffer, uboStatic, viewProj, mode);
}

void SceneRenderer::waterPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo, uint32_t frameIdx, VkDescriptorSet perTextureDescriptorSet, bool wireframeEnabled, bool profilingEnabled, VkQueryPool queryPool, float waterTime, bool skipBackFace, VkImageView skyView, VkImageView cubeReflectionView) {
    if (commandBuffer == VK_NULL_HANDLE) {
        std::cerr << "[SceneRenderer::waterPass] commandBuffer is VK_NULL_HANDLE, skipping." << std::endl;
        return;
    }

    // Update the water render UBO with the active layer time value.
    if (waterRenderUBOBuffer_.buffer != VK_NULL_HANDLE) {
        WaterRenderUBO renderUbo{};
        renderUbo.timeParams = glm::vec4(waterTime, 0.0f, 0.0f, 0.0f);
        void* data = nullptr;
        vkMapMemory(app->getDevice(), waterRenderUBOBuffer_.memory, 0, sizeof(WaterRenderUBO), 0, &data);
        memcpy(data, &renderUbo, sizeof(WaterRenderUBO));
        vkUnmapMemory(app->getDevice(), waterRenderUBOBuffer_.memory);
    }

    // Delegate water offscreen work to WaterRenderer — record on the same
    // command buffer so the solid pass outputs are available for sampling.
    VkImageView sceneColorView = solidRenderer->getColorView(frameIdx);
    VkImageView sceneDepthView = solidRenderer->getDepthView(frameIdx);

    // Update WaterRenderer's scene texture binding so it sees current back-face
    // depth view and any available cubemap reflection view.
    VkImageView backFaceDepthView = (backFaceRenderer) ? backFaceRenderer->getBackFaceDepthView(frameIdx) : VK_NULL_HANDLE;
    VkImageView cubeView = cubeReflectionView;
    if (waterRenderer) waterRenderer->updateSceneTexturesBinding(app, sceneColorView, sceneDepthView, frameIdx, skyView, backFaceDepthView, cubeView);

    if (wireframeEnabled && waterWireframe && waterWireframe->getPipeline() != VK_NULL_HANDLE) {
        // Wireframe path: use WaterRenderer for setup/pass management,
        // but bind the wireframe pipeline instead of the normal one.
        waterRenderer->prepareRender(app, commandBuffer, frameIdx, sceneColorView, sceneDepthView, skyView);
        if (backFaceRenderer && !skipBackFace) {
            backFaceRenderer->renderBackFacePass(app, commandBuffer, frameIdx,
                                                waterRenderer->getIndirectRenderer(),
                                                waterRenderer->getWaterGeometryPipelineLayout(),
                                                app->getMainDescriptorSet(),
                                                app->getMaterialDescriptorSet(),
                                                waterRenderer->getWaterDepthDescriptorSet(frameIdx),
                                                solidRenderer->getDepthImage(frameIdx));
        }
        waterRenderer->beginWaterGeometryPass(commandBuffer, frameIdx);

        waterWireframe->draw(commandBuffer, app,
            {app->getMainDescriptorSet(),
             app->getMaterialDescriptorSet(),
             waterRenderer->getWaterDepthDescriptorSet(frameIdx)},
            waterRenderer->getIndirectRenderer());

        waterRenderer->endWaterGeometryPass(commandBuffer);
        waterRenderer->postRenderBarrier(commandBuffer, frameIdx);
    } else {
        if (backFaceRenderer && !skipBackFace) {
            backFaceRenderer->renderBackFacePass(app, commandBuffer, frameIdx,
                                                waterRenderer->getIndirectRenderer(),
                                                waterRenderer->getWaterGeometryPipelineLayout(),
                                                app->getMainDescriptorSet(),
                                                app->getMaterialDescriptorSet(),
                                                waterRenderer->getWaterDepthDescriptorSet(frameIdx),
                                                solidRenderer->getDepthImage(frameIdx));
        }
        waterRenderer->render(app, commandBuffer, frameIdx, sceneColorView, sceneDepthView, skyView);
    }

    // Post-processing should run inside the active main render pass; caller (e.g. MyApp::draw) should invoke
    // `postProcessRenderer->render` with valid scene/water views when available. Keep this function focused
    // on executing offscreen geometry and returning control to the main pass.
}

void SceneRenderer::init(VulkanApp* app, TextureArrayManager* textureArrayManager, MaterialManager* materialManager, const std::vector<WaterParams>& waterParams) {
    if (!app) {
        std::cerr << "[SceneRenderer::init] app is nullptr!" << std::endl;
        return;
    }
    // skySettingsRef was initialized at construction and must be valid
    

    // Bind external texture arrays if provided; allocation/initialization should be done by the application
    if (vegetationRenderer) {
        if (textureArrayManager) {
            vegetationRenderer->setTextureArrayManager(textureArrayManager, app);
            vegetationRenderer->init();
        } else {
            std::cerr << "[SceneRenderer::init] No TextureArrayManager provided — vegetation renderer initialization deferred" << std::endl;
        }
    }
    
    solidRenderer->init();
    // Create render targets first so the solid renderer's renderPass is available
    solidRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    solidRenderer->createPipelines(app);

    // Create pipelines for all renderers (solid renderer now has its render pass ready)
    skyRenderer->init(app, solidRenderer->getRenderPass());
    // Create offscreen sky targets so the sky can be sampled as a texture by water
    skyRenderer->createOffscreenTargets(app, app->getWidth(), app->getHeight());
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
    
    // Create main uniform buffer (TRANSFER_DST for vkCmdUpdateBuffer in cubemap 360 render)
    mainUniformBuffer = app->createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
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
            std::cerr << "[SceneRenderer::init] Skipping descriptor binding " << binding
                      << ": imageView=" << (void*)view
                      << " sampler=" << (void*)sampler << std::endl;
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
        std::cerr << "[SceneRenderer::init] No TextureArrayManager set — skipping texture array descriptor writes" << std::endl;
    }
    addImageWrite(4, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(0), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    addImageWrite(8, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(1), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    addImageWrite(9, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(2), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

    // Create and bind Materials SSBO at binding 5. Require an external MaterialManager.
    materialManagerPtr = materialManager;
    if (!materialManager) {
        throw std::runtime_error("SceneRenderer::init requires a valid MaterialManager");
    }
    materialsBuffer = materialManager->getBuffer();
    if (materialsBuffer.buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("MaterialManager provided but materials buffer is not allocated");
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

    // Initialize WaterRenderer early and allocate a params SSBO sized to texture layers.
    // Use the passed vector of WaterParams as the source of truth for layer count.
    // Do not fall back to texture-array sizes; require explicit water parameters.
    uint32_t layerCount = waterParams.size();
    if (layerCount == 0) {
        throw std::runtime_error("SceneRenderer::init requires at least one WaterParams entry (no fallback allowed)");
    }
    
    size_t paramsBufferSize = sizeof(WaterParamsGPU) * static_cast<size_t>(layerCount);
    waterParamsBuffer_ = app->createBuffer(paramsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // Create scene-owned water sub-renderers. Back-face renderpass must exist
    // before water pipelines are created, so create it first.
    backFaceRenderer = std::make_unique<WaterBackFaceRenderer>();
    solid360Renderer = std::make_unique<Solid360Renderer>();
    if (backFaceRenderer) backFaceRenderer->createRenderPass(app);

    // Initialize WaterRenderer (creates its pipeline layout and initializes the param SSBO)
    waterRenderer->init(app, waterParamsBuffer_, waterParams, layerCount);

    // Now that WaterRenderer has created its pipeline layout, allow the
    // back-face renderer to create pipelines that depend on it.
    if (backFaceRenderer) backFaceRenderer->createPipelines(app, waterRenderer->getWaterGeometryPipelineLayout());
    // Create back-face render targets early so their image views are
    // available before the first frame's water pass attempts to bind them.
    if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    if (solid360Renderer) solid360Renderer->init(app);

    // Bind water params SSBO to binding 7 of main descriptor set
    VkDescriptorBufferInfo waterParamsInfo{};
    waterParamsInfo.buffer = waterParamsBuffer_.buffer;
    waterParamsInfo.offset = 0;
    waterParamsInfo.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet waterParamsWrite{};
    waterParamsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    waterParamsWrite.dstSet = mainDs;
    waterParamsWrite.dstBinding = 7;
    waterParamsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    waterParamsWrite.descriptorCount = 1;
    waterParamsWrite.pBufferInfo = &waterParamsInfo;
    writes.push_back(waterParamsWrite);

    // Bind water render UBO to binding 10 of main descriptor set
    waterRenderUBOBuffer_ = app->createBuffer(sizeof(WaterRenderUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDescriptorBufferInfo waterRenderUBOInfo{};
    waterRenderUBOInfo.buffer = waterRenderUBOBuffer_.buffer;
    waterRenderUBOInfo.offset = 0;
    waterRenderUBOInfo.range = sizeof(WaterRenderUBO);
    VkWriteDescriptorSet waterRenderUBOWrite{};
    waterRenderUBOWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    waterRenderUBOWrite.dstSet = mainDs;
    waterRenderUBOWrite.dstBinding = 10;
    waterRenderUBOWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    waterRenderUBOWrite.descriptorCount = 1;
    waterRenderUBOWrite.pBufferInfo = &waterRenderUBOInfo;
    writes.push_back(waterRenderUBOWrite);

    // Perform descriptor update (clean up temporary image infos afterwards)
    app->updateDescriptorSet(writes);
    for (auto &w : writes) {
        if (w.pImageInfo) delete w.pImageInfo;
    }

    // ── Allocate a shadow-specific descriptor set.
    //    It mirrors the main descriptor set but binding 4 (shadow map sampler)
    //    points to a tiny dummy depth image that stays in READ_ONLY layout.
    //    This avoids layout-mismatch validation errors because the real shadow
    //    map is in DEPTH_STENCIL_ATTACHMENT_OPTIMAL while the shadow pass renders.
    {
        shadowDescriptorSet = app->createDescriptorSet(app->getDescriptorSetLayout());

        // UBO (binding 0) — same buffer
        VkDescriptorBufferInfo shadowBufInfo{ mainUniformBuffer.buffer, 0, sizeof(UniformObject) };
        VkWriteDescriptorSet shadowUboWrite{};
        shadowUboWrite.sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        shadowUboWrite.dstSet         = shadowDescriptorSet;
        shadowUboWrite.dstBinding     = 0;
        shadowUboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        shadowUboWrite.descriptorCount = 1;
        shadowUboWrite.pBufferInfo    = &shadowBufInfo;

        std::vector<VkWriteDescriptorSet> shadowWrites;
        shadowWrites.push_back(shadowUboWrite);

        auto addShadowImageWrite = [&](uint32_t binding, VkSampler sampler, VkImageView view, VkImageLayout layout) {
            if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
            VkDescriptorImageInfo* info = new VkDescriptorImageInfo();
            info->sampler     = sampler;
            info->imageView   = view;
            info->imageLayout = layout;
            VkWriteDescriptorSet w{};
            w.sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet         = shadowDescriptorSet;
            w.dstBinding     = binding;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.descriptorCount = 1;
            w.pImageInfo     = info;
            shadowWrites.push_back(w);
        };

        // Texture arrays (bindings 1, 2, 3) — same as main
        if (textureArrayManager) {
            addShadowImageWrite(1, textureArrayManager->albedoSampler, textureArrayManager->albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addShadowImageWrite(2, textureArrayManager->normalSampler, textureArrayManager->normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addShadowImageWrite(3, textureArrayManager->bumpSampler, textureArrayManager->bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        // Binding 4, 8, 9: dummy depth image instead of real shadow maps (all cascades)
        addShadowImageWrite(4, shadowMapper->getShadowMapSampler(), shadowMapper->getDummyDepthView(),
                            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        addShadowImageWrite(8, shadowMapper->getShadowMapSampler(), shadowMapper->getDummyDepthView(),
                            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        addShadowImageWrite(9, shadowMapper->getShadowMapSampler(), shadowMapper->getDummyDepthView(),
                            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

        // Materials SSBO (binding 5)
        VkDescriptorBufferInfo shadowMatInfo{ materialsBuffer.buffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet shadowMatWrite{};
        shadowMatWrite.sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        shadowMatWrite.dstSet         = shadowDescriptorSet;
        shadowMatWrite.dstBinding     = 5;
        shadowMatWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        shadowMatWrite.descriptorCount = 1;
        shadowMatWrite.pBufferInfo    = &shadowMatInfo;
        shadowWrites.push_back(shadowMatWrite);

        // Water params SSBO (binding 7)
        VkDescriptorBufferInfo shadowWaterInfo{ waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet shadowWaterWrite{};
        shadowWaterWrite.sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        shadowWaterWrite.dstSet         = shadowDescriptorSet;
        shadowWaterWrite.dstBinding     = 7;
        shadowWaterWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        shadowWaterWrite.descriptorCount = 1;
        shadowWaterWrite.pBufferInfo    = &shadowWaterInfo;
        shadowWrites.push_back(shadowWaterWrite);

        // Water render UBO (binding 10)
        VkDescriptorBufferInfo shadowWaterRenderInfo{ waterRenderUBOBuffer_.buffer, 0, sizeof(WaterRenderUBO) };
        VkWriteDescriptorSet shadowWaterRenderWrite{};
        shadowWaterRenderWrite.sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        shadowWaterRenderWrite.dstSet         = shadowDescriptorSet;
        shadowWaterRenderWrite.dstBinding     = 10;
        shadowWaterRenderWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        shadowWaterRenderWrite.descriptorCount = 1;
        shadowWaterRenderWrite.pBufferInfo    = &shadowWaterRenderInfo;
        shadowWrites.push_back(shadowWaterRenderWrite);

        app->updateDescriptorSet(shadowWrites);
        for (auto &w : shadowWrites) {
            if (w.pImageInfo) delete w.pImageInfo;
        }
        std::cerr << "[SceneRenderer::init] shadowDescriptorSet = " << (void*)shadowDescriptorSet << std::endl;
    }

    // Register listener so we update the main descriptor set when texture arrays are allocated later
    if (textureArrayManager) {
        textureArrayManager->addAllocationListener([this, app, textureArrayManager]() {
            this->updateTextureDescriptorSet(app, textureArrayManager);
        });
    }
    waterRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());

    // Ensure back-face render targets are created as well so the
    // `backFaceDepthView` is valid before the first frame's water pass.
    if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());

    // Create cubemap → equirect 360° reflection targets for water (owned by SceneRenderer)
    if (solid360Renderer) solid360Renderer->createSolid360Targets(app, solidRenderer->getRenderPass(), waterRenderer->getLinearSampler());

    // Create wireframe pipelines for solid and water passes
    if (solidWireframe) {
        std::vector<VkDescriptorSetLayout> solidSetLayouts = { app->getDescriptorSetLayout() };
        solidWireframe->createPipeline(app, solidRenderer->getRenderPass(), 1,
            solidSetLayouts,
            "shaders/main.vert.spv", "shaders/main.frag.spv",
            "shaders/main.tesc.spv", "shaders/main.tese.spv",
            "solid wireframe");
    }
    if (waterWireframe) {
        std::vector<VkDescriptorSetLayout> waterSetLayouts = {
            app->getDescriptorSetLayout(),
            app->getMaterialDescriptorSetLayout(),
            waterRenderer->getWaterDepthDescriptorSetLayout()
        };
        waterWireframe->createPipeline(app, waterRenderer->getWaterRenderPass(), 1,
            waterSetLayouts,
            "shaders/water.vert.spv", "shaders/water.frag.spv",
            "shaders/water.tesc.spv", "shaders/water.tese.spv",
            "water wireframe");
    }

    // Initialize post-process renderer (composites scene + water into swapchain)
    postProcessRenderer->init(app);
    postProcessRenderer->setRenderSize(app->getWidth(), app->getHeight());
    
    // Initialize sky renderer with sphere VBO now that descriptor sets are ready
    skyRenderer->init(app, *skySettings, mainDs);
    

    printf("[SceneRenderer::init] Initialization complete\n");
}

// Update only the texture / materials bindings in the app's main descriptor set.
void SceneRenderer::updateTextureDescriptorSet(VulkanApp* app, TextureArrayManager * textureArrayManager) {
    if (!app) return;
    VkDescriptorSet mainDs = app->getMainDescriptorSet();
    if (mainDs == VK_NULL_HANDLE) return;

    std::vector<VkWriteDescriptorSet> writes;

    // Helper to add image write if valid (allocates VkDescriptorImageInfo)
    auto addImageWrite = [&](uint32_t binding, VkSampler sampler, VkImageView view, VkImageLayout layout) {
        if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) {
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

    addImageWrite(1, textureArrayManager->albedoSampler, textureArrayManager->albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    addImageWrite(2, textureArrayManager->normalSampler, textureArrayManager->normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    addImageWrite(3, textureArrayManager->bumpSampler, textureArrayManager->bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


    // Shadow map samplers (bindings 4, 8, 9) for all cascades
    if (shadowMapper) {
        addImageWrite(4, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(0), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        addImageWrite(8, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(1), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        addImageWrite(9, shadowMapper->getShadowMapSampler(), shadowMapper->getShadowMapView(2), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    }

    // Materials SSBO (binding 5) — refresh from MaterialManager in case the buffer
    // was allocated after SceneRenderer::init (setupTextures may run on a separate thread)
    if (materialManagerPtr && materialManagerPtr->getBuffer().buffer != VK_NULL_HANDLE) {
        materialsBuffer = materialManagerPtr->getBuffer();
    }
    if (materialsBuffer.buffer != VK_NULL_HANDLE) {
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
    } else {
        std::cerr << "[SceneRenderer::updateTextureDescriptorSet] materials buffer not available — skipping binding 5\n";
    }

    // Rebind water params UBO at binding 7 (must stay valid after texture re-allocation)
    if (waterParamsBuffer_.buffer != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo waterParamsInfo{};
        waterParamsInfo.buffer = waterParamsBuffer_.buffer;
        waterParamsInfo.offset = 0;
        waterParamsInfo.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet waterParamsWrite{};
        waterParamsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        waterParamsWrite.dstSet = mainDs;
        waterParamsWrite.dstBinding = 7;
        waterParamsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        waterParamsWrite.descriptorCount = 1;
        waterParamsWrite.pBufferInfo = &waterParamsInfo;
        writes.push_back(waterParamsWrite);
    }

    // Apply descriptor updates
    app->updateDescriptorSet(writes);

    // cleanup allocated image infos
    for (auto &w : writes) {
        if (w.pImageInfo) delete w.pImageInfo;
    }

    // ── Also update shadow descriptor set (bindings 1-3, 5, 7) with new textures/materials.
    //    Bindings 4, 8, 9 stay as the dummy depth image (never changes).
    if (shadowDescriptorSet != VK_NULL_HANDLE) {
        std::vector<VkWriteDescriptorSet> shadowWrites;
        auto addShadowImageWrite = [&](uint32_t binding, VkSampler sampler, VkImageView view, VkImageLayout layout) {
            if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
            VkDescriptorImageInfo* info = new VkDescriptorImageInfo();
            info->sampler = sampler; info->imageView = view; info->imageLayout = layout;
            VkWriteDescriptorSet w{}; w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = shadowDescriptorSet; w.dstBinding = binding;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.descriptorCount = 1;
            w.pImageInfo = info;
            shadowWrites.push_back(w);
        };
        addShadowImageWrite(1, textureArrayManager->albedoSampler, textureArrayManager->albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addShadowImageWrite(2, textureArrayManager->normalSampler, textureArrayManager->normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        addShadowImageWrite(3, textureArrayManager->bumpSampler, textureArrayManager->bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (materialsBuffer.buffer != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo sMatInfo{ materialsBuffer.buffer, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet sMatWrite{}; sMatWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            sMatWrite.dstSet = shadowDescriptorSet; sMatWrite.dstBinding = 5;
            sMatWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; sMatWrite.descriptorCount = 1;
            sMatWrite.pBufferInfo = &sMatInfo;
            shadowWrites.push_back(sMatWrite);
        } else {
            std::cerr << "[SceneRenderer::updateTextureDescriptorSet] shadow materials buffer not available — skipping shadow binding 5\n";
        }
        if (waterParamsBuffer_.buffer != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo sWaterInfo{ waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet sWaterWrite{}; sWaterWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            sWaterWrite.dstSet = shadowDescriptorSet; sWaterWrite.dstBinding = 7;
            sWaterWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; sWaterWrite.descriptorCount = 1;
            sWaterWrite.pBufferInfo = &sWaterInfo;
            shadowWrites.push_back(sWaterWrite);
        }
        app->updateDescriptorSet(shadowWrites);
        for (auto &w : shadowWrites) { if (w.pImageInfo) delete w.pImageInfo; }
    }

    std::cerr << "[SceneRenderer] updateTextureDescriptorSet: updated main descriptor set with texture bindings" << std::endl;
}



// Drain whatever CPU-generated mesh data the background loading thread has
// queued since the last frame, and perform the actual Vulkan GPU uploads.
// Must be called from the main (render) thread each frame.
void SceneRenderer::processPendingMeshes(VulkanApp* app) {
    std::deque<PendingMeshData> batch;
    {
        std::lock_guard<std::mutex> lock(pendingMeshMutex);
        batch.swap(pendingMeshQueue);
    }
    if (batch.empty()) return;
    for (auto& pd : batch) {
        updateMeshForNode(app, pd.layer, pd.nid, pd.nodeData, pd.geom);
    }
    // Batch rebuild: one GPU upload per frame instead of one per mesh.
    // updateMeshForNode appends geometry to CPU arrays and marks dirty=true;
    // rebuild() here consolidates all changes into the GPU buffers at once.
    IndirectRenderer& solidIR = solidRenderer->getIndirectRenderer();
    if (solidIR.isDirty()) solidIR.rebuild(app);
    IndirectRenderer& waterIR = waterRenderer->getIndirectRenderer();
    if (waterIR.isDirty()) waterIR.rebuild(app);
}

void SceneRenderer::processNodeLayer(Scene& scene, Layer layer, NodeID nid, OctreeNodeData& nodeData, const std::function<void(Layer, NodeID, const OctreeNodeData&, const Geometry&)>& onGeometry) {

    // Make a local copy of the node data so the callback may safely outlive this stack frame
    OctreeNodeData nodeCopy = nodeData;
    // Capture nodeCopy by value so async callbacks receive a safe copy
    scene.requestModel3D(layer, nodeCopy, [this, layer, nid, nodeCopy, &onGeometry](const Geometry &geom) {
        onGeometry(layer, nid, nodeCopy, geom);
    });
    
}

// Return Solid/Liquid change handlers that reference the callbacks stored on this object
SolidSpaceChangeHandler SceneRenderer::makeSolidSpaceChangeHandler(Scene* scene, VulkanApp* app) {
    solidNodeEventCallback = [this, scene](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        // Trigger solid mesh update for this node if needed.
        // CPU tessellation runs here (background thread); result is queued for
        // main-thread GPU upload via processPendingMeshes().
        if (nd.containmentType != ContainmentType::Disjoint) {
            OctreeNodeData nodeCopy = nd;
            this->processNodeLayer(*scene, LAYER_OPAQUE, nid, nodeCopy,
                [this](Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
                    std::lock_guard<std::mutex> lock(pendingMeshMutex);
                    pendingMeshQueue.push_back({layer, nid, nd, geom});
                }
            );
        }
    };
    
    solidNodeEraseCallback = [this, scene](const OctreeNodeData& nd) {
        std::lock_guard<std::recursive_mutex> lock(chunksMutex);
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        //std::cout << "[SceneRenderer] Solid node erase: nid=" << nid << "\n";
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

    liquidNodeEventCallback = [this, scene](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        // CPU tessellation runs here (background thread); result is queued for
        // main-thread GPU upload via processPendingMeshes().
        if (nd.containmentType != ContainmentType::Disjoint) {
            OctreeNodeData nodeCopy = nd;
            this->processNodeLayer(*scene, LAYER_TRANSPARENT, nid, nodeCopy,
                [this](Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
                    std::lock_guard<std::mutex> lock(pendingMeshMutex);
                    pendingMeshQueue.push_back({layer, nid, nd, geom});
                }
            );
        }
    };

    liquidNodeEraseCallback = [this, scene](const OctreeNodeData& nd) {
        std::lock_guard<std::recursive_mutex> lock(chunksMutex);
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        //std::cout << "[SceneRenderer] Liquid node erase: nid=" << nid << "\n";
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
    std::lock_guard<std::recursive_mutex> lock(chunksMutex);
    IndirectRenderer &renderer = layer == LAYER_OPAQUE ? solidRenderer->getIndirectRenderer() : waterRenderer->getIndirectRenderer();
    auto &cur = layer == LAYER_OPAQUE ? solidChunks : transparentChunks;
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
    cur[nid] = mv;
    // Upload mesh into the renderer (may mark renderer dirty). The renderer
    // rebuild will perform GPU uploads; we keep that responsibility centralized
    // so uploads may be performed asynchronously inside the renderer.
    renderer.uploadMesh(app, meshId);
    // Rebuild is deferred to the end of processPendingMeshes() for efficiency.
    // When called from brush rebuild paths, the caller invokes rebuild() explicitly.

    // Generate vegetation instances for this node using the compute shader.
    // We create temporary device-local vertex/index buffers for the mesh geometry,
    // dispatch the compute shader to write instances, then free the temporary buffers.
    if (layer == LAYER_OPAQUE && vegetationRenderer) {
        if (geom.indices.size() >= 3 && !geom.vertices.empty()) {
            try {
                // Create tightly-packed position buffer (vec3[]) for the compute shader
                std::vector<glm::vec3> positions;
                positions.reserve(geom.vertices.size());
                for (const auto &v : geom.vertices) positions.push_back(v.position);

                VkDevice device = app->getDevice();

                Buffer posBuf = app->createDeviceLocalBuffer(positions.data(), positions.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                Buffer idxBuf = app->createDeviceLocalBuffer(geom.indices.data(), geom.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

                uint32_t vertexCount = static_cast<uint32_t>(positions.size());
                uint32_t indexCount = static_cast<uint32_t>(geom.indices.size());
                uint32_t instancesPerTriangle = 2u;
                uint32_t seed = static_cast<uint32_t>(nid & 0xffffffffull);

                // Pass Buffer objects to vegetationRenderer and let it defer destruction
                vegetationRenderer->generateChunkInstances(nid, posBuf, vertexCount, idxBuf, indexCount, instancesPerTriangle, app, seed);
            } catch (const std::exception &e) {
                std::cerr << "[SceneRenderer] Vegetation generation failed for node " << (unsigned long long)nid
                          << ": " << e.what() << std::endl;
            }
        }
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
        throw std::runtime_error("SceneRenderer::addDebugCubeForGeometry requires non-empty geometry (no fallback allowed)");
    }
    c.cube = BoundingBox(minp, maxp);
    c.color = (layer == LAYER_OPAQUE) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(0.0f, 0.5f, 1.0f);
    addDebugCubeForNode(nid, c);
}


// --- Brush scene change handlers ---

// Helper to update mesh for a brush node (uses brush chunk maps)
static void updateBrushMeshForNode(SceneRenderer* sr, VulkanApp* app, Layer layer, NodeID nid,
                                    const OctreeNodeData &nd, const Geometry &geom,
                                    std::unordered_map<NodeID, Model3DVersion>& chunkMap) {
    std::lock_guard<std::recursive_mutex> lock(sr->chunksMutex);
    IndirectRenderer &renderer = layer == LAYER_OPAQUE
        ? sr->solidRenderer->getIndirectRenderer()
        : sr->waterRenderer->getIndirectRenderer();

    auto it = chunkMap.find(nid);
    if (it != chunkMap.end()) {
        if (it->second.version >= nd.node->version) return;
        if (it->second.meshId != UINT32_MAX) {
            renderer.removeMesh(it->second.meshId);
        }
    }
    uint32_t meshId = renderer.addMesh(geom);
    Model3DVersion mv{meshId, nd.node->version};
    chunkMap[nid] = mv;
    renderer.uploadMesh(app, meshId);
    if (renderer.isDirty()) {
        renderer.rebuild(app);
    }
}

SolidSpaceChangeHandler SceneRenderer::makeBrushSolidSpaceChangeHandler(Scene* scene, VulkanApp* app) {
    brushSolidNodeEventCallback = [this, scene, app](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        if (nd.containmentType != ContainmentType::Disjoint) {
            OctreeNodeData nodeCopy = nd;
            this->processNodeLayer(*scene, LAYER_OPAQUE, nid, nodeCopy,
                [this, app](Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
                    updateBrushMeshForNode(this, app, layer, nid, nd, geom, this->brushSolidChunks);
                }
            );
        }
    };

    brushSolidNodeEraseCallback = [this](const OctreeNodeData& nd) {
        std::lock_guard<std::recursive_mutex> lock(chunksMutex);
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        auto it = brushSolidChunks.find(nid);
        if (it != brushSolidChunks.end()) {
            if (it->second.meshId != UINT32_MAX) {
                solidRenderer->getIndirectRenderer().removeMesh(it->second.meshId);
            }
            brushSolidChunks.erase(it);
        }
    };

    return SolidSpaceChangeHandler(brushSolidNodeEventCallback, brushSolidNodeEraseCallback);
}

LiquidSpaceChangeHandler SceneRenderer::makeBrushLiquidSpaceChangeHandler(Scene* scene, VulkanApp* app) {
    brushLiquidNodeEventCallback = [this, scene, app](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        if (nd.containmentType != ContainmentType::Disjoint) {
            OctreeNodeData nodeCopy = nd;
            this->processNodeLayer(*scene, LAYER_TRANSPARENT, nid, nodeCopy,
                [this, app](Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
                    updateBrushMeshForNode(this, app, layer, nid, nd, geom, this->brushTransparentChunks);
                }
            );
        }
    };

    brushLiquidNodeEraseCallback = [this](const OctreeNodeData& nd) {
        std::lock_guard<std::recursive_mutex> lock(chunksMutex);
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        auto it = brushTransparentChunks.find(nid);
        if (it != brushTransparentChunks.end()) {
            if (it->second.meshId != UINT32_MAX) {
                waterRenderer->getIndirectRenderer().removeMesh(it->second.meshId);
            }
            brushTransparentChunks.erase(it);
        }
    };

    return LiquidSpaceChangeHandler(brushLiquidNodeEventCallback, brushLiquidNodeEraseCallback);
}

void SceneRenderer::clearBrushMeshes() {
    std::lock_guard<std::recursive_mutex> lock(chunksMutex);
    // Remove all brush opaque meshes from solid renderer
    for (auto &entry : brushSolidChunks) {
        if (entry.second.meshId != UINT32_MAX) {
            solidRenderer->getIndirectRenderer().removeMesh(entry.second.meshId);
        }
    }
    brushSolidChunks.clear();

    // Remove all brush transparent meshes from water renderer
    for (auto &entry : brushTransparentChunks) {
        if (entry.second.meshId != UINT32_MAX) {
            waterRenderer->getIndirectRenderer().removeMesh(entry.second.meshId);
        }
    }
    brushTransparentChunks.clear();
}
