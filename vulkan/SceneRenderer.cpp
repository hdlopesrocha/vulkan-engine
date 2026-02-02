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
}

SceneRenderer::SceneRenderer(VulkanApp* app_)
    : app(app_),
      shadowMapper(std::make_unique<ShadowRenderer>(app_, 8192)),
      waterRenderer(std::make_unique<WaterRenderer>(app_)),
      skyRenderer(std::make_unique<SkyRenderer>(app_)),
      solidRenderer(std::make_unique<SolidRenderer>(app_)),
      vegetationRenderer(std::make_unique<VegetationRenderer>(app_))
{
    // All renderer members are now properly instantiated
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

void SceneRenderer::mainPass(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &mainPassInfo, uint32_t frameIdx, bool hasWater, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, bool wireframeEnabled, bool profilingEnabled, VkQueryPool queryPool, const glm::mat4 &viewProj,
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
    if (!solidRenderer) {
        fprintf(stderr, "[SceneRenderer::mainPass] solidRenderer is nullptr, skipping.\n");
        return;
    }
    solidRenderer->draw(commandBuffer, app, perTextureDescriptorSet, wireframeEnabled);
    //fprintf(stderr, "[SceneRenderer::mainPass] After solidRenderer->draw.\n");

    // Sky pass - render with depth test GREATER_OR_EQUAL so sky only shows through gaps
    if (!skyRenderer) {
        fprintf(stderr, "[SceneRenderer::mainPass] skyRenderer is nullptr, skipping.\n");
    } else {
        try {
            // Forward the valid main uniform buffer provided by the caller
            skyRenderer->render(commandBuffer, perTextureDescriptorSet, mainUniformBuffer, uboStatic, viewProj, SkyMode::Gradient);
            //fprintf(stderr, "[SceneRenderer::mainPass] After skyRenderer->render.\n");
        } catch (...) {
            fprintf(stderr, "[SceneRenderer::mainPass] Exception in skyRenderer->render.\n");
        }
    }

    // Vegetation pass
    if (!vegetationRenderer) {
        fprintf(stderr, "[SceneRenderer::mainPass] vegetationRenderer is nullptr, skipping.\n");
    } else {
        try {
            vegetationRenderer->draw(commandBuffer, perTextureDescriptorSet, viewProj);
            //fprintf(stderr, "[SceneRenderer::mainPass] After vegetationRenderer->draw.\n");
        } catch (...) {
            fprintf(stderr, "[SceneRenderer::mainPass] Exception in vegetationRenderer->draw.\n");
        }
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
    waterRenderer->updateSceneTexturesBinding(waterRenderer->getSceneColorView(frameIdx), waterRenderer->getSceneDepthView(frameIdx), frameIdx);

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

    // SkyRenderer exposes public init() which creates its pipelines
    if (skyRenderer) skyRenderer->init();



    // Water pipeline creation requires initialization with buffers/targets and is handled by WaterRenderer::init()/createRenderTargets elsewhere
    // Shadow pipeline creation is performed during ShadowRenderer::init()
    shadowMapper->init();

    if (vegetationRenderer) vegetationRenderer->createPipelines();
}

void SceneRenderer::init(VulkanApp* app_, SkyWidget* skyWidget, VkDescriptorSet descriptorSet) {
    if (!app_) {
        fprintf(stderr, "[SceneRenderer::init] app_ is nullptr!\n");
        return;
    }
    app = app_;
    
    // Allocate and initialize texture arrays
    if (vegetationRenderer) {
        textureArrayManager.allocate(64, 256, 256);
        vegetationRenderer->setTextureArrayManager(&textureArrayManager);
        vegetationRenderer->init(app);
        textureArrayManager.initialize(app);
    }
    
    // Create pipelines for all renderers
    createPipelines();
    
    if (solidRenderer) {
        solidRenderer->init(app);
    }
    
    // Create main uniform buffer
    mainUniformBuffer = app->createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    VkDescriptorSet mainDs = app->getMainDescriptorSet();
    printf("[SceneRenderer::init] mainDescriptorSet = 0x%llx\n", (unsigned long long)mainDs);
    
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
    VkDescriptorImageInfo albedoInfo{};
    albedoInfo.sampler = textureArrayManager.albedoSampler;
    albedoInfo.imageView = textureArrayManager.albedoArray.view;
    albedoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet albedoWrite{};
    albedoWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    albedoWrite.dstSet = mainDs;
    albedoWrite.dstBinding = 1;
    albedoWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    albedoWrite.descriptorCount = 1;
    albedoWrite.pImageInfo = &albedoInfo;

    VkDescriptorImageInfo normalInfo{};
    normalInfo.sampler = textureArrayManager.normalSampler;
    normalInfo.imageView = textureArrayManager.normalArray.view;
    normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet normalWrite{};
    normalWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    normalWrite.dstSet = mainDs;
    normalWrite.dstBinding = 2;
    normalWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    normalWrite.descriptorCount = 1;
    normalWrite.pImageInfo = &normalInfo;

    VkDescriptorImageInfo bumpInfo{};
    bumpInfo.sampler = textureArrayManager.bumpSampler;
    bumpInfo.imageView = textureArrayManager.bumpArray.view;
    bumpInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet bumpWrite{};
    bumpWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    bumpWrite.dstSet = mainDs;
    bumpWrite.dstBinding = 3;
    bumpWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bumpWrite.descriptorCount = 1;
    bumpWrite.pImageInfo = &bumpInfo;

    // Bind shadow map at binding 4
    VkDescriptorImageInfo shadowInfo{};
    shadowInfo.sampler = shadowMapper->getShadowMapSampler();
    shadowInfo.imageView = shadowMapper->getShadowMapView();
    shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet shadowWrite{};
    shadowWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    shadowWrite.dstSet = mainDs;
    shadowWrite.dstBinding = 4;
    shadowWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowWrite.descriptorCount = 1;
    shadowWrite.pImageInfo = &shadowInfo;

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
    
    app->updateDescriptorSet(mainDs, { uboWrite, albedoWrite, normalWrite, bumpWrite, shadowWrite, materialsWrite });

    // Initialize WaterRenderer
    Buffer waterParamsBuffer = app->createBuffer(sizeof(WaterUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    waterRenderer->init(waterParamsBuffer);
    waterRenderer->createRenderTargets(app->getWidth(), app->getHeight());
    
    // Initialize sky renderer with sphere VBO now that descriptor sets are ready
    if (skyRenderer) {
        skyRenderer->initSky(skyWidget, mainDs);
    }
    
    printf("[SceneRenderer::init] Initialization complete\n");
}
