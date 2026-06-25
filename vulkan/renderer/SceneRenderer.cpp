// ...existing code...
#include "SceneRenderer.hpp"


#include <stdexcept>
#include "../../utils/SolidSpaceChangeHandler.hpp"
#include "../../utils/LiquidSpaceChangeHandler.hpp"
#include "../../utils/LocalScene.hpp"
#include "../../math/ContainmentType.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cfloat>
#include <mutex>
#include <random>
#include <unordered_set>
#include <cstdlib>

namespace {
constexpr float kDebugSDFClip = 10.0f;

const uint32_t kDebugSDFFaces[6][4] = {
    {0, 1, 3, 2},
    {4, 6, 7, 5},
    {0, 4, 5, 1},
    {2, 3, 7, 6},
    {0, 2, 6, 4},
    {1, 5, 7, 3}
};

bool isDrawableSDF(float v) {
    return std::isfinite(v) && std::abs(v) <= kDebugSDFClip;
}

bool hasDrawableSDFFace(const std::array<float, 8>& sdf) {
    for (const auto& face : kDebugSDFFaces) {
        for (uint32_t corner : face) {
            if (isDrawableSDF(sdf[corner])) {
                return true;
            }
        }

        for (uint32_t edge = 0; edge < 4; ++edge) {
            const float a = sdf[face[edge]];
            const float b = sdf[face[(edge + 1) % 4]];
            if (std::isfinite(a) && std::isfinite(b) && ((a <= 0.0f && b >= 0.0f) || (a >= 0.0f && b <= 0.0f))) {
                return true;
            }
        }
    }
    return false;
}

void collectLeafSDFCubes(OctreeNode* node, const BoundingCube& cube, OctreeAllocator& allocator,
                         std::vector<DebugSDFRenderer::CubeSDF>& out) {
    if (!node) return;

    if (node->isSimplified()) {
        DebugSDFRenderer::CubeSDF debugCube{};
        debugCube.cube = cube;
        for (size_t i = 0; i < debugCube.sdf.size(); ++i) {
            debugCube.sdf[i] = node->sdf[i];
        }
        debugCube.brushIndex = node->vertex.brushIndex;
        if (hasDrawableSDFFace(debugCube.sdf)) {
            out.push_back(debugCube);
            return;  // Parent covers this subtree — children are redundant
        }
        // Parent is simplified but has no drawable faces; traverse children
        // in case individual child leaves still have visible SDF faces.
    }

    ChildBlock* block = node->getBlock(allocator);
    if (!block) return;
    for (uint32_t i = 0; i < 8; ++i) {
        OctreeNode* child = block->get(i, allocator);
        if (child && child != node) {
            collectLeafSDFCubes(child, cube.getChild(i), allocator, out);
        }
    }
}
}

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
    if (debugSDFRenderer) {
        debugSDFRenderer->cleanup();
    }
    if (solidWireframe) {
        solidWireframe->cleanup();
    }
    if (waterWireframe) {
        waterWireframe->cleanup();
    }

    // Clear local CPU-side handles; Vulkan objects are destroyed via VulkanResourceManager
    for (auto &b : mainUniformBuffers) {
        if (b.buffer != VK_NULL_HANDLE) b = {};
    }
    mainUniformBuffers.clear();

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
            solid360Renderer->createSolid360Targets(app, waterRenderer->getLinearSampler());
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
    debugSDFRenderer(std::make_unique<DebugSDFRenderer>()),
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

void SceneRenderer::shadowPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkDescriptorSet mainDescriptorSet, Buffer &mainUniformBuffer, const UniformObject &uboStatic, bool shadowsEnabled, bool vegetationEnabled) {
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

        // Wait for previous cascade draws to finish reading the UBO
        {
            VkMemoryBarrier preBarrier{};
            preBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            preBarrier.srcAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
            preBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &preBarrier, 0, nullptr, 0, nullptr);
        }

        vkCmdUpdateBuffer(commandBuffer, mainUniformBuffer.buffer, 0, sizeof(UniformObject), &shadowUBO);
        {
            VkMemoryBarrier memBarrier{};
            memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT;
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 1, &memBarrier, 0, nullptr, 0, nullptr);
        }

        // Acquire vegetation instance/indirect buffers before
        // vkCmdBeginRendering (barriers illegal inside dynamic rendering).
        if (vegetationEnabled && vegetationRenderer) {
            vegetationRenderer->recordReadBarriers(commandBuffer);
        }

        shadowMapper->beginShadowPass(app, commandBuffer, c, lsMatrix);

        // Bind shadow descriptor set (uses dummy depth at bindings 4,8,9)
        VkPipelineLayout layout = shadowMapper->getShadowPipelineLayout();
        VkDescriptorSet ds = VK_NULL_HANDLE;
        if (!shadowDescriptorSets.empty()) {
            uint32_t idx = app->getCurrentFrame() % static_cast<uint32_t>(shadowDescriptorSets.size());
            ds = shadowDescriptorSets[idx];
        }
        if (layout != VK_NULL_HANDLE && ds != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &ds, 0, nullptr);
        }

        // Bind solid shadow pipeline first and draw solid geometry.
        // Must draw solid BEFORE vegetation: VegetationRenderer::drawShadow
        // binds 2 vertex buffers (bindings 0 and 1) and a different pipeline.
        // On RADV, stale binding 1 leaks across pipeline switches and corrupts
        // subsequent solid draws that only use binding 0.
        VkPipeline solidShadowPipeline = shadowMapper->getShadowPipeline();
        if (solidShadowPipeline != VK_NULL_HANDLE) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, solidShadowPipeline);
        }
        solidRenderer->getIndirectRenderer().drawAll(commandBuffer);

        // Vegetation shadow pass: drawn after solid so its 2-buffer vertex
        // bindings don't leak into the solid draw.
        if (vegetationEnabled && vegetationRenderer) {
            const glm::vec3 cameraPos = glm::vec3(uboStatic.viewPos);
            vegetationRenderer->drawShadow(app, commandBuffer, ds, uboStatic.viewProjection, cameraPos);
        }

        shadowMapper->endShadowPass(app, commandBuffer, c);
    }

    // Restore the main UBO so subsequent passes see the original data.
    // Wait for all shadow cascade draws to finish reading the UBO first.
    {
        VkMemoryBarrier preBarrier{};
        preBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        preBarrier.srcAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        preBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &preBarrier, 0, nullptr, 0, nullptr);
    }

    vkCmdUpdateBuffer(commandBuffer, mainUniformBuffer.buffer, 0, sizeof(UniformObject), &uboStatic);
    {
        VkMemoryBarrier memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }
}

void SceneRenderer::mainPass(VulkanApp* app, VkCommandBuffer &commandBuffer, uint32_t frameIdx, bool hasWater, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, bool renderSolid, bool wireframeEnabled, const glm::mat4 &viewProj,
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
    if (renderSolid) {
        solidRenderer->renderDepthPrepass(commandBuffer, app, perTextureDescriptorSet);
        solidRenderer->render(commandBuffer, app, perTextureDescriptorSet);
        if (wireframeEnabled) {
            solidWireframe->draw(commandBuffer, app, {perTextureDescriptorSet}, solidRenderer->getIndirectRenderer());
        } 
    }
    
}

void SceneRenderer::skyPass(VulkanApp* app, VkCommandBuffer &commandBuffer, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, const UniformObject &uboStatic, const glm::mat4 &viewProj) {
    SkySettings::Mode mode = skySettings->mode;
    skyRenderer->render(app, commandBuffer, perTextureDescriptorSet, mainUniformBuffer, uboStatic, viewProj, mode);
}

void SceneRenderer::drawSolidWireframeOverlay(VulkanApp* app, VkCommandBuffer &commandBuffer, uint32_t frameIdx, VkDescriptorSet perTextureDescriptorSet, bool wireframeEnabled) {
    solidWireframe->draw(commandBuffer, app, {perTextureDescriptorSet}, solidRenderer->getIndirectRenderer());
}

void SceneRenderer::waterPass(VulkanApp* app, VkCommandBuffer &commandBuffer, uint32_t frameIdx, VkDescriptorSet perTextureDescriptorSet, bool waterWireframeEnabled, float waterTime, bool skipBackFace, VkImageView skyView, VkImageView cubeReflectionView) {
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
    VkImage sceneDepthImage = solidRenderer->getDepthImage(frameIdx);

    const char* _wg_dis = std::getenv("VULKAN_DISABLE_WATERGEOM");
    bool _wg_env_skip = (_wg_dis && _wg_dis[0] != '\0');
    if (_wg_env_skip) std::cerr << "[SceneRenderer] VULKAN_DISABLE_WATERGEOM set; skipping water geometry operations" << std::endl;

    // Initialize water geometry depth from the scene depth so water geometry
    // rasterization can depth-test against solid geometry and avoid rendering
    // water where it is occluded by solids.
    if (waterRenderer && !_wg_env_skip) {
        waterRenderer->initializeGeomDepthFromSceneDepth(app, commandBuffer, frameIdx, sceneDepthImage);
    }

    // Scene textures were already bound before the async back-face/solid360 tasks were
    // launched (see main.cpp), so we must NOT call updateSceneTexturesBinding here.
    // Calling it after the async tasks submit their command buffers would update a
    // descriptor set that is already referenced by a pending command buffer
    // (VUID-vkUpdateDescriptorSets-None-03047).

    bool wf = waterWireframeEnabled;
    if (wf && waterWireframe && waterWireframe->getPipeline() != VK_NULL_HANDLE) {
        // Wireframe path: use WaterRenderer for setup/pass management,
        // but bind the wireframe pipeline instead of the normal one.
        if (!_wg_env_skip) {
            waterRenderer->prepareRender(app, commandBuffer, frameIdx, sceneColorView, sceneDepthView, skyView);
            const char* _bf_dis = std::getenv("VULKAN_DISABLE_BACKFACE");
            bool _bf_env_skip = (_bf_dis && _bf_dis[0] != '\0');
            if (_bf_env_skip) std::cerr << "[SceneRenderer] VULKAN_DISABLE_BACKFACE set; skipping back-face pass" << std::endl;
            if (backFaceRenderer && !skipBackFace && !_bf_env_skip) {
                backFaceRenderer->renderBackFacePass(app, commandBuffer, frameIdx,
                                                    waterRenderer->getIndirectRenderer(),
                                                    waterRenderer->getWaterGeometryPipelineLayout(),
                                                    app->getMainDescriptorSet(),
                                                    app->getMaterialDescriptorSet(),
                                                    waterRenderer->getWaterDepthDescriptorSet(frameIdx),
                                                    solidRenderer->getDepthImage(frameIdx));
            }
            waterRenderer->beginWaterGeometryPass(commandBuffer, frameIdx);

            // First render filled water geometry to populate the water depth
            // buffer so the wireframe can depth-test against actual water depth.
            VkPipeline waterPipe = waterRenderer->getWaterGeometryPipeline();
            VkPipelineLayout waterLayout = waterRenderer->getWaterGeometryPipelineLayout();
            if (waterPipe != VK_NULL_HANDLE && waterLayout != VK_NULL_HANDLE) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, waterPipe);

                VkDescriptorSet mainDs = app->getMainDescriptorSet();
                if (mainDs != VK_NULL_HANDLE) {
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, waterLayout, 0, 1, &mainDs, 0, nullptr);
                }

                VkDescriptorSet materialDs = app->getMaterialDescriptorSet();
                if (materialDs != VK_NULL_HANDLE) {
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, waterLayout, 1, 1, &materialDs, 0, nullptr);
                }

                VkDescriptorSet sceneDs = waterRenderer->getWaterDepthDescriptorSet(frameIdx);
                if (sceneDs != VK_NULL_HANDLE) {
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, waterLayout, 2, 1, &sceneDs, 0, nullptr);
                }

                // Draw filled water geometry (will update depth buffer)
                waterRenderer->getIndirectRenderer().drawPrepared(commandBuffer);
            }

            // Draw wireframe overlay on top, inside the same render pass,
            // reusing the depth buffer populated by the filled geometry pass.
            // Bind descriptor sets individually with null checks (same pattern
            // as the filled water pipeline) to handle missing sets gracefully.
            VkPipeline waterWfPipe = waterWireframe->getPipeline();
            VkPipelineLayout wfLayout = waterWireframe->getPipelineLayout();
            if (waterWfPipe != VK_NULL_HANDLE && wfLayout != VK_NULL_HANDLE) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, waterWfPipe);

                VkDescriptorSet wfMainDs = app->getMainDescriptorSet();
                if (wfMainDs != VK_NULL_HANDLE)
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wfLayout, 0, 1, &wfMainDs, 0, nullptr);

                VkDescriptorSet wfMatDs = app->getMaterialDescriptorSet();
                if (wfMatDs != VK_NULL_HANDLE)
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wfLayout, 1, 1, &wfMatDs, 0, nullptr);

                VkDescriptorSet wfDepthDs = waterRenderer->getWaterDepthDescriptorSet(frameIdx);
                if (wfDepthDs != VK_NULL_HANDLE)
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wfLayout, 2, 1, &wfDepthDs, 0, nullptr);

                waterRenderer->getIndirectRenderer().drawPrepared(commandBuffer);
            }

            waterRenderer->endWaterGeometryPass(commandBuffer);
            waterRenderer->postRenderBarrier(commandBuffer, frameIdx);
        } else {
            // Skipping water geometry operations as requested by env guard
        }
    } else {
        const char* _bf_dis2 = std::getenv("VULKAN_DISABLE_BACKFACE");
        bool _bf_env_skip2 = (_bf_dis2 && _bf_dis2[0] != '\0');
        if (_bf_env_skip2) std::cerr << "[SceneRenderer] VULKAN_DISABLE_BACKFACE set; skipping back-face pass" << std::endl;
        if (backFaceRenderer && !skipBackFace && !_bf_env_skip2) {
            backFaceRenderer->renderBackFacePass(app, commandBuffer, frameIdx,
                                                waterRenderer->getIndirectRenderer(),
                                                waterRenderer->getWaterGeometryPipelineLayout(),
                                                app->getMainDescriptorSet(),
                                                app->getMaterialDescriptorSet(),
                                                waterRenderer->getWaterDepthDescriptorSet(frameIdx),
                                                solidRenderer->getDepthImage(frameIdx));
        }
        if (!_wg_env_skip) {
            waterRenderer->render(app, commandBuffer, frameIdx, sceneColorView, sceneDepthView, skyView);
        } else {
            // Skipping waterRenderer::render due to VULKAN_DISABLE_WATERGEOM
        }
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
    skyRenderer->init(app);
    // Create offscreen sky targets so the sky can be sampled as a texture by water
    skyRenderer->createOffscreenTargets(app, app->getWidth(), app->getHeight());
    shadowMapper->init(app);
    vegetationRenderer->init(app);

    // Initialize debug cube renderer
    if (debugCubeRenderer) {
        debugCubeRenderer->init(app);
    }
    // Initialize bounding box renderer (reuses cube wireframe pipeline)
    if (boundingBoxRenderer) {
        boundingBoxRenderer->init(app);
    }
    if (debugSDFRenderer) {
        debugSDFRenderer->init(app);
    }
    
    // Create per-frame main uniform buffers (TRANSFER_DST for vkCmdUpdateBuffer in cubemap 360 render)
    size_t dsCount = app->getMainDescriptorSetCount();
    if (dsCount == 0) dsCount = 1;
    mainUniformBuffers.clear();
    mainUniformBuffers.resize(dsCount);
    for (size_t i = 0; i < dsCount; ++i) {
        mainUniformBuffers[i] = app->createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    VkDescriptorSet mainDs = app->getMainDescriptorSetForFrame(0);
    printf("[SceneRenderer::init] mainDescriptorSet[0] = 0x%llx\n", (unsigned long long)mainDs);

    // Initialize sky renderer with our owned settings now that descriptor sets are ready.
    // Must write Sky UBO to ALL per-frame descriptor sets, not just frame 0.
    if (skyRenderer) {
        skyRenderer->init(app, *skySettings, mainDs);
        // Write Sky UBO to remaining per-frame descriptor sets
        for (uint32_t i = 1; i < static_cast<uint32_t>(app->getMainDescriptorSetCount()); ++i) {
            VkDescriptorSet ds = app->getMainDescriptorSetForFrame(i);
            if (ds != VK_NULL_HANDLE) {
                skyRenderer->init(app, *skySettings, ds);
            }
        }
    }
    
    // Prepare UBO write template (dstSet and pBufferInfo will be set per-frame)
    VkWriteDescriptorSet uboWrite{};
    uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    uboWrite.dstBinding = 0;
    uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboWrite.descriptorCount = 1;

    printf("[SceneRenderer::init] Binding UBO buffers for %zu frames\n", mainUniformBuffers.size());
    
    // Bind texture arrays (bindings 1, 2, 3)
    // Prepare descriptor writes dynamically: only include image samplers that have valid image views
    std::vector<VkWriteDescriptorSet> writes;

    // UBO write (always present) — update for each per-frame descriptor set
    // We'll write descriptors per-frame so each descriptor set references a dedicated UBO buffer.

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
    VkDescriptorBufferInfo* materialsInfo = new VkDescriptorBufferInfo{ materialsBuffer.buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet materialsWrite{};
    materialsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    materialsWrite.dstSet = mainDs;
    materialsWrite.dstBinding = 5;
    materialsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    materialsWrite.descriptorCount = 1;
    materialsWrite.pBufferInfo = materialsInfo;
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

    // Initialize WaterRenderer (creates its pipeline layout and initializes the param SSBO)
    waterRenderer->init(app, waterParamsBuffer_, waterParams, layerCount);

    // Now that WaterRenderer has created its pipeline layout, allow the
    // back-face renderer to create pipelines that depend on it.
    if (backFaceRenderer) backFaceRenderer->createPipelines(app, waterRenderer->getWaterGeometryPipelineLayout());
    // Create back-face render targets early so their image views are
    // available before the first frame's water pass attempts to bind them.
    if (backFaceRenderer) backFaceRenderer->createRenderTargets(app, app->getWidth(), app->getHeight());
    if (solid360Renderer) {
        solid360Renderer->init(app);
        // Create cubemap targets now so the image view is available for
        // the environment-map descriptor binding (binding 11) below.
        solid360Renderer->createSolid360Targets(app, waterRenderer->getLinearSampler());
        // Binding 11: environment cubemap for solid-shader reflections
        VkImageView cubeView = solid360Renderer->getSolid360View();
        VkSampler cubeSampler = solid360Renderer->getSolid360Sampler();
        if (cubeView != VK_NULL_HANDLE && cubeSampler != VK_NULL_HANDLE) {
            addImageWrite(11, cubeSampler, cubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }

    // Bind water params SSBO to binding 7 of main descriptor set.
    // Allocate on heap — the cleanup loop below uniformly delete-s all pBufferInfo.
    VkDescriptorBufferInfo* waterParamsInfo = new VkDescriptorBufferInfo{ waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet waterParamsWrite{};
    waterParamsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    waterParamsWrite.dstSet = mainDs;
    waterParamsWrite.dstBinding = 7;
    waterParamsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    waterParamsWrite.descriptorCount = 1;
    waterParamsWrite.pBufferInfo = waterParamsInfo;
    writes.push_back(waterParamsWrite);

    // Bind water render UBO to binding 10 of main descriptor set
    waterRenderUBOBuffer_ = app->createBuffer(sizeof(WaterRenderUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDescriptorBufferInfo* waterRenderUBOInfo = new VkDescriptorBufferInfo{ waterRenderUBOBuffer_.buffer, 0, sizeof(WaterRenderUBO) };
    VkWriteDescriptorSet waterRenderUBOWrite{};
    waterRenderUBOWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    waterRenderUBOWrite.dstSet = mainDs;
    waterRenderUBOWrite.dstBinding = 10;
    waterRenderUBOWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    waterRenderUBOWrite.descriptorCount = 1;
    waterRenderUBOWrite.pBufferInfo = waterRenderUBOInfo;
    writes.push_back(waterRenderUBOWrite);

    // Perform descriptor updates per-frame
    for (size_t fi = 0; fi < mainUniformBuffers.size(); ++fi) {
        std::vector<VkWriteDescriptorSet> frameWrites;
        frameWrites.reserve(writes.size());

        // UBO for this frame — allocate on heap so the cleanup loop below
        // can uniformly delete all pBufferInfo pointers.
        VkDescriptorBufferInfo* mainBufInfo = new VkDescriptorBufferInfo{ mainUniformBuffers[fi].buffer, 0, sizeof(UniformObject) };
        VkWriteDescriptorSet uboWriteLocal = uboWrite;
        uboWriteLocal.dstSet = app->getMainDescriptorSetForFrame(static_cast<uint32_t>(fi));
        uboWriteLocal.pBufferInfo = mainBufInfo;
        frameWrites.push_back(uboWriteLocal);

        // Rebuild image/buffer writes for the other bindings using the same logic as above
        for (auto &w : writes) {
            if (w.dstBinding == 0) continue; // skip original ubo placeholder
            VkWriteDescriptorSet copy = w;
            // For image writes we need to allocate fresh VkDescriptorImageInfo so pointers are valid
            if (w.pImageInfo) {
                VkDescriptorImageInfo* info = new VkDescriptorImageInfo();
                *info = w.pImageInfo[0];
                copy.pImageInfo = info;
            } else if (w.pBufferInfo) {
                VkDescriptorBufferInfo* info = new VkDescriptorBufferInfo();
                *info = w.pBufferInfo[0];
                copy.pBufferInfo = info;
            }
            copy.dstSet = app->getMainDescriptorSetForFrame(static_cast<uint32_t>(fi));
            frameWrites.push_back(copy);
        }

        app->updateDescriptorSet(frameWrites);

        // Clean up any allocated infos for this frameWrites
        for (auto &w : frameWrites) {
            if (w.pImageInfo) delete w.pImageInfo;
            if (w.pBufferInfo) delete w.pBufferInfo;
        }
    }
    // Clean up original template writes' allocated infos
    for (auto &w : writes) {
        if (w.pImageInfo) { delete w.pImageInfo; w.pImageInfo = nullptr; }
        if (w.pBufferInfo) { delete w.pBufferInfo; w.pBufferInfo = nullptr; }
    }

    // ── Allocate shadow-specific descriptor sets per-frame (mirror main sets but use a dummy depth view)
    shadowDescriptorSets.clear();
    shadowDescriptorSets.resize(mainUniformBuffers.size());
    for (size_t fi = 0; fi < shadowDescriptorSets.size(); ++fi) {
        VkDescriptorSet ds = app->createDescriptorSet(app->getDescriptorSetLayout());
        shadowDescriptorSets[fi] = ds;

        VkDescriptorBufferInfo shadowBufInfo{ mainUniformBuffers[fi].buffer, 0, sizeof(UniformObject) };
        VkWriteDescriptorSet shadowUboWrite{};
        shadowUboWrite.sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        shadowUboWrite.dstSet         = ds;
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
            w.dstSet         = ds;
            w.dstBinding     = binding;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.descriptorCount = 1;
            w.pImageInfo     = info;
            shadowWrites.push_back(w);
        };

        if (textureArrayManager) {
            addShadowImageWrite(1, textureArrayManager->albedoSampler, textureArrayManager->albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addShadowImageWrite(2, textureArrayManager->normalSampler, textureArrayManager->normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addShadowImageWrite(3, textureArrayManager->bumpSampler, textureArrayManager->bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        addShadowImageWrite(4, shadowMapper->getShadowMapSampler(), shadowMapper->getDummyDepthView(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        addShadowImageWrite(8, shadowMapper->getShadowMapSampler(), shadowMapper->getDummyDepthView(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        addShadowImageWrite(9, shadowMapper->getShadowMapSampler(), shadowMapper->getDummyDepthView(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

        // Binding 11: environment cubemap (if available) for shadow passes that use the same layout
        if (solid360Renderer) {
            VkImageView cubeView = solid360Renderer->getSolid360View();
            VkSampler cubeSampler = solid360Renderer->getSolid360Sampler();
            if (cubeView != VK_NULL_HANDLE && cubeSampler != VK_NULL_HANDLE) {
                addShadowImageWrite(11, cubeSampler, cubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }

        VkDescriptorBufferInfo shadowMatInfo{ materialsBuffer.buffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet shadowMatWrite{};
        shadowMatWrite.sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        shadowMatWrite.dstSet         = ds;
        shadowMatWrite.dstBinding     = 5;
        shadowMatWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        shadowMatWrite.descriptorCount = 1;
        shadowMatWrite.pBufferInfo    = &shadowMatInfo;
        shadowWrites.push_back(shadowMatWrite);

        VkDescriptorBufferInfo shadowWaterInfo{ waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet shadowWaterWrite{};
        shadowWaterWrite.sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        shadowWaterWrite.dstSet         = ds;
        shadowWaterWrite.dstBinding     = 7;
        shadowWaterWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        shadowWaterWrite.descriptorCount = 1;
        shadowWaterWrite.pBufferInfo    = &shadowWaterInfo;
        shadowWrites.push_back(shadowWaterWrite);

        VkDescriptorBufferInfo shadowWaterRenderInfo{ waterRenderUBOBuffer_.buffer, 0, sizeof(WaterRenderUBO) };
        VkWriteDescriptorSet shadowWaterRenderWrite{};
        shadowWaterRenderWrite.sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        shadowWaterRenderWrite.dstSet         = ds;
        shadowWaterRenderWrite.dstBinding     = 10;
        shadowWaterRenderWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        shadowWaterRenderWrite.descriptorCount = 1;
        shadowWaterRenderWrite.pBufferInfo    = &shadowWaterRenderInfo;
        shadowWrites.push_back(shadowWaterRenderWrite);

        app->updateDescriptorSet(shadowWrites);
        for (auto &w : shadowWrites) {
            if (w.pImageInfo) delete w.pImageInfo;
        }
        std::cerr << "[SceneRenderer::init] shadowDescriptorSet[" << fi << "] = " << (void*)ds << std::endl;
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

    // Create wireframe pipelines for solid and water passes
    if (solidWireframe) {
        std::vector<VkDescriptorSetLayout> solidSetLayouts = { app->getDescriptorSetLayout() };
        solidWireframe->createPipeline(app, {app->getSwapchainImageFormat()},
            solidSetLayouts,
            "shaders/main.vert.spv", "shaders/wireframe.frag.spv",
            "shaders/main.tesc.spv", "shaders/main.tese.spv",
            "solid wireframe");
    }
    if (waterWireframe) {
        std::vector<VkDescriptorSetLayout> waterSetLayouts = {
            app->getDescriptorSetLayout(),
            app->getMaterialDescriptorSetLayout(),
            waterRenderer->getWaterDepthDescriptorSetLayout()
        };
        waterWireframe->createPipeline(app, {VK_FORMAT_R32G32B32A32_SFLOAT},
            waterSetLayouts,
            "shaders/water.vert.spv", "shaders/water_wireframe.frag.spv",
            "shaders/water.tesc.spv", "shaders/water.tese.spv",
            "water wireframe");
    }

    // Initialize post-process renderer (composites scene + water into swapchain)
    postProcessRenderer->init(app);
    postProcessRenderer->setRenderSize(app->getWidth(), app->getHeight());
    
    // Initialize sky renderer with sphere VBO now that descriptor sets are ready.
    // Write Sky UBO to all per-frame descriptor sets.
    skyRenderer->init(app, *skySettings, mainDs);
    for (uint32_t i = 1; i < static_cast<uint32_t>(app->getMainDescriptorSetCount()); ++i) {
        VkDescriptorSet ds = app->getMainDescriptorSetForFrame(i);
        if (ds != VK_NULL_HANDLE) {
            skyRenderer->init(app, *skySettings, ds);
        }
    }
    

    printf("[SceneRenderer::init] Initialization complete\n");
}

// Update only the texture / materials bindings in the app's main descriptor set.
void SceneRenderer::updateTextureDescriptorSet(VulkanApp* app, TextureArrayManager * textureArrayManager) {
    if (!app) return;

    // Update ALL main descriptor sets (double-buffered).  Vegetation and
    // solid draws use getMainDescriptorSet() which round-robins through
    // mainDescriptorSets; if only the current-frame set is updated, the
    // other frame's draws miss shadow-map bindings → no self-shadowing.
    const size_t setCount = app->getMainDescriptorSetCount();
    for (size_t s = 0; s < setCount; ++s) {
    VkDescriptorSet mainDs = app->getMainDescriptorSetForFrame(static_cast<uint32_t>(s));
    if (mainDs == VK_NULL_HANDLE) continue;

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
        VkDescriptorBufferInfo* materialsInfo = new VkDescriptorBufferInfo{ materialsBuffer.buffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet materialsWrite{};
        materialsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        materialsWrite.dstSet = mainDs;
        materialsWrite.dstBinding = 5;
        materialsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        materialsWrite.descriptorCount = 1;
        materialsWrite.pBufferInfo = materialsInfo;
        writes.push_back(materialsWrite);
    } else {
        std::cerr << "[SceneRenderer::updateTextureDescriptorSet] materials buffer not available — skipping binding 5\n";
    }

    // Rebind water params UBO at binding 7 (must stay valid after texture re-allocation)
    if (waterParamsBuffer_.buffer != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo* waterParamsInfo = new VkDescriptorBufferInfo{ waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet waterParamsWrite{};
        waterParamsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        waterParamsWrite.dstSet = mainDs;
        waterParamsWrite.dstBinding = 7;
        waterParamsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        waterParamsWrite.descriptorCount = 1;
        waterParamsWrite.pBufferInfo = waterParamsInfo;
        writes.push_back(waterParamsWrite);
    }

    // Apply descriptor updates
    app->updateDescriptorSet(writes);

    // cleanup allocated infos
    for (auto &w : writes) {
        if (w.pImageInfo) delete w.pImageInfo;
        if (w.pBufferInfo) delete w.pBufferInfo;
    }

    // ── Also update shadow descriptor sets (bindings 1-3, 5, 7) with new textures/materials.
    //    Bindings 4, 8, 9 stay as the dummy depth image (never changes).
    if (!shadowDescriptorSets.empty()) {
        for (size_t si = 0; si < shadowDescriptorSets.size(); ++si) {
            VkDescriptorSet ds = shadowDescriptorSets[si];
            std::vector<VkWriteDescriptorSet> shadowWrites;
            auto addShadowImageWrite = [&](uint32_t binding, VkSampler sampler, VkImageView view, VkImageLayout layout) {
                if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
                VkDescriptorImageInfo* info = new VkDescriptorImageInfo();
                info->sampler = sampler; info->imageView = view; info->imageLayout = layout;
                VkWriteDescriptorSet w{}; w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = ds; w.dstBinding = binding;
                w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.descriptorCount = 1;
                w.pImageInfo = info;
                shadowWrites.push_back(w);
            };
            addShadowImageWrite(1, textureArrayManager->albedoSampler, textureArrayManager->albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addShadowImageWrite(2, textureArrayManager->normalSampler, textureArrayManager->normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addShadowImageWrite(3, textureArrayManager->bumpSampler, textureArrayManager->bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (materialsBuffer.buffer != VK_NULL_HANDLE) {
                VkDescriptorBufferInfo* sMatInfo = new VkDescriptorBufferInfo{ materialsBuffer.buffer, 0, VK_WHOLE_SIZE };
                VkWriteDescriptorSet sMatWrite{}; sMatWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                sMatWrite.dstSet = ds; sMatWrite.dstBinding = 5;
                sMatWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; sMatWrite.descriptorCount = 1;
                sMatWrite.pBufferInfo = sMatInfo;
                shadowWrites.push_back(sMatWrite);
            } else {
                std::cerr << "[SceneRenderer::updateTextureDescriptorSet] shadow materials buffer not available — skipping shadow binding 5\n";
            }
            if (waterParamsBuffer_.buffer != VK_NULL_HANDLE) {
                VkDescriptorBufferInfo* sWaterInfo = new VkDescriptorBufferInfo{ waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE };
                VkWriteDescriptorSet sWaterWrite{}; sWaterWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                sWaterWrite.dstSet = ds; sWaterWrite.dstBinding = 7;
                sWaterWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; sWaterWrite.descriptorCount = 1;
                sWaterWrite.pBufferInfo = sWaterInfo;
                shadowWrites.push_back(sWaterWrite);
            }
            app->updateDescriptorSet(shadowWrites);
            for (auto &w : shadowWrites) { if (w.pImageInfo) delete w.pImageInfo; if (w.pBufferInfo) delete w.pBufferInfo; }
        }
    }
    } // for each main descriptor set

    std::cerr << "[SceneRenderer] updateTextureDescriptorSet: updated main descriptor set with texture bindings" << std::endl;
}



// Drain whatever CPU-generated mesh data the background loading thread has
// queued since the last frame, and perform the actual Vulkan GPU uploads.
// Must be called from the main (render) thread each frame.
void SceneRenderer::processPendingMeshes(VulkanApp* app, glm::vec3 cameraPos) {
    // Cap uploads per frame so the render loop stays responsive.
    // Remaining entries stay in the queue for subsequent frames.
    static constexpr size_t kMaxPerFrame = 10;
    std::deque<PendingMeshData> batch;
    {
        std::lock_guard<std::mutex> lock(pendingMeshMutex);
        // Sort by ascending distance so chunks closest to the camera are uploaded first.
        std::sort(pendingMeshQueue.begin(), pendingMeshQueue.end(),
            [&cameraPos](const PendingMeshData& a, const PendingMeshData& b) {
                glm::vec3 da = cameraPos - a.nodeData.cube.getCenter();
                glm::vec3 db = cameraPos - b.nodeData.cube.getCenter();
                return glm::dot(da, da) < glm::dot(db, db);
            });
        size_t n = std::min(pendingMeshQueue.size(), kMaxPerFrame);
        batch.insert(batch.end(),
                     std::make_move_iterator(pendingMeshQueue.begin()),
                     std::make_move_iterator(pendingMeshQueue.begin() + n));
        pendingMeshQueue.erase(pendingMeshQueue.begin(), pendingMeshQueue.begin() + n);
    }
    if (batch.empty()) return;

    // Compute per-layer totals for the incoming batch so we can pre-size
    // renderer buffers and enable incremental uploads when possible.
    size_t solidNewV = 0, solidNewI = 0, solidNewM = 0;
    size_t waterNewV = 0, waterNewI = 0, waterNewM = 0;
    for (const auto &pd : batch) {
        if (pd.layer == LAYER_OPAQUE) {
            solidNewV += pd.geom.vertices.size();
            solidNewI += pd.geom.indices.size();
            solidNewM += 1;
        } else {
            waterNewV += pd.geom.vertices.size();
            waterNewI += pd.geom.indices.size();
            waterNewM += 1;
        }
    }

    IndirectRenderer& solidIR = solidRenderer->getIndirectRenderer();
    IndirectRenderer& waterIR = waterRenderer->getIndirectRenderer();

    bool solidCanIncremental = true;
    bool waterCanIncremental = true;

    if (solidNewM > 0) {
        size_t desiredV = solidIR.getMergedVertexCount() + solidNewV;
        size_t desiredI = solidIR.getMergedIndexCount() + solidNewI;
        size_t desiredM = solidIR.getMeshCount() + solidNewM;
        solidCanIncremental = solidIR.ensureCapacity(desiredV, desiredI, desiredM);
#if 0
        if (solidCanIncremental) {
            printf("[SceneRenderer::processPendingMeshes] Solid renderer pre-sized for incremental uploads (v=%zu,i=%zu,m=%zu)\n", desiredV, desiredI, desiredM);
        } else {
            printf("[SceneRenderer::processPendingMeshes] Solid renderer needs rebuild to grow capacity (v=%zu,i=%zu,m=%zu)\n", desiredV, desiredI, desiredM);
        }
#endif
    }

    if (waterNewM > 0) {
        size_t desiredV = waterIR.getMergedVertexCount() + waterNewV;
        size_t desiredI = waterIR.getMergedIndexCount() + waterNewI;
        size_t desiredM = waterIR.getMeshCount() + waterNewM;
        waterCanIncremental = waterIR.ensureCapacity(desiredV, desiredI, desiredM);
#if 0
        if (waterCanIncremental) {
            printf("[SceneRenderer::processPendingMeshes] Water renderer pre-sized for incremental uploads (v=%zu,i=%zu,m=%zu)\n", desiredV, desiredI, desiredM);
        } else {
            printf("[SceneRenderer::processPendingMeshes] Water renderer needs rebuild to grow capacity (v=%zu,i=%zu,m=%zu)\n", desiredV, desiredI, desiredM);
        }
#endif
    }

    // Process meshes and attempt incremental upload only when pre-sizing succeeded.
    bool solidOrWaterHadRemovals = false;
    for (auto& pd : batch) {
        bool attemptUpload = (pd.layer == LAYER_OPAQUE) ? solidCanIncremental : waterCanIncremental;
        updateMeshForNode(app, pd.layer, pd.nid, pd.nodeData, pd.geom, attemptUpload, pd.version, &solidOrWaterHadRemovals);
    }

    // Batch rebuild: only needed when buffers were created/grown (canIncremental == false)
    // OR when meshes were removed — removals leave stale indirect commands on GPU
    // that must be compacted away by a full rebuild.
    if (solidIR.isDirty()) {
        if (solidCanIncremental && solidNewM > 0 && !solidOrWaterHadRemovals) {
            solidIR.setDirty(false);
        } else {
            solidIR.rebuild(app);
        }
    }
    if (waterIR.isDirty()) {
        if (waterCanIncremental && waterNewM > 0 && !solidOrWaterHadRemovals) {
            waterIR.setDirty(false);
        } else {
            waterIR.rebuild(app);
        }
    }
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
        if (auto* localScene = dynamic_cast<LocalScene*>(scene)) {
            this->updateDebugSDFCubesForChunk(nid, nd, localScene->getOpaqueOctree());
        }
        OctreeNodeData nodeCopy = nd;
        this->processNodeLayer(*scene, LAYER_OPAQUE, nid, nodeCopy,
            [this](Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
                std::lock_guard<std::mutex> lock(pendingMeshMutex);
                pendingMeshQueue.push_back({layer, nid, nd, geom, nd.node->version});
            }
        );
    
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
        }
        removeDebugCubeForNode(nid);
        removeDebugSDFCubesForNode(nid);
    };
    
    return SolidSpaceChangeHandler(solidNodeEventCallback, solidNodeEraseCallback);
}

LiquidSpaceChangeHandler SceneRenderer::makeLiquidSpaceChangeHandler(Scene* scene, VulkanApp* app) {

    liquidNodeEventCallback = [this, scene](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        // CPU tessellation runs here (background thread); result is queued for
        // main-thread GPU upload via processPendingMeshes().
        if (auto* localScene = dynamic_cast<LocalScene*>(scene)) {
            this->updateDebugSDFCubesForChunk(nid, nd, localScene->transparentOctree);
        }
        OctreeNodeData nodeCopy = nd;
        this->processNodeLayer(*scene, LAYER_TRANSPARENT, nid, nodeCopy,
            [this](Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
                std::lock_guard<std::mutex> lock(pendingMeshMutex);
                pendingMeshQueue.push_back({layer, nid, nd, geom, nd.node->version});
            }
        );
    
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
        }
        removeDebugCubeForNode(nid);
        removeDebugSDFCubesForNode(nid);
    };


    return LiquidSpaceChangeHandler(liquidNodeEventCallback, liquidNodeEraseCallback);
}


// Ensure mesh exists and is up-to-date for a node: insert or replace when needed
void SceneRenderer::updateMeshForNode(VulkanApp* app, Layer layer, NodeID nid, const OctreeNodeData &nd, const Geometry &geom, bool attemptUpload, uint sourceVersion, bool* hadRemovals) {
    std::lock_guard<std::recursive_mutex> lock(chunksMutex);
    IndirectRenderer &renderer = layer == LAYER_OPAQUE ? solidRenderer->getIndirectRenderer() : waterRenderer->getIndirectRenderer();
    auto &cur = layer == LAYER_OPAQUE ? solidChunks : transparentChunks;
    auto it = cur.find(nid);
    uint effectiveVersion = sourceVersion != 0 ? sourceVersion : nd.node->version;
        if (it != cur.end()) {
        if (it->second.version >= effectiveVersion) {
            //printf("[SceneRenderer::updateMeshForNode] Node %llu already up-to-date (version %u >= %u)\n", (unsigned long long)nid, it->second.version, effectiveVersion);
            return; // already up-to-date
        }
        if (it->second.meshId != UINT32_MAX) {
            //printf("[SceneRenderer::updateMeshForNode] Removing old mesh for node %llu (meshId=%u)\n", (unsigned long long)nid, it->second.meshId);
            renderer.removeMesh(it->second.meshId);
            // Do NOT call eraseMeshFromGPU here — it maps & zeroes the GPU
            // indirect buffer while the previous in-flight frame may still be
            // reading that slot.  The stale indirect command is harmless: it
            // points to vertex/index data that hasn't been overwritten
            // (append-only), so the old mesh renders correctly until the next
            // rebuild() compacts the buffers.
            if (hadRemovals) *hadRemovals = true;
        }
    }
    uint32_t meshId = renderer.addMesh(geom);
    Model3DVersion mv{meshId, effectiveVersion};
    cur[nid] = mv;
    // Upload mesh into the renderer (may mark renderer dirty). The renderer
    // rebuild will perform GPU uploads; we keep that responsibility centralized
    // so uploads may be performed asynchronously inside the renderer.
    if (attemptUpload) {
        renderer.uploadMesh(app, meshId);
    }
    // Rebuild is deferred to the end of processPendingMeshes() for efficiency.
    // When called from brush rebuild paths, the caller invokes rebuild() explicitly.

    // Generate vegetation instances for this node using the compute shader.
    // The generated instance buffer is published only after the compute fence
    // signals, and graphics waits on the compute semaphore before drawing it.
    if (layer == LAYER_OPAQUE && vegetationRenderer) {
        if (geom.indices.size() >= 3 && !geom.vertices.empty()) {
            try {
                constexpr int kGrassBrushIndex = 3; // See LandBrush::grass
                // Instances per world-space unit² of triangle area.
                constexpr float kVegetationDensity = 0.01f;

                // Create tightly-packed position buffer (vec3[]) for the compute shader
                std::vector<glm::vec3> positions;
                positions.reserve(geom.vertices.size());
                for (const auto &v : geom.vertices) positions.push_back(v.position);

                // Build area-weighted virtual slots using unbiased stochastic rounding.
                // expected = area * density (instances per world-space unit area)
                // count = floor(expected) + Bernoulli(frac(expected))
                // This preserves area-proportional density without bias.
                std::vector<uint32_t> grassIndices;
                grassIndices.reserve(geom.indices.size());
                const uint32_t chunkSeed = static_cast<uint32_t>(nid ^ (nid >> 32)) ^ 0x9e3779b9u;
                std::mt19937 samplingRng(chunkSeed);
                std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);
                for (size_t i = 0; i + 2 < geom.indices.size(); i += 3) {
                    const uint32_t i0 = geom.indices[i + 0];
                    const uint32_t i1 = geom.indices[i + 1];
                    const uint32_t i2 = geom.indices[i + 2];
                    if (i0 >= geom.vertices.size() || i1 >= geom.vertices.size() || i2 >= geom.vertices.size()) continue;
                    const bool hasGrass =
                        geom.vertices[i0].brushIndex == kGrassBrushIndex ||
                        geom.vertices[i1].brushIndex == kGrassBrushIndex ||
                        geom.vertices[i2].brushIndex == kGrassBrushIndex;
                    if (!hasGrass) continue;
                    const glm::vec3& v0 = geom.vertices[i0].position;
                    const glm::vec3& v1 = geom.vertices[i1].position;
                    const glm::vec3& v2 = geom.vertices[i2].position;
                    // Skip steep / downward-facing triangles (same criterion as compute shader).
                    // This avoids allocating output slots that the compute shader would discard,
                    // preventing garbage uninitialized memory from reaching the draw call.
                    const glm::vec3 faceNormal = glm::cross(v1 - v0, v2 - v0);
                    if (glm::abs(faceNormal.y) <= 0.5f * glm::length(faceNormal)) continue;
                    const float area = 0.5f * glm::length(faceNormal);
                    const float expectedInstances = std::max(0.0f, area * kVegetationDensity);
                    uint32_t slotCount = static_cast<uint32_t>(std::floor(expectedInstances));
                    const float fractional = expectedInstances - static_cast<float>(slotCount);
                    if (unitDist(samplingRng) < fractional) {
                        ++slotCount;
                    }
                    for (uint32_t s = 0; s < slotCount; ++s) {
                        grassIndices.push_back(i0);
                        grassIndices.push_back(i1);
                        grassIndices.push_back(i2);
                    }
                }

                // Shuffle virtual triangle slots per chunk so reducing indirect instanceCount
                // keeps a random spatial subset instead of always dropping the tail.
                if (grassIndices.size() >= 6) {
                    std::mt19937 shuffleRng(chunkSeed ^ 0x85ebca6bu);
                    const size_t triangleCount = grassIndices.size() / 3;
                    for (size_t slot = triangleCount - 1; slot > 0; --slot) {
                        std::uniform_int_distribution<size_t> dist(0, slot);
                        const size_t other = dist(shuffleRng);
                        if (other == slot) continue;
                        for (size_t component = 0; component < 3; ++component) {
                            std::swap(grassIndices[slot * 3 + component], grassIndices[other * 3 + component]);
                        }
                    }
                }

                // Each virtual triangle slot produces exactly 1 instance.
                uint32_t instancesPerTriangle = 1u;
                uint32_t seed = static_cast<uint32_t>(nid & 0xffffffffull);
                glm::vec3 chunkCenter(0.0f);
                for (const auto& position : positions) {
                    chunkCenter += position;
                }
                if (!positions.empty()) {
                    chunkCenter /= static_cast<float>(positions.size());
                }
                if (grassIndices.size() < 3) {
                    // No grass triangles in this chunk; ensure old chunk vegetation is cleared.
                    if (std::getenv("VULKAN_DISABLE_VEGETATION")) {
                        std::cerr << "[SceneRenderer] VULKAN_DISABLE_VEGETATION set; skipping vegetation clear for node " << (unsigned long long)nid << std::endl;
                        return;
                    }
                    vegetationRenderer->generateChunkInstances(nid, Buffer{}, 0, Buffer{}, 0, chunkCenter, instancesPerTriangle, app, seed);
                    return;
                }

                // CPU-side instance generation — avoids RADV GPUVM faults where
                // the Texture Cache/Pipe cannot read storage buffers on iGPUs.
                vegetationRenderer->generateChunkInstancesCPU(nid, positions, grassIndices,
                    chunkCenter, instancesPerTriangle, app, seed);
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

void SceneRenderer::updateDebugSDFCubesForChunk(NodeID nid, const OctreeNodeData& nd, const Octree& tree) {
    if (!debugSDFRenderer || !nd.node || !tree.allocator) return;

    std::vector<DebugSDFRenderer::CubeSDF> cubes;
    collectLeafSDFCubes(nd.node, nd.cube, *tree.allocator, cubes);

    std::lock_guard<std::recursive_mutex> lock(chunksMutex);
    if (cubes.empty()) {
        nodeDebugSDFCubes.erase(nid);
    } else {
        nodeDebugSDFCubes[nid] = std::move(cubes);
    }
}

void SceneRenderer::removeDebugSDFCubesForNode(NodeID id) {
    std::lock_guard<std::recursive_mutex> lock(chunksMutex);
    nodeDebugSDFCubes.erase(id);
}

void SceneRenderer::clearDebugSDFCubes() {
    std::lock_guard<std::recursive_mutex> lock(chunksMutex);
    nodeDebugSDFCubes.clear();
}

std::vector<DebugSDFRenderer::CubeSDF> SceneRenderer::getDebugSDFCubes() {
    std::lock_guard<std::recursive_mutex> lock(chunksMutex);
    std::vector<DebugSDFRenderer::CubeSDF> out;
    size_t total = 0;
    for (const auto& entry : nodeDebugSDFCubes) {
        total += entry.second.size();
    }
    out.reserve(total);
    for (const auto& entry : nodeDebugSDFCubes) {
        out.insert(out.end(), entry.second.begin(), entry.second.end());
    }
    return out;
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
        OctreeNodeData nodeCopy = nd;
        this->processNodeLayer(*scene, LAYER_OPAQUE, nid, nodeCopy,
            [this, app](Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
                updateBrushMeshForNode(this, app, layer, nid, nd, geom, this->brushSolidChunks);
            }
        );
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
        OctreeNodeData nodeCopy = nd;
        this->processNodeLayer(*scene, LAYER_TRANSPARENT, nid, nodeCopy,
            [this, app](Layer layer, NodeID nid, const OctreeNodeData& nd, const Geometry& geom) {
                updateBrushMeshForNode(this, app, layer, nid, nd, geom, this->brushTransparentChunks);
            }
        );
        
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
