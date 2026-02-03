// Standard library includes first
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <stdexcept>
#include <mutex>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "Uniforms.hpp"
#include "vulkan/VulkanApp.hpp"
#include "vulkan/SceneRenderer.hpp"
#include "utils/LocalScene.hpp"
#include "widgets/SettingsWidget.hpp"
#include "widgets/SkyWidget.hpp"
#include "widgets/SkySettings.hpp"
#include "widgets/WaterWidget.hpp"
#include "widgets/RenderPassDebugWidget.hpp"
#include "widgets/BillboardWidget.hpp"
#include "widgets/BillboardCreator.hpp"
#include "widgets/AnimatedTextureWidget.hpp"
#include "widgets/TextureViewerWidget.hpp"
#include "widgets/CameraWidget.hpp"
#include "widgets/DebugWidget.hpp"
#include "widgets/ShadowMapWidget.hpp"
#include "widgets/LightWidget.hpp"
#include "widgets/VulkanObjectsWidget.hpp"
#include "widgets/VegetationAtlasEditor.hpp"
#include "widgets/OctreeExplorerWidget.hpp"
#include "utils/MainSceneLoader.hpp"
#include "widgets/Settings.hpp"
// ...existing includes...
#include "widgets/BillboardWidgetManager.hpp"
#include "widgets/WidgetManager.hpp"
#include "math/Camera.hpp"
#include "math/Light.hpp"
#include "events/EventManager.hpp"
#include "events/KeyboardPublisher.hpp"
#include "events/CloseWindowEvent.hpp"
#include "events/ToggleFullscreenEvent.hpp"
#include "vulkan/TextureArrayManager.hpp"
#include "vulkan/MaterialManager.hpp"
#include "vulkan/BillboardManager.hpp"
#include "vulkan/AtlasManager.hpp"
#include "vulkan/TextureMixer.hpp"
#include "vulkan/ShadowParams.hpp"

class MyApp : public VulkanApp, public IEventHandler {
public:
    Settings settings;
    std::unique_ptr<SceneRenderer> sceneRenderer;
    LocalScene * mainScene;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    static constexpr uint32_t QUERY_COUNT = 12;
    float timestampPeriod = 0.0f;
    bool profilingEnabled = true;
    float profileShadowCull = 0.0f;
    float profileShadowDraw = 0.0f;
    float profileMainCull = 0.0f;
    float profileDepthPrepass = 0.0f;
    float profileMainDraw = 0.0f;
    float profileSky = 0.0f;
    float profileImGui = 0.0f;
    float profileWater = 0.0f;
    float profileCpuUpdate = 0.0f;
    float profileCpuRecord = 0.0f;
    VkDescriptorSet shadowPassDescriptorSet = VK_NULL_HANDLE;
    UniformObject uboStatic = {};
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    std::shared_ptr<SettingsWidget> settingsWidget;
    std::shared_ptr<SkyWidget> skyWidget;
    std::shared_ptr<WaterWidget> waterWidget;
    std::shared_ptr<RenderPassDebugWidget> renderPassDebugWidget;
    std::shared_ptr<BillboardWidget> billboardWidget;
    std::shared_ptr<BillboardCreator> billboardCreator;
    std::unique_ptr<BillboardWidgetManager> billboardWidgetManager;
    std::shared_ptr<AnimatedTextureWidget> animatedTextureWidget;
    std::shared_ptr<TextureViewer> textureViewer;
    std::shared_ptr<CameraWidget> cameraWidget;
    std::shared_ptr<DebugWidget> debugWidget;
    std::shared_ptr<ShadowMapWidget> shadowWidget;
    std::shared_ptr<LightWidget> lightWidget;
    std::shared_ptr<VulkanObjectsWidget> vulkanObjectsWidget;
    std::shared_ptr<VegetationAtlasEditor> vegetationAtlasEditor;
    std::shared_ptr<OctreeExplorerWidget> octreeExplorerWidget;
    WidgetManager widgetManager;
    uint32_t loadedTextureLayers = 0;

    // Billboard editing / vegetation resources
    BillboardManager billboardManager;
    AtlasManager vegetationAtlasManager;
    TextureArrayManager vegetationTextureArrayManager;

    // Texture editing / UI helpers
    std::shared_ptr<TextureMixer> textureMixer;
    std::vector<MixerParameters> mixerParams;
    std::vector<MaterialProperties> materials;
    ShadowParams shadowParams;
    size_t cubeCount = 0;

    // Camera and input
    Camera camera = Camera(glm::vec3(3456, 915, 2000), Math::eulerToQuat(0, 0, 0));
    Light light = Light(glm::vec3(0.0f, -1.0f, 0.0f));
    EventManager eventManager;
    KeyboardPublisher keyboardPublisher;
    bool sceneLoading = false;

    void setup() override {

        // Ensure SceneRenderer exists and initialize it (SceneRenderer now owns SkySettings)
        if (!sceneRenderer) {
            sceneRenderer = std::make_unique<SceneRenderer>(this);
            sceneRenderer->init(this, getMainDescriptorSet());
            sceneRenderer->createPipelines();
            printf("[MyApp::setup] Created and initialized SceneRenderer\n");
        } else {
            if (sceneRenderer->solidRenderer) {
                sceneRenderer->solidRenderer->getIndirectRenderer().rebuild(this);
                printf("[MyApp::setup] Rebuilt opaque IndirectRenderer\n");
            }
            if (sceneRenderer->waterRenderer) {
                sceneRenderer->waterRenderer->getIndirectRenderer().rebuild(this);
                printf("[MyApp::setup] Rebuilt water IndirectRenderer\n");
            }
            // If SceneRenderer already existed ensure the sky UBO is bound into the main descriptor set
            if (sceneRenderer && sceneRenderer->skyRenderer) {
                sceneRenderer->skyRenderer->initSky(sceneRenderer->getSkySettings(), getMainDescriptorSet());
                printf("[MyApp::setup] Initialized SkyRenderer with SceneRenderer-owned SkySettings and main descriptor set\n");
            }
        }

        // Load legacy texture triples (albedo, normal, height) into the texture array manager
        if (sceneRenderer) {
            uint32_t loadedTextureLayers_local = 0;
            struct TextureTriple { const char* albedo; const char* normal; const char* bump; };
            const std::vector<TextureTriple> textureTriples = {
                { "textures/bricks_color.jpg", "textures/bricks_normal.jpg", "textures/bricks_bump.jpg" },
                { "textures/dirt_color.jpg", "textures/dirt_normal.jpg", "textures/dirt_bump.jpg" },
                { "textures/forest_color.jpg", "textures/forest_normal.jpg", "textures/forest_bump.jpg" },
                { "textures/grass_color.jpg", "textures/grass_normal.jpg", "textures/grass_bump.jpg" },
                { "textures/lava_color.jpg", "textures/lava_normal.jpg", "textures/lava_bump.jpg" },
                { "textures/metal_color.jpg", "textures/metal_normal.jpg", "textures/metal_bump.jpg" },
                { "textures/pixel_color.jpg", "textures/pixel_normal.jpg", "textures/pixel_bump.jpg" },
                { "textures/rock_color.jpg", "textures/rock_normal.jpg", "textures/rock_bump.jpg" },
                { "textures/sand_color.jpg", "textures/sand_normal.jpg", "textures/sand_bump.jpg" },
                { "textures/snow_color.jpg", "textures/snow_normal.jpg", "textures/snow_bump.jpg" },
                { "textures/soft_sand_color.jpg", "textures/soft_sand_normal.jpg", "textures/soft_sand_bump.jpg" }
            };

            sceneRenderer->textureArrayManager.currentLayer = 0;
            for (const auto& triple : textureTriples) {
                if (sceneRenderer->textureArrayManager.currentLayer >= sceneRenderer->textureArrayManager.layerAmount) {
                    fprintf(stderr, "[TextureLoad] Reached texture array capacity (%u layers)\n", sceneRenderer->textureArrayManager.layerAmount);
                    break;
                }
                try {
                    sceneRenderer->textureArrayManager.load(triple.albedo, triple.normal, triple.bump);
                    ++loadedTextureLayers_local;
                } catch (const std::exception& e) {
                    fprintf(stderr, "[TextureLoad] Failed to load %s: %s\n", triple.albedo ? triple.albedo : "(null)", e.what());
                }
            }
            // Record into member so UI can display counts
            loadedTextureLayers = loadedTextureLayers_local;

            for (uint32_t i = 0; i < loadedTextureLayers; ++i) {
                sceneRenderer->textureArrayManager.getImTexture(i, 0);
                sceneRenderer->textureArrayManager.getImTexture(i, 1);
                sceneRenderer->textureArrayManager.getImTexture(i, 2);
            }
        }

        // Restore additional widgets and editors
        uint32_t mixWidth = sceneRenderer ? (sceneRenderer->textureArrayManager.width ? sceneRenderer->textureArrayManager.width : 512u) : 512u;
        uint32_t mixHeight = sceneRenderer ? (sceneRenderer->textureArrayManager.height ? sceneRenderer->textureArrayManager.height : 512u) : 512u;
        textureMixer = std::make_shared<TextureMixer>();
        textureMixer->init(this, mixWidth, mixHeight, sceneRenderer ? &sceneRenderer->textureArrayManager : nullptr);
        mixerParams.clear();
        uint32_t layerCount = sceneRenderer ? sceneRenderer->textureArrayManager.layerAmount : 1u;
        uint32_t editableLayer = (loadedTextureLayers < layerCount) ? loadedTextureLayers : 0u;
        uint32_t availableLayers = std::max(layerCount, std::max(loadedTextureLayers, 1u));
        MixerParameters defaultMixer{};
        defaultMixer.targetLayer = editableLayer;
        defaultMixer.primaryTextureIdx = (availableLayers > 1) ? 1u : 0u;
        defaultMixer.secondaryTextureIdx = (availableLayers > 2) ? 2u : defaultMixer.primaryTextureIdx;
        mixerParams.push_back(defaultMixer);
        textureMixer->generateInitialTextures(mixerParams);
        textureMixer->setEditableLayer(defaultMixer.targetLayer);
        // Prime ImGui descriptors so the texture viewer shows immediately
        if (sceneRenderer) {
            sceneRenderer->textureArrayManager.setLayerInitialized(defaultMixer.targetLayer, true);
            sceneRenderer->textureArrayManager.getImTexture(defaultMixer.targetLayer, 0);
            sceneRenderer->textureArrayManager.getImTexture(defaultMixer.targetLayer, 1);
            sceneRenderer->textureArrayManager.getImTexture(defaultMixer.targetLayer, 2);
        }

        animatedTextureWidget = std::make_shared<AnimatedTextureWidget>(textureMixer, mixerParams, "Editable Textures");

        size_t materialCount = std::max<size_t>(static_cast<size_t>(loadedTextureLayers), static_cast<size_t>(defaultMixer.targetLayer + 1));
        if (materialCount == 0) {
            materialCount = layerCount ? layerCount : 1u;
        }
        materials.assign(materialCount, MaterialProperties{});
        textureViewer = std::make_shared<TextureViewer>();
        textureViewer->init(sceneRenderer ? &sceneRenderer->textureArrayManager : nullptr, &materials);
        textureViewer->setOnMaterialChanged([](size_t) {});
        skyWidget = std::make_shared<SkyWidget>(sceneRenderer->getSkySettings());
        // Create settings widget (was missing previously)
        settingsWidget = std::make_shared<SettingsWidget>(settings);
        waterWidget = std::make_shared<WaterWidget>(sceneRenderer ? sceneRenderer->waterRenderer.get() : nullptr);
        renderPassDebugWidget = std::make_shared<RenderPassDebugWidget>(this, sceneRenderer ? sceneRenderer->waterRenderer.get() : nullptr, sceneRenderer ? sceneRenderer->solidRenderer.get() : nullptr);
        billboardWidget = std::make_shared<BillboardWidget>();
        billboardWidgetManager = std::make_unique<BillboardWidgetManager>(billboardWidget, sceneRenderer ? sceneRenderer->vegetationRenderer.get() : nullptr, nullptr);
        billboardCreator = std::make_shared<BillboardCreator>(&billboardManager, &vegetationAtlasManager, &vegetationTextureArrayManager);

        cameraWidget = std::make_shared<CameraWidget>(&camera);
        debugWidget = std::make_shared<DebugWidget>(&materials, &camera, &cubeCount);
        shadowWidget = std::make_shared<ShadowMapWidget>(sceneRenderer ? sceneRenderer->shadowMapper.get() : nullptr, &shadowParams);
        lightWidget = std::make_shared<LightWidget>(&light);
        vulkanObjectsWidget = std::make_shared<VulkanObjectsWidget>(this);

        vegetationTextureArrayManager.allocate(3, 512, 512);
        vegetationTextureArrayManager.initialize(this);
        vegetationAtlasEditor = std::make_shared<VegetationAtlasEditor>(&vegetationTextureArrayManager, &vegetationAtlasManager);

        auto sr = sceneRenderer.get();

        // Initialize and load the main scene so rendering has valid scene data
        SolidSpaceChangeHandler solidHandler = sceneRenderer->makeSolidSpaceChangeHandler();
        LiquidSpaceChangeHandler liquidHandler = sceneRenderer->makeLiquidSpaceChangeHandler();
        
        mainScene = new LocalScene(solidHandler, liquidHandler);
        MainSceneLoader loader = MainSceneLoader();
        mainScene->loadScene(loader);
    
        sceneRenderer->processPendingNodeChanges(mainScene);
        // Create octree explorer widget bound to loaded scene
        octreeExplorerWidget = std::make_shared<OctreeExplorerWidget>(mainScene);

        widgetManager.addWidget(animatedTextureWidget);
        widgetManager.addWidget(textureViewer);
        widgetManager.addWidget(cameraWidget);
        widgetManager.addWidget(debugWidget);
        widgetManager.addWidget(shadowWidget);
        widgetManager.addWidget(settingsWidget);
        widgetManager.addWidget(lightWidget);
        widgetManager.addWidget(skyWidget);
        widgetManager.addWidget(waterWidget);
        widgetManager.addWidget(renderPassDebugWidget);
        widgetManager.addWidget(vulkanObjectsWidget);
        widgetManager.addWidget(vegetationAtlasEditor);
        widgetManager.addWidget(billboardWidget);
        widgetManager.addWidget(billboardCreator);
        widgetManager.addWidget(octreeExplorerWidget);

        
        // Subscribe event handlers
        eventManager.subscribe(&camera);  // Camera handles translate/rotate events
        eventManager.subscribe(this);     // MyApp handles close/fullscreen events
        
        // Set up camera projection matrix
        float aspectRatio = static_cast<float>(getWidth()) / static_cast<float>(getHeight());
        glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspectRatio, 0.1f, 10000.0f);
        proj[1][1] *= -1; // Vulkan Y-flip
        camera.setProjection(proj);
        shadowParams.update(camera.getPosition(), light);
        
        // Position camera to view the terrain
        printf("[Camera Setup] Final Position: (%.1f, %.1f, %.1f)\n", camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
        printf("[Camera Setup] Forward: (%.3f, %.3f, %.3f)\n", camera.getForward().x, camera.getForward().y, camera.getForward().z);
    }

    void update(float deltaTime) override {
        // Poll keyboard input and publish events
        keyboardPublisher.update(getWindow(), &eventManager, camera, deltaTime, false);
        eventManager.processQueued();

        shadowParams.update(camera.getPosition(), light);
        
        // Advance water animation time (owned by WaterRenderer) when enabled
        const bool waterEnabled = settings.waterEnabled;
        if (waterEnabled && sceneRenderer && sceneRenderer->waterRenderer) {
            sceneRenderer->waterRenderer->advanceTime(deltaTime);
        }
        // Process any pending mesh updates from scene change handlers
        if (sceneRenderer) sceneRenderer->processPendingNodeChanges(mainScene);
    }

    void preRenderPass(VkCommandBuffer &commandBuffer) override {
        // Shadow pass (uses separate command buffer internally)
        if (sceneRenderer) {
            sceneRenderer->shadowPass(commandBuffer, queryPool, shadowPassDescriptorSet, uboStatic, true, false);
        } else {
            fprintf(stderr, "[MyApp::preRenderPass] sceneRenderer is null, skipping shadow pass\n");
        }

        // Build per-frame UBO
        glm::mat4 viewProj = camera.getViewProjectionMatrix();
        uboStatic.viewProjection = viewProj;
        uboStatic.viewPos = glm::vec4(camera.getPosition(), 1.0f);
        uboStatic.lightDir = glm::vec4(light.getDirection(), 0.0f);
        uboStatic.lightColor = glm::vec4(1.0f, 1.0f, 0.9f, 1.0f);
        uboStatic.lightSpaceMatrix = shadowParams.lightSpaceMatrix;
        // Encode debug/triplanar/tess parameters into the shared UBO
        uboStatic.debugParams = glm::vec4(static_cast<float>(settings.debugMode), 0.0f, 0.0f, 0.0f);
        uboStatic.triplanarSettings = glm::vec4(settings.triplanarThreshold, settings.triplanarExponent, 0.0f, 0.0f);
        uboStatic.tessParams = glm::vec4(
            settings.tessMinDistance,
            settings.tessMaxDistance,
            settings.tessMinLevel,
            settings.tessMaxLevel
        );
        // passParams: x = isShadowPass, y = tessEnabled (also gates displacement in TCS/TE)
        uboStatic.passParams = glm::vec4(0.0f, settings.tessellationEnabled ? 1.0f : 0.0f, 0.0f, 0.0f);

        // Upload UBO to GPU
        if (sceneRenderer) {
            void* data;
            vkMapMemory(getDevice(), sceneRenderer->mainUniformBuffer.memory, 0, sizeof(UniformObject), 0, &data);
            memcpy(data, &uboStatic, sizeof(UniformObject));
            vkUnmapMemory(getDevice(), sceneRenderer->mainUniformBuffer.memory);
        } else {
            fprintf(stderr, "[MyApp::preRenderPass] sceneRenderer is null, skipping UBO upload\n");
        }

        // Run GPU frustum culling for opaque geometry
        if (sceneRenderer && sceneRenderer->solidRenderer) {
            sceneRenderer->solidRenderer->getIndirectRenderer().prepareCull(commandBuffer, viewProj);
        }

            const bool waterEnabled = settings.waterEnabled;
            const bool vegetationEnabled = settings.vegetationEnabled;

            // Render sky + solids/vegetation into the solid offscreen framebuffer (one per frame)
            if (sceneRenderer && sceneRenderer->waterRenderer && sceneRenderer->solidRenderer) {
            uint32_t frameIdx = getCurrentFrame();
            VkClearValue colorClear{};
            // Clear solid offscreen color to transparent so composite starts from empty scene
            colorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
            VkClearValue depthClear{};
            depthClear.depthStencil = {1.0f, 0};
                sceneRenderer->solidRenderer->beginPass(commandBuffer, frameIdx, colorClear, depthClear);

            VkRenderPassBeginInfo unusedRpInfo{};
            sceneRenderer->skyPass(commandBuffer, getMainDescriptorSet(), sceneRenderer->mainUniformBuffer, uboStatic, viewProj);
            sceneRenderer->mainPass(commandBuffer, unusedRpInfo, frameIdx, waterEnabled, vegetationEnabled, getMainDescriptorSet(), sceneRenderer->mainUniformBuffer, settings.wireframeMode, profilingEnabled, queryPool,
                viewProj, uboStatic, sceneRenderer->waterRenderer->getParams(), sceneRenderer->waterRenderer->getTime(), true, false, true, 0, 0.0f, 0.0f);

            // Render debug cubes for expanded octree nodes
            if (octreeExplorerWidget && octreeExplorerWidget->isVisible() && sceneRenderer->debugCubeRenderer) {
                // Convert from OctreeExplorerWidget::CubeWithColor to DebugCubeRenderer::CubeWithColor
                const auto& widgetCubes = octreeExplorerWidget->getExpandedCubes();
                std::vector<DebugCubeRenderer::CubeWithColor> debugCubes;
                debugCubes.reserve(widgetCubes.size());
                for (const auto& wc : widgetCubes) {
                    // Convert BoundingCube -> BoundingBox for renderer
                    debugCubes.push_back({BoundingBox(wc.cube.getMin(), wc.cube.getMax()), wc.color});
                }
                sceneRenderer->debugCubeRenderer->setCubes(debugCubes);
                sceneRenderer->debugCubeRenderer->render(commandBuffer, getMainDescriptorSet());
            }

            // Render per-mesh bounding boxes (if enabled in settings)
            if (settings.showBoundingBoxes && sceneRenderer && sceneRenderer->boundingBoxRenderer) {
                std::vector<DebugCubeRenderer::CubeWithColor> boxes;
                auto gatherBoxesFrom = [&](const IndirectRenderer &ir, const glm::vec3 &color){
                    auto infos = ir.getActiveMeshInfos();
                    boxes.reserve(boxes.size() + infos.size());
                    for (const auto &mi : infos) {
                        // Object-space AABB
                        glm::vec3 omin = glm::vec3(mi.boundsMin);
                        glm::vec3 omax = glm::vec3(mi.boundsMax);
                        // Transform 8 corners by model matrix and recompute world AABB
                        glm::vec3 worldMin(FLT_MAX);
                        glm::vec3 worldMax(-FLT_MAX);
                        for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 2; ++c) {
                            glm::vec3 corner = glm::vec3(a ? omax.x : omin.x, b ? omax.y : omin.y, c ? omax.z : omin.z);
                            glm::vec4 wc = mi.model * glm::vec4(corner, 1.0f);
                            worldMin = glm::min(worldMin, glm::vec3(wc));
                            worldMax = glm::max(worldMax, glm::vec3(wc));
                        }
                        boxes.push_back({BoundingBox(worldMin, worldMax), color});
                    }
                };

                if (sceneRenderer->solidRenderer) {
                    gatherBoxesFrom(sceneRenderer->solidRenderer->getIndirectRenderer(), glm::vec3(0.0f, 1.0f, 0.0f));
                }
                if (sceneRenderer->waterRenderer) {
                    gatherBoxesFrom(sceneRenderer->waterRenderer->getIndirectRenderer(), glm::vec3(0.0f, 0.5f, 1.0f));
                }

                if (!boxes.empty()) {
                    sceneRenderer->boundingBoxRenderer->setCubes(boxes);
                    sceneRenderer->boundingBoxRenderer->render(commandBuffer, getMainDescriptorSet());
                }
            }

                sceneRenderer->solidRenderer->endPass(commandBuffer);

            // Run water geometry pass offscreen and bind scene textures for post-process
            if (waterEnabled) {
                sceneRenderer->waterPass(commandBuffer, unusedRpInfo, frameIdx, getMainDescriptorSet(), profilingEnabled, queryPool,
                    sceneRenderer->waterRenderer->getParams(), sceneRenderer->waterRenderer->getTime());
            }
        }
    }

    void renderImGui() override {
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Exit")) requestClose();
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Show Demo", NULL, &imguiShowDemo);
                ImGui::MenuItem("Show Profiling", NULL, &profilingEnabled);
                if (ImGui::MenuItem("Fullscreen", "F11", isFullscreen)) {
                    toggleFullscreen();
                }
                ImGui::EndMenu();
            }
            // Widget menu
            widgetManager.renderMenu();
            ImGui::EndMainMenuBar();

            // Small top-left overlay under the main menu bar showing FPS and visible count
            {
                ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
                ImGui::SetNextWindowBgAlpha(0.35f);
                float padding = 10.0f;
                float y = ImGui::GetFrameHeight() + 6.0f; // position just under the main menu bar
                ImGui::SetNextWindowPos(ImVec2(padding, y), ImGuiCond_Always);

                ImGui::Begin("StatsOverlay", nullptr, flags);

                // Statistics: loaded/visible counts
                ImGui::Text("Textures Loaded (CPU): %u", loadedTextureLayers);

                // Opaque (solid)
                size_t opaqueLoaded = sceneRenderer && sceneRenderer->solidRenderer ? sceneRenderer->solidRenderer->getIndirectRenderer().getMeshCount() : 0;
                uint32_t opaqueVisible = sceneRenderer && sceneRenderer->solidRenderer ? sceneRenderer->solidRenderer->getIndirectRenderer().readVisibleCount(this) : 0;
                ImGui::Text("Opaque - Loaded (GPU): %zu  Visible (GPU cull): %u", opaqueLoaded, opaqueVisible);
                size_t opaqueTracked = sceneRenderer ? sceneRenderer->getRegisteredModelCount() : 0;
                ImGui::Text("Opaque Models Tracked: %zu", opaqueTracked);

                // Transparent / water
                size_t transparentLoaded = sceneRenderer && sceneRenderer->waterRenderer ? sceneRenderer->waterRenderer->getIndirectRenderer().getMeshCount() : 0;
                uint32_t transparentVisible = sceneRenderer && sceneRenderer->waterRenderer ? sceneRenderer->waterRenderer->getIndirectRenderer().readVisibleCount(this) : 0;
                ImGui::Text("Transparent - Loaded (GPU): %zu  Visible (GPU cull): %u", transparentLoaded, transparentVisible);
                size_t transparentTracked = sceneRenderer ? sceneRenderer->getTransparentModelCount() : 0;
                ImGui::Text("Transparent Models Tracked: %zu", transparentTracked);

                // Vegetation
                size_t vegChunks = sceneRenderer && sceneRenderer->vegetationRenderer ? sceneRenderer->vegetationRenderer->getChunkCount() : 0;
                size_t vegInstances = sceneRenderer && sceneRenderer->vegetationRenderer ? sceneRenderer->vegetationRenderer->getInstanceTotal() : 0;
                ImGui::Text("Vegetation Chunks: %zu", vegChunks);
                ImGui::Text("Vegetation Instances: %zu", vegInstances);

                // Pending node change queues
                size_t pendingCreated = sceneRenderer ? sceneRenderer->getPendingCreatedCount() : 0;
                size_t pendingUpdated = sceneRenderer ? sceneRenderer->getPendingUpdatedCount() : 0;
                size_t pendingErased = sceneRenderer ? sceneRenderer->getPendingErasedCount() : 0;
                ImGui::Text("Pending Created/Updated/Erased: %zu / %zu / %zu", pendingCreated, pendingUpdated, pendingErased);

                if (profilingEnabled) {
                    ImGui::Separator();
                    ImGui::Text("--- GPU Timing (ms) ---");
                    float gpuTotal = profileShadowCull + profileShadowDraw + profileMainCull +
                                     profileDepthPrepass + profileMainDraw + profileSky + profileImGui;
                    ImGui::Text("Shadow Cull:   %.2f", profileShadowCull);
                    ImGui::Text("Shadow Draw:   %.2f", profileShadowDraw);
                    ImGui::Text("Main Cull:     %.2f", profileMainCull);
                    ImGui::Text("Depth Prepass: %.2f", profileDepthPrepass);
                    ImGui::Text("Main Draw:     %.2f", profileMainDraw);
                    ImGui::Text("Sky:           %.2f", profileSky);
                    ImGui::Text("ImGui:         %.2f", profileImGui);
                    ImGui::Text("GPU Total:     %.2f", gpuTotal);
                    ImGui::Separator();
                    ImGui::Text("--- CPU Timing (ms) ---");
                    ImGui::Text("Update:        %.2f", profileCpuUpdate);
                    ImGui::Text("Record:        %.2f", profileCpuRecord);
                }

                ImGui::End();
            }
        }

        if (imguiShowDemo) ImGui::ShowDemoWindow(&imguiShowDemo);

        cubeCount = sceneRenderer ? sceneRenderer->getRegisteredModelCount() : 0;

        // Render all widgets
        widgetManager.renderAll();
    }

    void draw(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo) override {
        // Only record draw commands; command buffer and render pass are already active
        if (commandBuffer == VK_NULL_HANDLE) {
            fprintf(stdout, "[MyApp::draw] Error: commandBuffer is VK_NULL_HANDLE, skipping draw.\n");
            return;
        }
        if (!sceneRenderer) {
            fprintf(stdout  , "[MyApp::draw] Error: sceneRenderer is nullptr, skipping draw.\n");
            return;
        }
        if (!mainScene) {
            fprintf(stdout  , "[MyApp::draw] Error: mainScene is nullptr, skipping draw.\n");
            return;
        }
        
        uint32_t frameIdx = getCurrentFrame();
        glm::mat4 viewProj = camera.getViewProjectionMatrix();
        glm::mat4 invViewProj = glm::inverse(viewProj);

        // Composite offscreen scene + water into the swapchain
        if (sceneRenderer && sceneRenderer->waterRenderer) {
            sceneRenderer->waterRenderer->renderWaterPostProcess(
                commandBuffer,
                renderPassInfo.framebuffer,
                renderPassInfo.renderPass,
                sceneRenderer->solidRenderer ? sceneRenderer->solidRenderer->getColorView(frameIdx) : VK_NULL_HANDLE,
                sceneRenderer->solidRenderer ? sceneRenderer->solidRenderer->getDepthView(frameIdx) : VK_NULL_HANDLE,
                sceneRenderer->waterRenderer->getParams(),
                viewProj,
                invViewProj,
                glm::vec3(uboStatic.viewPos),
                sceneRenderer->waterRenderer->getTime(),
                false);
        }
        //fprintf(stderr, "[MyApp::draw] waterPass returned. Rendering ImGui...\n");
        // ImGui rendering
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (!draw_data) {
            fprintf(stdout, "[MyApp::draw] Warning: ImGui::GetDrawData() returned nullptr, skipping ImGui rendering.\n");
        } else if (commandBuffer == VK_NULL_HANDLE) {
            fprintf(stdout, "[MyApp::draw] Error: commandBuffer is VK_NULL_HANDLE before ImGui rendering, skipping ImGui.\n");
        } else {
            //fprintf(stderr, "[MyApp::draw] ImGui::GetDrawData() valid, calling ImGui_ImplVulkan_RenderDrawData...\n");
            ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);
        }
    }

    void clean() override {
        // Cleanup scene renderer and all sub-renderers
        if (sceneRenderer) {
            sceneRenderer->cleanup();
        }
        if (textureMixer) {
            textureMixer->cleanup();
        }
    }

    void onSwapchainResized(uint32_t width, uint32_t height) override {
        if (sceneRenderer) {
            sceneRenderer->onSwapchainResized(width, height);
        }
    }

    void onEvent(const EventPtr &event) override {
        if (auto closeEvent = std::dynamic_pointer_cast<CloseWindowEvent>(event)) {
            requestClose();
        } else if (auto fullscreenEvent = std::dynamic_pointer_cast<ToggleFullscreenEvent>(event)) {
            toggleFullscreen();
        }
    }
};

int main(int argc, char** argv) {
    try {
        MyApp app;
        app.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}


