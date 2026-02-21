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
#include <thread>
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
#include "widgets/TextureMixerWidget.hpp"
#include "widgets/TextureViewerWidget.hpp"
#include "widgets/CameraWidget.hpp"
#include "widgets/DebugWidget.hpp"
#include "widgets/ShadowMapWidget.hpp"
#include "widgets/LightWidget.hpp"
#include "widgets/VulkanResourcesManagerWidget.hpp"
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
    SceneRenderer * sceneRenderer;
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
    std::shared_ptr<TextureMixerWidget> textureMixerWidget;
    std::shared_ptr<TextureViewer> textureViewer;
    std::shared_ptr<CameraWidget> cameraWidget;
    std::shared_ptr<DebugWidget> debugWidget;
    std::shared_ptr<ShadowMapWidget> shadowWidget;
    std::shared_ptr<LightWidget> lightWidget;
    std::shared_ptr<VulkanResourcesManagerWidget> vulkanResourcesManagerWidget;
    std::shared_ptr<VegetationAtlasEditor> vegetationAtlasEditor;
    std::shared_ptr<OctreeExplorerWidget> octreeExplorerWidget;
    WidgetManager widgetManager;
    uint32_t loadedTextureLayers = 0;

    // Billboard editing / vegetation resources
    BillboardManager billboardManager;
    AtlasManager vegetationAtlasManager;
    TextureArrayManager vegetationTextureArrayManager = TextureArrayManager();

    // Global texture arrays and material manager (moved from SceneRenderer)
    TextureArrayManager textureArrayManager = TextureArrayManager();
    MaterialManager materialManager = MaterialManager();

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

    // setupTextures (defined out-of-line to avoid inline/member-definition issues)
    void setupTextures() {
        uint32_t layerCount = 32;

        textureArrayManager.allocate(layerCount, 1024, 1024, this);

  
        // Use shared TextureTriple defined in TextureArrayManager.hpp
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

        // Bulk load the triples directly using TextureTriple vector already defined above
        loadedTextureLayers = textureArrayManager.loadTriples(this, textureTriples);
        // Ensure mixer descriptor sets are updated with newly loaded arrays
        
        textureMixer = std::make_shared<TextureMixer>();
        textureMixer->init(this, &textureArrayManager);
        textureMixer->attachTextureArrayManager(&textureArrayManager);
        // Record into member so UI can display counts
        mixerParams.clear();
        mixerParams.push_back(MixerParameters{loadedTextureLayers++, 3u, 8u}); // grassMixSand
        mixerParams.push_back(MixerParameters{loadedTextureLayers++, 3u, 9u}); // grassMixSnow
        mixerParams.push_back(MixerParameters{loadedTextureLayers++, 7u, 3u}); // rockMixGrass
        mixerParams.push_back(MixerParameters{loadedTextureLayers++, 7u, 9u}); // rockMixSnow
        mixerParams.push_back(MixerParameters{loadedTextureLayers++, 7u, 8u}); // rockMixSand

        for (uint32_t i = 0; i < loadedTextureLayers; ++i) {
            textureArrayManager.getImTexture(i, 0);
            textureArrayManager.getImTexture(i, 1);
            textureArrayManager.getImTexture(i, 2);
        }

        uint32_t editableLayer = (loadedTextureLayers < layerCount) ? loadedTextureLayers : 0u;
        uint32_t availableLayers = std::max(layerCount, std::max(loadedTextureLayers, 1u));



        // Trigger initial generation for configured mixers so UI previews show meaningful results
        // (Previously this was deferred to the user pressing "Generate" in the UI)
        fprintf(stderr, "[TextureMixer] Running initial generation for configured mixers...\n");
        textureMixer->setEditableLayer(editableLayer);
        // Prime ImGui descriptors so the texture viewer shows immediately
        textureArrayManager.setLayerInitialized(editableLayer, true);
        textureArrayManager.getImTexture(editableLayer, 0);
        textureArrayManager.getImTexture(editableLayer, 1);
        textureArrayManager.getImTexture(editableLayer, 2);

        // Generate textures for all configured mixer entries (async submissions tracked by TextureMixer)
        textureMixer->generateInitialTextures(mixerParams);

        textureMixerWidget = std::make_shared<TextureMixerWidget>(textureMixer, mixerParams, "Texture Mixer");

        size_t materialCount = std::max<size_t>(static_cast<size_t>(loadedTextureLayers), static_cast<size_t>(loadedTextureLayers + 1));
        if (materialCount == 0) {
            materialCount = layerCount ? layerCount : 1u;
        }
        materials.assign(materialCount, MaterialProperties{});
        // Allocate GPU-side material storage via MaterialManager
        materialManager.allocate(materialCount, this);
        for (size_t i = 0; i < materialCount; ++i) materialManager.update(i, materials[i], this);
    }

    void setup() override {
        std::thread([this]() {
            setupVegetationTextures();
        }).detach();

        std::thread([this]() {
            setupTextures();
        }).detach();

        printf("[MyApp::setup] Created and initialized SceneRenderer\n");
        sceneRenderer = new SceneRenderer();
        sceneRenderer->init(this, &textureArrayManager, &materialManager);

        textureViewer = std::make_shared<TextureViewer>();
        textureViewer->init(&textureArrayManager, &materials);
        textureViewer->setOnMaterialChanged([](size_t) {});

        skyWidget = std::make_shared<SkyWidget>(sceneRenderer->getSkySettings());
        // Create settings widget (was missing previously)
        settingsWidget = std::make_shared<SettingsWidget>(settings);
        // Inform SceneRenderer about the MaterialManager
        waterWidget = std::make_shared<WaterWidget>(sceneRenderer->waterRenderer.get());
        renderPassDebugWidget = std::make_shared<RenderPassDebugWidget>(sceneRenderer->waterRenderer.get() , sceneRenderer->solidRenderer.get());
        // Initialize widget frame info from MyApp (avoid storing VulkanApp* inside widget)
        if (renderPassDebugWidget) renderPassDebugWidget->setFrameInfo(getCurrentFrame(), getWidth(), getHeight());


        cameraWidget = std::make_shared<CameraWidget>(&camera);
        debugWidget = std::make_shared<DebugWidget>(&materials, &camera, &cubeCount);
        shadowWidget = std::make_shared<ShadowMapWidget>(sceneRenderer->shadowMapper.get(), &shadowParams);
        lightWidget = std::make_shared<LightWidget>(&light);
        vulkanResourcesManagerWidget = std::make_shared<VulkanResourcesManagerWidget>(&resources);
        vulkanResourcesManagerWidget->updateWithApp(this);
  // Create octree explorer widget bound to loaded scene

        
        widgetManager.addWidget(textureMixerWidget);
        widgetManager.addWidget(textureViewer);
        widgetManager.addWidget(cameraWidget);
        widgetManager.addWidget(debugWidget);
        widgetManager.addWidget(shadowWidget);
        widgetManager.addWidget(settingsWidget);
        widgetManager.addWidget(lightWidget);
        widgetManager.addWidget(skyWidget);
        widgetManager.addWidget(waterWidget);
        widgetManager.addWidget(renderPassDebugWidget);
        widgetManager.addWidget(vulkanResourcesManagerWidget);
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
   
//        std::thread([this]() {
            setupScene();
//        }).detach();

    }

    // Move vegetation texture setup into its own method for clarity
    void setupVegetationTextures();
    // Move scene-loading into its own method for clarity
    void setupScene();

// (setup implementation defined out-of-line below)

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
        // Flush any pending texture generation requests so they happen before command buffers are recorded
        if (textureMixer) textureMixer->flushPendingRequests(this);
        // Poll for completed async generations and process their fences
        if (textureMixer) textureMixer->pollPendingGenerations(this);

    }

    void preRenderPass(VkCommandBuffer &commandBuffer) override {
        // Shadow pass (uses separate command buffer internally)
        if (sceneRenderer) {
            sceneRenderer->shadowPass(this, commandBuffer, queryPool, shadowPassDescriptorSet, uboStatic, true, false);
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


        sceneRenderer->solidRenderer->getIndirectRenderer().prepareCull(commandBuffer, viewProj);

        const bool waterEnabled = settings.waterEnabled;
        const bool vegetationEnabled = settings.vegetationEnabled;

        // Render sky + solids/vegetation into the solid offscreen framebuffer (one per frame)
        uint32_t frameIdx = getCurrentFrame();
        VkClearValue colorClear{};
        // Clear solid offscreen color to transparent so composite starts from empty scene
        colorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        VkClearValue depthClear{};
        depthClear.depthStencil = {1.0f, 0};
            sceneRenderer->solidRenderer->beginPass(commandBuffer, frameIdx, colorClear, depthClear);

        VkRenderPassBeginInfo unusedRpInfo{};
        sceneRenderer->skyPass(this, commandBuffer, getMainDescriptorSet(), sceneRenderer->mainUniformBuffer, uboStatic, viewProj);
        sceneRenderer->mainPass(this, commandBuffer, unusedRpInfo, frameIdx, waterEnabled, vegetationEnabled, getMainDescriptorSet(), sceneRenderer->mainUniformBuffer, settings.wireframeMode, profilingEnabled, queryPool,
            viewProj, uboStatic, sceneRenderer->waterRenderer->getParams(), sceneRenderer->waterRenderer->getTime(), true, false, true, 0, 0.0f, 0.0f);

        // Render debug cubes for expanded octree nodes + node instances from change handlers
        if (settings.showDebugCubes) {
            std::vector<DebugCubeRenderer::CubeWithColor> debugCubes;
            // Add widget-expanded cubes when explorer is visible
            if (octreeExplorerWidget->isVisible()) {
                const auto& widgetCubes = octreeExplorerWidget->getExpandedCubes();
                debugCubes.reserve(widgetCubes.size());
                for (const auto& wc : widgetCubes) {
                    // Convert BoundingCube -> BoundingBox for renderer
                    debugCubes.push_back({BoundingBox(wc.cube.getMin(), wc.cube.getMax()), wc.color});
                }
            }
            // Append node cubes produced by change handlers (post-tessellation)
            auto nodeCubes = sceneRenderer->getDebugNodeCubes();
            debugCubes.reserve(debugCubes.size() + nodeCubes.size());
            for (auto &nc : nodeCubes) debugCubes.push_back(nc);

            if (!debugCubes.empty()) {
                sceneRenderer->debugCubeRenderer->setCubes(debugCubes);
                sceneRenderer->debugCubeRenderer->render(this, commandBuffer, getMainDescriptorSet());
            }
        }

        // Render per-mesh bounding boxes (if enabled in settings)
        if (settings.showBoundingBoxes && sceneRenderer && sceneRenderer->boundingBoxRenderer) {
            std::vector<DebugCubeRenderer::CubeWithColor> boxes;
            auto gatherBoxesFrom = [&](const IndirectRenderer &ir, const glm::vec3 &color){
                auto infos = ir.getActiveMeshInfos();
                boxes.reserve(boxes.size() + infos.size());
                for (const auto &mi : infos) {
                    // Object-space AABB (meshes are not transformed by per-mesh model matrices)
                    glm::vec3 worldMin = glm::vec3(mi.boundsMin);
                    glm::vec3 worldMax = glm::vec3(mi.boundsMax);
                    boxes.push_back({BoundingBox(worldMin, worldMax), color});
                }
            };

            gatherBoxesFrom(sceneRenderer->solidRenderer->getIndirectRenderer(), glm::vec3(0.0f, 1.0f, 0.0f));
            gatherBoxesFrom(sceneRenderer->waterRenderer->getIndirectRenderer(), glm::vec3(0.0f, 0.5f, 1.0f));
        

            if (!boxes.empty()) {
                sceneRenderer->boundingBoxRenderer->setCubes(boxes);
                sceneRenderer->boundingBoxRenderer->render(this, commandBuffer, getMainDescriptorSet());
            }
        }

            sceneRenderer->solidRenderer->endPass(commandBuffer);

        // Run water geometry pass offscreen and bind scene textures for post-process
        if (waterEnabled) {
            sceneRenderer->waterPass(this, commandBuffer, unusedRpInfo, frameIdx, getMainDescriptorSet(), profilingEnabled, queryPool,
                sceneRenderer->waterRenderer->getParams(), sceneRenderer->waterRenderer->getTime());
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
                size_t opaqueLoaded = sceneRenderer->solidRenderer->getIndirectRenderer().getMeshCount();
                uint32_t opaqueVisible = sceneRenderer->solidRenderer->getIndirectRenderer().readVisibleCount(this);
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

        // Update per-frame widget state (avoid storing VulkanApp* inside widgets)
        if (renderPassDebugWidget) renderPassDebugWidget->setFrameInfo(getCurrentFrame(), getWidth(), getHeight());
        if (vulkanResourcesManagerWidget) vulkanResourcesManagerWidget->updateWithApp(this);

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

        // --- SAFETY: Ensure indirect buffers are rebuilt if dirty before first draw ---
        if (sceneRenderer->solidRenderer->getIndirectRenderer().isDirty()) {
            printf("[MyApp::draw] solidRenderer indirect buffer dirty, rebuilding before draw...\n");
            sceneRenderer->solidRenderer->getIndirectRenderer().rebuild(this);
        }
        if (sceneRenderer->waterRenderer->getIndirectRenderer().isDirty()) {
            printf("[MyApp::draw] waterRenderer indirect buffer dirty, rebuilding before draw...\n");
            sceneRenderer->waterRenderer->getIndirectRenderer().rebuild(this);
        }

        uint32_t frameIdx = getCurrentFrame();
        glm::mat4 viewProj = camera.getViewProjectionMatrix();
        glm::mat4 invViewProj = glm::inverse(viewProj);

        // Composite offscreen scene + water into the swapchain
        if (sceneRenderer && sceneRenderer->waterRenderer) {
            sceneRenderer->waterRenderer->renderWaterPostProcess(
                this,
                commandBuffer,
                renderPassInfo.framebuffer,
                renderPassInfo.renderPass,
                sceneRenderer->solidRenderer->getColorView(frameIdx),
                sceneRenderer->solidRenderer->getDepthView(frameIdx),
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
            sceneRenderer->cleanup(this);
        }
        // NOTE: Vulkan-owned objects for global managers are now cleaned up by
        // `VulkanResourceManager::cleanup(device)`. Avoid calling manager-level
        // destroy/cleanup routines here that perform Vulkan destroys to prevent
        // double-destruction ordering issues. If a manager needs CPU-only
        // cleanup, add a dedicated method and call it here.
    }

    void onSwapchainResized(uint32_t width, uint32_t height) override {
        if (sceneRenderer) {
            sceneRenderer->onSwapchainResized(this, width, height);
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

// Implementation: setup scene
void MyApp::setupScene() {
    // Initialize and load the main scene so rendering has valid scene data
    mainScene = new LocalScene();
    octreeExplorerWidget = std::make_shared<OctreeExplorerWidget>(mainScene);

    // If you have vegetation: sceneRenderer->vegetationRenderer->rebuildBuffers(this);

    SolidSpaceChangeHandler solidHandler = sceneRenderer->makeSolidSpaceChangeHandler(mainScene, this);
    LiquidSpaceChangeHandler liquidHandler = sceneRenderer->makeLiquidSpaceChangeHandler(mainScene, this);
    UniqueOctreeChangeHandler uniqueSolidHandler = UniqueOctreeChangeHandler(solidHandler);
    UniqueOctreeChangeHandler uniqueLiquidHandler = UniqueOctreeChangeHandler(liquidHandler);

    MainSceneLoader loader = MainSceneLoader();
    mainScene->loadScene(loader, uniqueSolidHandler, uniqueLiquidHandler);

    uniqueSolidHandler.handleEvents();
    uniqueLiquidHandler.handleEvents();

    sceneRenderer->solidRenderer->getIndirectRenderer().setDirty(true);
    sceneRenderer->solidRenderer->getIndirectRenderer().rebuild(this);

    sceneRenderer->waterRenderer->getIndirectRenderer().setDirty(true);
    sceneRenderer->waterRenderer->getIndirectRenderer().rebuild(this);
}

// Implementation: setup vegetation textures
void MyApp::setupVegetationTextures() {
    // Allocate 3-layer texture arrays for vegetation (foliage, grass, wild)
    vegetationTextureArrayManager.allocate(3, 512, 512, this);
    vegetationAtlasEditor = std::make_shared<VegetationAtlasEditor>(&vegetationTextureArrayManager, &vegetationAtlasManager);
    billboardWidget = std::make_shared<BillboardWidget>();
    billboardWidgetManager = std::make_unique<BillboardWidgetManager>(billboardWidget, sceneRenderer->vegetationRenderer.get() , nullptr);
    billboardCreator = std::make_shared<BillboardCreator>(&billboardManager, &vegetationAtlasManager, &vegetationTextureArrayManager);
    
    // Load the vegetation atlas textures (albedo, normal, opacity) into the texture array
    std::vector<TextureTriple> vegTriples = {
        { "textures/vegetation/foliage_color.jpg", "textures/vegetation/foliage_normal.jpg", "textures/vegetation/foliage_opacity.jpg" },
        { "textures/vegetation/grass_color.jpg",   "textures/vegetation/grass_normal.jpg",   "textures/vegetation/grass_opacity.jpg" },
        { "textures/vegetation/wild_color.jpg",    "textures/vegetation/wild_normal.jpg",    "textures/vegetation/wild_opacity.jpg" }
    };
    size_t loaded = vegetationTextureArrayManager.loadTriples(this, vegTriples);
    fprintf(stderr, "[MyApp::setupVegetationTextures] Loaded %zu vegetation texture layers\n", loaded);

    // Auto-detect atlas tiles from opacity maps and populate AtlasManager for each texture
    const char* opacityPaths[3] = { "textures/vegetation/foliage_opacity.jpg", "textures/vegetation/grass_opacity.jpg", "textures/vegetation/wild_opacity.jpg" };
    for (int atlasIndex = 0; atlasIndex < 3; ++atlasIndex) {
        try {
            int added = vegetationAtlasManager.autoDetectTiles(atlasIndex, opacityPaths[atlasIndex]);
            fprintf(stderr, "[MyApp::setupVegetationTextures] Atlas %d: auto-detected %d tiles\n", atlasIndex, added);
        } catch (...) {
            fprintf(stderr, "[MyApp::setupVegetationTextures] Atlas %d: autoDetectTiles failed\n", atlasIndex);
        }
    }

    // Create simple billboards for each detected atlas tile so they appear in the editor
    for (int atlasIndex = 0; atlasIndex < 3; ++atlasIndex) {
        size_t tileCount = vegetationAtlasManager.getTileCount(atlasIndex);
        for (size_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
            std::string name = "Veg_" + std::to_string(atlasIndex) + "_" + std::to_string(tileIndex);
            size_t bidx = billboardManager.createBillboard(name);
            billboardManager.addLayer(bidx, atlasIndex, static_cast<int>(tileIndex));
        }
    }
}


