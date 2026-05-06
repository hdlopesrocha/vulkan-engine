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
#include <filesystem>
#include <cstring>
#include <cstdlib>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "vulkan/ubo/UniformObject.hpp"
#include "vulkan/VulkanApp.hpp"
#include "vulkan/renderer/SceneRenderer.hpp"
#include "utils/LocalScene.hpp"
#include "widgets/SettingsWidget.hpp"
#include "widgets/SkyWidget.hpp"
#include "widgets/SkySettings.hpp"
#include "widgets/WaterWidget.hpp"
#include "widgets/RenderTargetsWidget.hpp"
#include "widgets/BillboardCreator.hpp"
#include "widgets/ImpostorWidget.hpp"
#include "widgets/TextureMixerWidget.hpp"
#include "widgets/TextureViewerWidget.hpp"
#include "widgets/CameraWidget.hpp"
#include "events/ControllerManager.hpp"
#include "widgets/ControllerParametersWidget.hpp"
#include "widgets/GamepadWidget.hpp"
#include "widgets/LightWidget.hpp"
#include "widgets/VulkanResourcesManagerWidget.hpp"
#include "widgets/VegetationAtlasEditor.hpp"
#include "widgets/WindWidget.hpp"
#include "widgets/OctreeExplorerWidget.hpp"
#include "widgets/Brush3dWidget.hpp"
#include "widgets/MusicWidget.hpp"
#include "utils/MainSceneLoader.hpp"
#include "widgets/Settings.hpp"
#include "widgets/WidgetManager.hpp"
#include "math/Camera.hpp"
#include "math/Light.hpp"
#include "events/EventManager.hpp"
#include "events/KeyboardPublisher.hpp"
#include "events/GamepadPublisher.hpp"
#include "events/CloseWindowEvent.hpp"
#include "events/ToggleFullscreenEvent.hpp"
#include "events/RebuildBrushEvent.hpp"
#include "vulkan/TextureArrayManager.hpp"
#include "vulkan/MaterialManager.hpp"
#include "utils/BillboardManager.hpp"
#include "utils/AtlasManager.hpp"
#include "vulkan/TextureMixer.hpp"
#include "utils/ShadowParams.hpp"

class MyApp : public VulkanApp, public IEventHandler {
public:
    Settings settings;
    SceneRenderer * sceneRenderer;
    LocalScene * mainScene;
    LocalScene * brushScene = nullptr;
    std::shared_ptr<Brush3dWidget> brush3dWidget;
    // Shared brush entries edited by Brush3dWidget (owned by MyApp)
    Brush3dManager brushManager;
    static constexpr uint32_t QUERY_COUNT = 14; // 7 intervals × 2 timestamps each
    std::array<VkQueryPool, 2> queryPools = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    bool queryPoolReady[2] = {false, false};
    float timestampPeriod = 0.0f;
    bool profilingEnabled = true;
    float profileShadow = 0.0f;
    float profileMainCull = 0.0f;
    float profileSky = 0.0f;
    float profileSolidDraw = 0.0f;
    float profileVegetation = 0.0f;
    float profileWater = 0.0f;
    float profileImGui = 0.0f;
    float profileCpuUpdate = 0.0f;
    float profileCpuRecord = 0.0f;
    float profileFps = 0.0f;
    VkDescriptorSet shadowPassDescriptorSet = VK_NULL_HANDLE;
    UniformObject uboStatic = {};
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    std::shared_ptr<SettingsWidget> settingsWidget;
    std::shared_ptr<SkyWidget> skyWidget;
    std::shared_ptr<WaterWidget> waterWidget;
    std::shared_ptr<RenderTargetsWidget> renderTargetsWidget;
    std::shared_ptr<BillboardCreator> billboardCreator;
    std::shared_ptr<ImpostorWidget> impostorWidget;
    std::shared_ptr<TextureMixerWidget> textureMixerWidget;
    // flag set by background thread when mixer widget is ready; main thread will add it safely
    bool mixerWidgetPendingAdd = false;
    std::shared_ptr<TextureViewer> textureViewer;
    std::shared_ptr<CameraWidget> cameraWidget;
    ControllerManager controllerManager;
    std::shared_ptr<ControllerParametersWidget> controllerParametersWidget;
    std::shared_ptr<GamepadWidget> gamepadWidget;
    std::shared_ptr<LightWidget> lightWidget;
    std::shared_ptr<VulkanResourcesManagerWidget> vulkanResourcesManagerWidget;
    std::shared_ptr<VegetationAtlasEditor> vegetationAtlasEditor;
    std::shared_ptr<WindWidget> windWidget;
    std::shared_ptr<MusicWidget> mp3Widget;
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
    // Application-owned per-layer water parameters (initialized in setup)
    std::vector<WaterParams> waterParams;
    float mainTime = 0.0f;
    ShadowParams shadowParams;
    // When user clicks "Apply Brush" from ImGui we defer the heavy rebuild
    // until after the current frame is submitted to avoid waiting on fences
    // while the frame is being recorded (causes deadlock). Set by UI,
    // consumed in `postSubmit()`.
    bool brushRebuildPending = true;
    bool generateMapPending = false;
    bool loadScenePending = false;
    std::string pendingLoadPath;
    size_t cubeCount = 0;

    // Camera and input
    Camera camera = Camera(glm::vec3(2673.0f, 125.0f, 2043.0f), Math::eulerToQuat(0.0f, 0.0f, 0.0f));
    Light light = Light(glm::vec3(-1.0f, -1.0f, -1.0f));
    EventManager eventManager;
    KeyboardPublisher keyboardPublisher;
    GamepadPublisher gamepadPublisher;
    bool sceneLoading = false;

    // Handlers kept alive as members so the tessellation background thread
    // can safely use them after setup() returns.
    std::unique_ptr<SolidSpaceChangeHandler> sceneSolidHandler;
    std::unique_ptr<LiquidSpaceChangeHandler> sceneLiquidHandler;
    std::unique_ptr<UniqueOctreeChangeHandler> sceneUniqueSolidHandler;
    std::unique_ptr<UniqueOctreeChangeHandler> sceneUniqueLiquidHandler;
    std::thread sceneProcessThread; // tessellates chunks after octree is built

    ~MyApp() {
        if (sceneProcessThread.joinable()) sceneProcessThread.join();
    }

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
        std::cerr << "[TextureMixer] Running initial generation for configured mixers...\n";
        textureMixer->setEditableLayer(editableLayer);
        // Prime ImGui descriptors so the texture viewer shows immediately
        textureArrayManager.setLayerInitialized(editableLayer, true);
        textureArrayManager.getImTexture(editableLayer, 0);
        textureArrayManager.getImTexture(editableLayer, 1);
        textureArrayManager.getImTexture(editableLayer, 2);

        // Generate textures for all configured mixer entries (async submissions tracked by TextureMixer)
        textureMixer->generateInitialTextures(mixerParams);
        textureMixerWidget = std::make_shared<TextureMixerWidget>(textureMixer, mixerParams, "Texture Mixer");
        widgetManager.addWidget(textureMixerWidget);

        size_t materialCount = std::max<size_t>(static_cast<size_t>(loadedTextureLayers), static_cast<size_t>(loadedTextureLayers + 1));
        if (materialCount == 0) {
            materialCount = layerCount ? layerCount : 1u;
        }
        materials.assign(materialCount, MaterialProperties{});
        // Allocate GPU-side material storage via MaterialManager
        materialManager.allocate(materialCount, this);
        for (size_t i = 0; i < materialCount; ++i) materialManager.update(i, materials[i], this);

        // The descriptor set was bound to a dummy buffer at SceneRenderer::init() time
        // because materialManager hadn't been allocated yet (runs on a background thread).
        // Now that the real buffer exists, rebind descriptor set binding 5 so the GPU
        // reads from the actual materials SSBO.
        if (sceneRenderer) {
            sceneRenderer->updateTextureDescriptorSet(this, &textureArrayManager);
        }

    }

    void setup() override {
        sceneRenderer = new SceneRenderer();
        // Initialize application-owned water params with two default elements
        waterParams.push_back(WaterParams{}); // Add a third layer to demonstrate pagination in UI even without texture arrays
        {
            WaterParams wp = WaterParams();
            wp.noiseOctaves = 1;
            wp.waveScale = 8.0f;
            wp.deepColor = glm::vec3(0.0f, 0.1f, 0.0f);
            wp.causticColor = glm::vec3(0.5f, 1.0f, 0.0f);
            wp.shallowColor = glm::vec3(0.1f, 0.5f, 0.1f);
            wp.waterTint = 0.6f;
            wp.causticType = 1; // line-shaped caustics
            wp.causticIntensity = 0.2f;
            wp.causticScale = 0.01f;
            wp.causticDepthScale = 128.0f;
            wp.causticPower = 4.0f;
            wp.causticLineScale = 3.0f;
            waterParams.push_back(wp); // Add a third layer to demonstrate pagination in UI even without texture arrays
        }
        {
            WaterParams wp = WaterParams();
            wp.enableRefraction = false;
            wp.noiseOctaves = 0;
            wp.waveScale = 0.0f;
            wp.noiseScale = 0.0f;
            wp.deepColor = glm::vec3(0.0f, 0.0f, 0.0f);
            wp.causticColor = glm::vec3(1.0f, 1.0f, 1.0f);
            wp.shallowColor = glm::vec3(1.0f, 1.0f, 1.0f);
            wp.waterTint = 1.0f;
            wp.causticType = 1; // line-shaped caustics
            wp.causticIntensity = 0.0f;
            wp.causticDepthScale = 128.0f;
            wp.causticPower = 4.0f;
            wp.causticLineScale = 3.0f;
            wp.bumpAmplitude = 0.0f;
            wp.blurRadius = 4.0f;
            wp.enableBlur = true;
            wp.reflectionStrength = 1.0f;
            wp.fresnelPower = 1.0f;
            waterParams.push_back(wp); // Add a third layer to demonstrate pagination in UI even without texture arrays
        }


        // Create scene objects and build the octree synchronously in setup().
        // Chunk tessellation is deferred and processed on a background thread.
        // Vulkan GPU uploads happen on the main thread via processPendingMeshes().
        mainScene = new LocalScene();
        octreeExplorerWidget = std::make_shared<OctreeExplorerWidget>(mainScene, &camera);
        widgetManager.addWidget(octreeExplorerWidget);
        brushScene = new LocalScene();
        brushManager.getEntries().clear();
        brushManager.getEntries().resize(3);
        brushManager.getEntries()[0].sdfType = 0;
        brushManager.getEntries()[0].materialIndex = 0;
        brushManager.getEntries()[0].translate = glm::vec3(0.0f, 1024.0f, 0.0f);
        brushManager.getEntries()[0].scale = glm::vec3(256.0f);
        brushManager.getEntries()[1].sdfType = 1;
        brushManager.getEntries()[1].materialIndex = 1;
        brushManager.getEntries()[1].translate = glm::vec3(512.0f, 1024.0f, 0.0f);
        brushManager.getEntries()[1].scale = glm::vec3(256.0f);
        brushManager.getEntries()[2].sdfType = 3;
        brushManager.getEntries()[2].materialIndex = 2;
        brushManager.getEntries()[2].translate = glm::vec3(-512.0f, 1024.0f, 0.0f);
        brushManager.getEntries()[2].scale = glm::vec3(256.0f);
        sceneSolidHandler  = std::make_unique<SolidSpaceChangeHandler>(sceneRenderer->makeSolidSpaceChangeHandler(mainScene, this));
        sceneLiquidHandler = std::make_unique<LiquidSpaceChangeHandler>(sceneRenderer->makeLiquidSpaceChangeHandler(mainScene, this));
        sceneUniqueSolidHandler  = std::make_unique<UniqueOctreeChangeHandler>(*sceneSolidHandler);
        sceneUniqueLiquidHandler = std::make_unique<UniqueOctreeChangeHandler>(*sceneLiquidHandler);

        // Scene starts empty — use File > Generate Map to populate it.
        if (octreeExplorerWidget)
            octreeExplorerWidget->octreeReady.store(true, std::memory_order_release);

        setupVegetationTextures();
        setupTextures();

        sceneRenderer->init(this, &textureArrayManager, &materialManager, waterParams);

        // Re-wire impostors now that VegetationRenderer::init() has stored the render pass.
        if (impostorWidget) impostorWidget->rewire();

        // Keep the vegetation array manager wired for editor/atlas updates.
        if (sceneRenderer->vegetationRenderer)
            sceneRenderer->vegetationRenderer->setTextureArrayManager(&vegetationTextureArrayManager, this);

        // Bind billboard array textures (sampler2DArray per channel) to the vegetation renderer.
        if (sceneRenderer->vegetationRenderer && billboardCreator) {
            sceneRenderer->vegetationRenderer->setBillboardArrayTextures(
                billboardCreator->getAlbedoArrayView(),
                billboardCreator->getNormalArrayView(),
                billboardCreator->getOpacityArrayView(),
                billboardCreator->getArraySampler(),
                this
            );
        }

        printf("[MyApp::setup] Created and initialized SceneRenderer\n");

        textureViewer = std::make_shared<TextureViewer>();
        textureViewer->init(&textureArrayManager, &materials);
        textureViewer->setOnMaterialChanged([this](size_t idx) {
            materialManager.update(idx, materials[idx], this);
        });

        skyWidget = std::make_shared<SkyWidget>(sceneRenderer->getSkySettings());
        // Create settings widget (was missing previously)
        settingsWidget = std::make_shared<SettingsWidget>(settings, &shadowParams);
        // Water UI uses the application-owned water params vector and updates GPU state explicitly.
        waterWidget = std::make_shared<WaterWidget>(sceneRenderer->waterRenderer.get(), &waterParams);

        renderTargetsWidget = std::make_shared<RenderTargetsWidget>(
            this,
            sceneRenderer, sceneRenderer->solidRenderer.get(), sceneRenderer->skyRenderer.get(),
            sceneRenderer->shadowMapper.get(), &shadowParams);
        if (renderTargetsWidget) renderTargetsWidget->setFrameInfo(getCurrentFrame(), getWidth(), getHeight());

        cameraWidget = std::make_shared<CameraWidget>(&camera);
        controllerParametersWidget = std::make_shared<ControllerParametersWidget>(controllerManager.getParameters(), &brushManager);
        gamepadWidget = std::make_shared<GamepadWidget>(controllerManager.getParameters());
        lightWidget = std::make_shared<LightWidget>(&light);
        vulkanResourcesManagerWidget = std::make_shared<VulkanResourcesManagerWidget>(&resources);
        vulkanResourcesManagerWidget->updateWithApp(this);
        windWidget = std::make_shared<WindWidget>(sceneRenderer->vegetationRenderer.get());
        mp3Widget = std::make_shared<MusicWidget>();
  // Create octree explorer widget bound to loaded scene

 
        widgetManager.addWidget(textureViewer);
        widgetManager.addWidget(cameraWidget);
        widgetManager.addWidget(controllerParametersWidget);
        widgetManager.addWidget(gamepadWidget);
        widgetManager.addWidget(settingsWidget);
        widgetManager.addWidget(lightWidget);
        widgetManager.addWidget(skyWidget);
        widgetManager.addWidget(waterWidget);
        widgetManager.addWidget(renderTargetsWidget);
        widgetManager.addWidget(vulkanResourcesManagerWidget);
        widgetManager.addWidget(vegetationAtlasEditor);
        widgetManager.addWidget(windWidget);
        widgetManager.addWidget(mp3Widget);
        widgetManager.addWidget(billboardCreator);
        widgetManager.addWidget(impostorWidget);

      
        // Subscribe event handlers
        eventManager.subscribe(&camera);  // Camera handles translate/rotate events
        eventManager.subscribe(this);     // MyApp handles close/fullscreen events
        
        // Set up camera projection matrix
        float aspectRatio = static_cast<float>(getWidth()) / static_cast<float>(getHeight());
        glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspectRatio, settings.nearPlane, settings.farPlane);
        proj[1][1] *= -1; // Vulkan Y-flip
        camera.setProjection(proj);
        shadowParams.update(camera.getPosition(), light);
        
        // Position camera to view the terrain
        printf("[Camera Setup] Final Position: (%.1f, %.1f, %.1f)\n", camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
        printf("[Camera Setup] Forward: (%.3f, %.3f, %.3f)\n", camera.getForward().x, camera.getForward().y, camera.getForward().z);
   
        // Create brush3dWidget after setupTextures() so loadedTextureLayers is set.
        brush3dWidget = std::make_shared<Brush3dWidget>(&textureArrayManager, loadedTextureLayers, brushManager, &eventManager);
        widgetManager.addWidget(brush3dWidget);

        // Create per-frame timestamp query pools for GPU profiling
        {
            VkPhysicalDeviceProperties physProps{};
            vkGetPhysicalDeviceProperties(getPhysicalDevice(), &physProps);
            timestampPeriod = physProps.limits.timestampPeriod;
            if (timestampPeriod > 0.0f) {
                VkQueryPoolCreateInfo qpci{};
                qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
                qpci.queryCount = QUERY_COUNT;
                for (uint32_t f = 0; f < 2; ++f) {
                    if (vkCreateQueryPool(getDevice(), &qpci, nullptr, &queryPools[f]) != VK_SUCCESS)
                        throw std::runtime_error("Failed to create timestamp query pool");
                }
            }
        }
    }

    // Move vegetation texture setup into its own method for clarity
    void setupVegetationTextures();
    // Move scene-loading into its own method for clarity
    void setupScene();
    // Rebuild the brush preview scene from Brush3dWidget entries
    void rebuildBrushScene();
    // Clear GPU meshes, reset octrees and regenerate via MainSceneLoader
    void generateMap();
    // Clear GPU meshes, reset octrees, load from file and tessellate
    void loadSceneFromFile(const std::string& path);

// (setup implementation defined out-of-line below)

    void update(float deltaTime) override {
        if (deltaTime > 0.0f) profileFps = 1.0f / deltaTime;
        auto cpuUpdateT0 = std::chrono::high_resolution_clock::now();
        // Poll keyboard input and publish events
        keyboardPublisher.update(getWindow(), &eventManager, camera, deltaTime, &controllerManager, &brushManager, false);
        // Poll gamepad input and publish events (if a controller is connected)
        gamepadPublisher.update(&eventManager, camera, deltaTime, &controllerManager, &brushManager, false);
        eventManager.processQueued();

        shadowParams.update(camera.getPosition(), light);

        // Drain the pending mesh queue populated by the background scene-loading
        // thread.  GPU uploads happen here on the main thread so newly generated
        // chunks become visible progressively without blocking the render loop.
        if (sceneRenderer) sceneRenderer->processPendingMeshes(this, camera.getPosition());

        mainTime += deltaTime;
        if (sceneRenderer && sceneRenderer->vegetationRenderer) {
            sceneRenderer->vegetationRenderer->setWindTime(mainTime);
            sceneRenderer->vegetationRenderer->setImpostorDistance(settings.impostorDistance);
        }
        profileCpuUpdate = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - cpuUpdateT0).count();
    }

    void preRenderPass(VkCommandBuffer &commandBuffer) override {
        uint32_t frameIdx = getCurrentFrame();

        // Profiling: read previous frame's query results (fence guaranteed signaled), then reset for this frame
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE) {
            if (queryPoolReady[frameIdx]) {
                uint64_t ts[QUERY_COUNT] = {};
                if (vkGetQueryPoolResults(getDevice(), queryPools[frameIdx], 0, QUERY_COUNT,
                        sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS
                        && timestampPeriod > 0.0f) {
                    auto ms = [&](uint32_t a, uint32_t b) -> float {
                        return static_cast<float>(ts[b] - ts[a]) * timestampPeriod * 1e-6f;
                    };
                    profileShadow     = ms(0,  1);
                    profileMainCull   = ms(2,  3);
                    profileSky        = ms(4,  5);
                    profileSolidDraw  = ms(6,  7);
                    profileVegetation = ms(8,  9);
                    profileWater      = ms(10, 11);
                    profileImGui      = ms(12, 13);
                }
            }
            vkCmdResetQueryPool(commandBuffer, queryPools[frameIdx], 0, QUERY_COUNT);
            queryPoolReady[frameIdx] = true;
        }

        auto cpuRecordT0 = std::chrono::high_resolution_clock::now();

        // Rebuild projection matrix FIRST so viewProj below reflects current near/far settings
        {
            float aspectRatio = static_cast<float>(getWidth()) / static_cast<float>(getHeight());
            glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspectRatio, settings.nearPlane, settings.farPlane);
            proj[1][1] *= -1; // Vulkan Y-flip
            camera.setProjection(proj);
        }

        // Build per-frame UBO (viewProj now includes the updated projection)
        glm::mat4 viewProj = camera.getViewProjectionMatrix();
        uboStatic.viewProjection = viewProj;
        uboStatic.viewPos = glm::vec4(camera.getPosition(), 1.0f);
        uboStatic.lightDir = glm::vec4(light.getDirection(), 0.0f);
        uboStatic.lightColor = glm::vec4(1.0f, 1.0f, 0.9f, 1.0f);
        uboStatic.lightSpaceMatrix  = shadowParams.lightSpaceMatrix[0];
        uboStatic.lightSpaceMatrix1 = shadowParams.lightSpaceMatrix[1];
        uboStatic.lightSpaceMatrix2 = shadowParams.lightSpaceMatrix[2];
        // Encode debug/triplanar/tess parameters into the shared UBO
        uboStatic.debugParams = glm::vec4(static_cast<float>(settings.debugMode), 0.0f, 0.0f, 0.0f);
        uboStatic.triplanarSettings = glm::vec4(settings.triplanarThreshold, settings.triplanarExponent, 0.0f, 0.0f);
        uboStatic.tessParams = glm::vec4(
            settings.tessMinDistance,
            settings.tessMaxDistance,
            settings.tessMinLevel,
            settings.tessMaxLevel
        );
        // passParams: x = isShadowPass, y = tessEnabled, z = nearPlane, w = farPlane
        uboStatic.passParams = glm::vec4(0.0f, settings.tessellationEnabled ? 1.0f : 0.0f, settings.nearPlane, settings.farPlane);
        // materialFlags.w = global normal-mapping toggle (shader checks ubo.materialFlags.w > 0.5)
        uboStatic.materialFlags.w = settings.normalMappingEnabled ? 1.0f : 0.0f;
        // shadowEffects.w = global shadow toggle (shader checks ubo.shadowEffects.w > 0.5)
        uboStatic.shadowEffects.w = settings.enableShadows ? 1.0f : 0.0f;

        // Shadow pass: renders solid geometry into shadow map from light's perspective
        // Must run AFTER UBO is built (needs lightSpaceMatrix, passParams, etc.)
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 0);
        if (sceneRenderer) {
            sceneRenderer->shadowPass(this, commandBuffer, getMainDescriptorSet(), sceneRenderer->mainUniformBuffer, uboStatic, settings.enableShadows, settings.vegetationEnabled);
        } else {
            std::cerr << "[MyApp::preRenderPass] sceneRenderer is null, skipping shadow pass\n";
        }
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 1);

        // Upload UBO to GPU
        if (sceneRenderer) {
            void* data;
            vkMapMemory(getDevice(), sceneRenderer->mainUniformBuffer.memory, 0, sizeof(UniformObject), 0, &data);
            memcpy(data, &uboStatic, sizeof(UniformObject));
            vkUnmapMemory(getDevice(), sceneRenderer->mainUniformBuffer.memory);
        } else {
            std::cerr << "[MyApp::preRenderPass] sceneRenderer is null, skipping UBO upload\n";
        }


        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 2);
        sceneRenderer->solidRenderer->getIndirectRenderer().prepareCull(commandBuffer, viewProj);
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 3);

        const bool waterEnabled = settings.waterEnabled;
        const bool vegetationEnabled = settings.vegetationEnabled;

        // Render sky + solids/vegetation into the solid offscreen framebuffer (one per frame)

        // Render sky to its own offscreen FBO so it can be sampled as a texture by the water shader
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 4);
        if (sceneRenderer->skyRenderer) {
            SkySettings::Mode skyMode = sceneRenderer->getSkySettings().mode;
            sceneRenderer->skyRenderer->renderOffscreen(this, commandBuffer, frameIdx,
                getMainDescriptorSet(), sceneRenderer->mainUniformBuffer, uboStatic, viewProj, skyMode);
        }

        VkClearValue colorClear{};
        // Clear solid offscreen color to transparent so composite starts from empty scene
        colorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        VkClearValue depthClear{};
        depthClear.depthStencil = {1.0f, 0};
            sceneRenderer->solidRenderer->beginPass(commandBuffer, frameIdx, colorClear, depthClear, this);

        // Render sky first inside the solid pass so water composites on top of (sky + solid).
        // skyRenderer->render() temporarily overwrites the UBO with sky-specific values,
        // so we must re-upload the main UBO afterward.
        if (sceneRenderer->skyRenderer) {
            SkySettings::Mode skyMode = sceneRenderer->getSkySettings().mode;
            sceneRenderer->skyRenderer->render(this, commandBuffer, getMainDescriptorSet(),
                sceneRenderer->mainUniformBuffer, uboStatic, viewProj, skyMode);

            // Restore the main UBO that sky rendering overwrote
            void* data;
            vkMapMemory(getDevice(), sceneRenderer->mainUniformBuffer.memory, 0, sizeof(UniformObject), 0, &data);
            memcpy(data, &uboStatic, sizeof(UniformObject));
            vkUnmapMemory(getDevice(), sceneRenderer->mainUniformBuffer.memory);
        }
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 5);

        // Sky is now rendered inside the solid pass above, before solid geometry.
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 6);
        sceneRenderer->mainPass(this, commandBuffer, frameIdx, waterEnabled, getMainDescriptorSet(), sceneRenderer->mainUniformBuffer, settings.wireframeMode,
            viewProj, uboStatic, true, false, true, 0, 0.0f, 0.0f);
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 7);

        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 8);
        if (vegetationEnabled && sceneRenderer->vegetationRenderer) {
            sceneRenderer->vegetationRenderer->draw(this, commandBuffer, getMainDescriptorSet(), viewProj, camera.getPosition());
        }
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 9);

        // Render debug cubes for expanded octree nodes, node instances, and vegetation density centers.
        const bool showOctreeDebug = octreeExplorerWidget && octreeExplorerWidget->getShowDebugCubes();
        const bool showVegetationDensityDebug = false;
        if (showOctreeDebug || showVegetationDensityDebug) {
            std::vector<DebugCubeRenderer::CubeWithColor> debugCubes;
            // Add widget-expanded cubes when explorer is visible
            if (showOctreeDebug && octreeExplorerWidget->isVisible()) {
                const auto& widgetCubes = octreeExplorerWidget->getExpandedCubes();
                debugCubes.reserve(widgetCubes.size());
                for (const auto& wc : widgetCubes) {
                    // Convert BoundingCube -> BoundingBox for renderer
                    debugCubes.push_back({BoundingBox(wc.cube.getMin(), wc.cube.getMax()), wc.color});
                }
            }
            if (showOctreeDebug) {
                // Append node cubes produced by change handlers (post-tessellation)
                auto nodeCubes = sceneRenderer->getDebugNodeCubes();
                debugCubes.reserve(debugCubes.size() + nodeCubes.size());
                for (auto &nc : nodeCubes) debugCubes.push_back(nc);
            }
            if (showVegetationDensityDebug && sceneRenderer->vegetationRenderer) {
                auto densityCubes = sceneRenderer->vegetationRenderer->getDensityDebugCubes(camera.getPosition());
                debugCubes.reserve(debugCubes.size() + densityCubes.size());
                for (auto &cube : densityCubes) debugCubes.push_back(cube);
            }

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

            sceneRenderer->solidRenderer->endPass(commandBuffer, frameIdx, this);

        // Update the water scene descriptor set BEFORE launching async tasks.
        // The backFace async task binds waterDepthDescriptorSets[frameIdx] in its submitted
        // command buffer. If we update it AFTER the submission, the validation layer fires
        // VUID-vkUpdateDescriptorSets-None-03047 (updating a descriptor set in use by a
        // pending command buffer). By updating here we guarantee the descriptor set is
        // fully updated before any command buffer referencing it is submitted.
        if (waterEnabled && sceneRenderer && sceneRenderer->waterRenderer) {
            VkImageView wSceneColor = sceneRenderer->solidRenderer->getColorView(frameIdx);
            VkImageView wSceneDepth = sceneRenderer->solidRenderer->getDepthView(frameIdx);
            VkImageView wSky  = (sceneRenderer->skyRenderer)     ? sceneRenderer->skyRenderer->getSkyView(frameIdx) : VK_NULL_HANDLE;
            VkImageView wBack = (sceneRenderer->backFaceRenderer) ? sceneRenderer->backFaceRenderer->getBackFaceDepthView(frameIdx) : VK_NULL_HANDLE;
            VkImageView wCube = (sceneRenderer->solid360Renderer) ? sceneRenderer->solid360Renderer->getSolid360View() : VK_NULL_HANDLE;
            sceneRenderer->waterRenderer->updateSceneTexturesBinding(this, wSceneColor, wSceneDepth, frameIdx, wSky, wBack, wCube);
        }

        // Launch asynchronous recording+submit for independent offscreen passes
        std::vector<std::thread> asyncTasks;
        // RAII guard: ensure any launched threads are joined if an exception
        // escapes this function so std::terminate is not called from thread
        // destructors. Joining here replicates the previous explicit join
        // behavior in all exit paths.
        struct ThreadJoiner {
            std::vector<std::thread>& threads_;
            ThreadJoiner(std::vector<std::thread>& t) : threads_(t) {}
            ~ThreadJoiner() {
                for (auto &thr : threads_) {
                    if (thr.joinable()) thr.join();
                }
            }
        } threadJoiner(asyncTasks);
        bool launchedSolid360 = false;
        bool launchedBackFace = false;
        VkSemaphore semSolid360 = VK_NULL_HANDLE;
        VkSemaphore semBackFace = VK_NULL_HANDLE;

        // 360° cubemap (sky + solid) - record/submit on a separate primary command buffer
        if (waterEnabled && sceneRenderer && sceneRenderer->solid360Renderer) {
            launchedSolid360 = true;
            asyncTasks.emplace_back([this, viewProj, frameIdx, &semSolid360]() {
                VulkanApp* app = this;
                VkCommandBuffer cmd = app->allocatePrimaryCommandBuffer();
                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
                    std::cerr << "[Async] vkBeginCommandBuffer failed for solid360\n";
                    app->freeCommandBuffer(cmd);
                    return;
                }
                // Allocate per-task compact/visible buffers and descriptor set so cull+draw
                // recorded on this command buffer don't race with other submissions.
                IndirectRenderer &ind = this->sceneRenderer->solidRenderer->getIndirectRenderer();
                uint32_t numCmds = static_cast<uint32_t>(ind.getMeshCount());
                VkDeviceSize compactSize = sizeof(VkDrawIndexedIndirectCommand) * (numCmds > 0 ? numCmds : 1);

                Buffer taskCompact = app->createBuffer(compactSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                Buffer taskVisible = app->createBuffer(sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                VkDevice device = app->getDevice();
                // Create a small temporary descriptor pool for this task
                VkDescriptorPoolSize poolSize{};
                poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                poolSize.descriptorCount = 4;
                VkDescriptorPoolCreateInfo poolInfo{};
                poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                poolInfo.poolSizeCount = 1;
                poolInfo.pPoolSizes = &poolSize;
                poolInfo.maxSets = 1;
                poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
                // Try to allocate the compute descriptor set from the IndirectRenderer's shared pool
                VkDescriptorPool taskPool = VK_NULL_HANDLE;
                VkDescriptorSetLayout dsLayout = ind.getComputeDescriptorSetLayout();
                VkDescriptorSet computeDs = VK_NULL_HANDLE;
                VkDescriptorPool sharedPool = ind.getComputeDescriptorPool();
                if (dsLayout != VK_NULL_HANDLE && sharedPool != VK_NULL_HANDLE) {
                    VkDescriptorSetAllocateInfo ainfoShared{};
                    ainfoShared.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    ainfoShared.descriptorPool = sharedPool;
                    ainfoShared.descriptorSetCount = 1;
                    ainfoShared.pSetLayouts = &dsLayout;
                    if (vkAllocateDescriptorSets(device, &ainfoShared, &computeDs) == VK_SUCCESS) {
                        app->resources.addDescriptorSet(computeDs, "Shared: solid360 compute DS");
                    } else {
                        computeDs = VK_NULL_HANDLE;
                    }
                }

                // If shared allocation failed, create a temporary pool and allocate from it
                if (computeDs == VK_NULL_HANDLE) {
                    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &taskPool) != VK_SUCCESS) {
                        throw std::runtime_error("[Async] Failed to create descriptor pool for solid360 task (no fallback allowed)");
                    }
                    app->resources.addDescriptorPool(taskPool, "Temp: solid360 cull pool");

                    if (dsLayout != VK_NULL_HANDLE) {
                        VkDescriptorSetAllocateInfo ainfo{};
                        ainfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                        ainfo.descriptorPool = taskPool;
                        ainfo.descriptorSetCount = 1;
                        ainfo.pSetLayouts = &dsLayout;
                        if (vkAllocateDescriptorSets(device, &ainfo, &computeDs) != VK_SUCCESS) {
                            app->resources.removeDescriptorPool(taskPool);
                            vkDestroyDescriptorPool(device, taskPool, nullptr);
                            throw std::runtime_error("[Async] Failed to allocate compute descriptor set for solid360 task (no fallback allowed)");
                        }
                        app->resources.addDescriptorSet(computeDs, "Temp: solid360 compute DS");
                    }
                }

                // Run cull into per-task buffers - only when compute pipeline is ready (meshes loaded)
                if (computeDs != VK_NULL_HANDLE) {
                    VkDescriptorBufferInfo inBuf{}; inBuf.buffer = ind.getIndirectBuffer().buffer; inBuf.offset = 0; inBuf.range = VK_WHOLE_SIZE;
                    VkDescriptorBufferInfo outBuf{}; outBuf.buffer = taskCompact.buffer; outBuf.offset = 0; outBuf.range = VK_WHOLE_SIZE;
                    VkDescriptorBufferInfo boundsBuf{}; boundsBuf.buffer = ind.getBoundsBuffer().buffer; boundsBuf.offset = 0; boundsBuf.range = VK_WHOLE_SIZE;
                    VkDescriptorBufferInfo countBuf{}; countBuf.buffer = taskVisible.buffer; countBuf.offset = 0; countBuf.range = VK_WHOLE_SIZE;

                    VkWriteDescriptorSet writes[4]{};
                    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    writes[0].dstSet = computeDs;
                    writes[0].dstBinding = 0;
                    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[0].descriptorCount = 1;
                    writes[0].pBufferInfo = &inBuf;

                    writes[1] = writes[0]; writes[1].dstBinding = 1; writes[1].pBufferInfo = &outBuf;
                    writes[2] = writes[0]; writes[2].dstBinding = 2; writes[2].pBufferInfo = &boundsBuf;
                    writes[3] = writes[0]; writes[3].dstBinding = 3; writes[3].pBufferInfo = &countBuf;

                    vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
                    ind.prepareCullWithDescriptor(cmd, viewProj, computeDs, taskCompact.buffer, taskVisible.buffer);
                }

                // Render solid360 using per-task compact/visible buffers when available
                SkySettings::Mode skyMode360 = this->sceneRenderer->getSkySettings().mode;
                this->sceneRenderer->solid360Renderer->renderSolid360(
                    app, cmd,
                    this->sceneRenderer->skyRenderer.get(), skyMode360,
                    this->sceneRenderer->solidRenderer.get(),
                    app->getMainDescriptorSet(),
                    this->sceneRenderer->mainUniformBuffer, this->uboStatic,
                    (computeDs != VK_NULL_HANDLE) ? taskCompact.buffer : VK_NULL_HANDLE,
                    (computeDs != VK_NULL_HANDLE) ? taskVisible.buffer : VK_NULL_HANDLE);

                // Submit and schedule cleanup of temporary resources after fence signals
                VkFence f = app->submitCommandBufferAsync(cmd, &semSolid360);
                app->deferDestroyUntilFence(f, [device, taskPool, computeDs, taskCompact, taskVisible, app]() {
                    // Unregister and destroy descriptor set/pool
                    app->resources.removeDescriptorSet(computeDs);
                    if (taskPool != VK_NULL_HANDLE) {
                        app->resources.removeDescriptorPool(taskPool);
                        vkDestroyDescriptorPool(device, taskPool, nullptr);
                    }
                    // Destroy buffers and free memory
                    if (taskCompact.buffer != VK_NULL_HANDLE) {
                        app->resources.removeBuffer(taskCompact.buffer);
                        vkDestroyBuffer(device, taskCompact.buffer, nullptr);
                    }
                    if (taskCompact.memory != VK_NULL_HANDLE) {
                        app->resources.removeDeviceMemory(taskCompact.memory);
                        vkFreeMemory(device, taskCompact.memory, nullptr);
                    }
                    if (taskVisible.buffer != VK_NULL_HANDLE) {
                        app->resources.removeBuffer(taskVisible.buffer);
                        vkDestroyBuffer(device, taskVisible.buffer, nullptr);
                    }
                    if (taskVisible.memory != VK_NULL_HANDLE) {
                        app->resources.removeDeviceMemory(taskVisible.memory);
                        vkFreeMemory(device, taskVisible.memory, nullptr);
                    }
                });
            });
        }

        // Back-face depth for water - record/submit on a separate primary command buffer
        if (waterEnabled && sceneRenderer && sceneRenderer->backFaceRenderer) {
            launchedBackFace = true;
            asyncTasks.emplace_back([this, viewProj, frameIdx, &semBackFace]() {
                VulkanApp* app = this;
                VkCommandBuffer cmd = app->allocatePrimaryCommandBuffer();
                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
                    std::cerr << "[Async] vkBeginCommandBuffer failed for backFace" << std::endl;
                    app->freeCommandBuffer(cmd);
                    return;
                }
                // Allocate per-task compact/visible buffers and descriptor set so cull+draw
                // recorded on this command buffer don't race with other submissions.
                IndirectRenderer &ind = this->sceneRenderer->waterRenderer->getIndirectRenderer();
                uint32_t numCmds = static_cast<uint32_t>(ind.getMeshCount());
                VkDeviceSize compactSize = sizeof(VkDrawIndexedIndirectCommand) * (numCmds > 0 ? numCmds : 1);
                
                Buffer taskCompact = app->createBuffer(compactSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                Buffer taskVisible = app->createBuffer(sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                VkDevice device = app->getDevice();
                // Create a small temporary descriptor pool for this task
                VkDescriptorPoolSize poolSize{};
                poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                poolSize.descriptorCount = 4;
                VkDescriptorPoolCreateInfo poolInfo{};
                poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                poolInfo.poolSizeCount = 1;
                poolInfo.pPoolSizes = &poolSize;
                poolInfo.maxSets = 1;
                poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
                VkDescriptorPool taskPool = VK_NULL_HANDLE;
                if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &taskPool) != VK_SUCCESS) {
                    throw std::runtime_error("[Async] Failed to create descriptor pool for backFace task (no fallback allowed)");
                }
                app->resources.addDescriptorPool(taskPool, "Temp: backface cull pool");

                // Allocate descriptor set from IndirectRenderer's compute layout
                VkDescriptorSetLayout dsLayout = ind.getComputeDescriptorSetLayout();
                VkDescriptorSet computeDs = VK_NULL_HANDLE;
                if (dsLayout != VK_NULL_HANDLE) {
                    VkDescriptorSetAllocateInfo ainfo{};
                    ainfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    ainfo.descriptorPool = taskPool;
                    ainfo.descriptorSetCount = 1;
                    ainfo.pSetLayouts = &dsLayout;
                    if (vkAllocateDescriptorSets(device, &ainfo, &computeDs) != VK_SUCCESS) {
                        app->resources.removeDescriptorPool(taskPool);
                        vkDestroyDescriptorPool(device, taskPool, nullptr);
                        throw std::runtime_error("[Async] Failed to allocate compute descriptor set for backFace task (no fallback allowed)");
                    }
                    app->resources.addDescriptorSet(computeDs, "Temp: backface compute DS");

                    // Update descriptor set with buffers: inCmds, outCmds, bounds, visibleCount
                    VkDescriptorBufferInfo inBuf{}; inBuf.buffer = ind.getIndirectBuffer().buffer; inBuf.offset = 0; inBuf.range = VK_WHOLE_SIZE;
                    VkDescriptorBufferInfo outBuf{}; outBuf.buffer = taskCompact.buffer; outBuf.offset = 0; outBuf.range = VK_WHOLE_SIZE;
                    VkDescriptorBufferInfo boundsBuf{}; boundsBuf.buffer = ind.getBoundsBuffer().buffer; boundsBuf.offset = 0; boundsBuf.range = VK_WHOLE_SIZE;
                    VkDescriptorBufferInfo countBuf{}; countBuf.buffer = taskVisible.buffer; countBuf.offset = 0; countBuf.range = VK_WHOLE_SIZE;

                    VkWriteDescriptorSet writes[4]{};
                    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    writes[0].dstSet = computeDs;
                    writes[0].dstBinding = 0;
                    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[0].descriptorCount = 1;
                    writes[0].pBufferInfo = &inBuf;

                    writes[1] = writes[0]; writes[1].dstBinding = 1; writes[1].pBufferInfo = &outBuf;
                    writes[2] = writes[0]; writes[2].dstBinding = 2; writes[2].pBufferInfo = &boundsBuf;
                    writes[3] = writes[0]; writes[3].dstBinding = 3; writes[3].pBufferInfo = &countBuf;

                    vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
                }

                // Run cull into per-task buffers - only when compute pipeline is ready (meshes loaded)
                if (computeDs != VK_NULL_HANDLE) {
                    ind.prepareCullWithDescriptor(cmd, viewProj, computeDs, taskCompact.buffer, taskVisible.buffer);
                }

                // Render back-face pass using the per-task compact/visible buffers so draws consume the cull results
                this->sceneRenderer->backFaceRenderer->renderBackFacePass(app, cmd, frameIdx,
                                            ind,
                                            this->sceneRenderer->waterRenderer->getWaterGeometryPipelineLayout(),
                                            app->getMainDescriptorSet(),
                                            app->getMaterialDescriptorSet(),
                                            this->sceneRenderer->waterRenderer->getWaterDepthDescriptorSet(frameIdx),
                                            this->sceneRenderer->solidRenderer->getDepthImage(frameIdx),
                                            (computeDs != VK_NULL_HANDLE) ? taskCompact.buffer : VK_NULL_HANDLE,
                                            (computeDs != VK_NULL_HANDLE) ? taskVisible.buffer : VK_NULL_HANDLE);

                // Submit and schedule cleanup of temporary resources after fence signals
                VkFence f = app->submitCommandBufferAsync(cmd, &semBackFace);
                app->deferDestroyUntilFence(f, [device, taskPool, computeDs, taskCompact, taskVisible, app]() {
                    // Unregister and destroy descriptor set/pool
                    app->resources.removeDescriptorSet(computeDs);
                    if (taskPool != VK_NULL_HANDLE) {
                        app->resources.removeDescriptorPool(taskPool);
                        vkDestroyDescriptorPool(device, taskPool, nullptr);
                    }
                    // Destroy buffers and free memory
                    if (taskCompact.buffer != VK_NULL_HANDLE) {
                        app->resources.removeBuffer(taskCompact.buffer);
                        vkDestroyBuffer(device, taskCompact.buffer, nullptr);
                    }
                    if (taskCompact.memory != VK_NULL_HANDLE) {
                        app->resources.removeDeviceMemory(taskCompact.memory);
                        vkFreeMemory(device, taskCompact.memory, nullptr);
                    }
                    if (taskVisible.buffer != VK_NULL_HANDLE) {
                        app->resources.removeBuffer(taskVisible.buffer);
                        vkDestroyBuffer(device, taskVisible.buffer, nullptr);
                    }
                    if (taskVisible.memory != VK_NULL_HANDLE) {
                        app->resources.removeDeviceMemory(taskVisible.memory);
                        vkFreeMemory(device, taskVisible.memory, nullptr);
                    }
                });
            });
        }

        // Wait for any async recording/submits to complete before water pass so no two
        // threads call vkCmdBindDescriptorSets with the same descriptor set concurrently.
        for (auto &t : asyncTasks) {
            if (t.joinable()) t.join();
        }

        // Run water geometry pass offscreen and bind scene textures for post-process
        if (waterEnabled) {
            // GPU frustum cull water meshes (must run outside a render pass)
            sceneRenderer->waterRenderer->getIndirectRenderer().prepareCull(commandBuffer, viewProj);
            // Use 360° solid+sky reflection instead of the sky-only equirect view
            VkImageView cubeReflectionView = VK_NULL_HANDLE;
            if (sceneRenderer && sceneRenderer->solid360Renderer) cubeReflectionView = sceneRenderer->solid360Renderer->getSolid360View();
            VkImageView skyView = (sceneRenderer && sceneRenderer->skyRenderer) ? sceneRenderer->skyRenderer->getSkyView(frameIdx) : VK_NULL_HANDLE;
            // If we launched an async back-face submission, tell waterPass to skip issuing it again.
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 10);
            sceneRenderer->waterPass(this, commandBuffer, frameIdx, getMainDescriptorSet(), settings.wireframeMode,
                mainTime, launchedBackFace, skyView, cubeReflectionView);
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 11);
        }

        // Wait for any async recording/submits to complete their submit calls so semaphores are registered
        for (auto &t : asyncTasks) {
            if (t.joinable()) t.join();
        }

        profileCpuRecord = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - cpuRecordT0).count();
    }

    void renderImGui() override {
        static char sceneFolderBuf[512] = "scenes/my_scene.scene";
        static bool pickerForSave = false;
        static bool pickerOpenPending = false;
        static std::filesystem::path pickerDir = std::filesystem::path("scenes");
        static char pickerNameBuf[256] = "my_scene.scene";
        static std::string pickerError;

        auto copyPathToBuffer = [](const std::filesystem::path& p, char* dst, size_t dstSize) {
            std::string s = p.string();
            if (s.size() >= dstSize) {
                s.resize(dstSize - 1);
            }
            std::memcpy(dst, s.c_str(), s.size());
            dst[s.size()] = '\0';
        };

        auto openScenePicker = [&](bool forSave) {
            pickerForSave = forSave;
            pickerOpenPending = true;
            pickerError.clear();
            std::filesystem::path initialPath(sceneFolderBuf);
            if (initialPath.empty()) {
                initialPath = std::filesystem::path("scenes/my_scene.scene");
            }

            std::error_code ec;
            std::filesystem::path baseDir;
            if (std::filesystem::is_directory(initialPath, ec)) {
                baseDir = initialPath;
            } else {
                baseDir = initialPath.has_parent_path() ? initialPath.parent_path() : std::filesystem::path("scenes");
            }
            if (baseDir.empty()) {
                baseDir = std::filesystem::current_path(ec);
            }
            pickerDir = std::filesystem::absolute(baseDir, ec);
            if (ec) {
                pickerDir = baseDir;
            }

            std::string filename = initialPath.filename().string();
            if (filename.empty()) {
                filename = "my_scene.scene";
            }
            if (filename.size() >= sizeof(pickerNameBuf)) {
                filename.resize(sizeof(pickerNameBuf) - 1);
            }
            std::memcpy(pickerNameBuf, filename.c_str(), filename.size());
            pickerNameBuf[filename.size()] = '\0';
        };

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Generate Map")) generateMapPending = true;
                ImGui::Separator();
                if (ImGui::MenuItem("Save Scene...")) openScenePicker(true);
                if (ImGui::MenuItem("Load Scene...")) openScenePicker(false);
                ImGui::Separator();
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

            ImVec2 center = ImGui::GetMainViewport()->GetCenter();

            ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(ImVec2(700.0f, 420.0f), ImGuiCond_Appearing);
            if (pickerOpenPending) {
                ImGui::OpenPopup("Scene File Picker");
                pickerOpenPending = false;
            }
            if (ImGui::BeginPopupModal("Scene File Picker", NULL, ImGuiWindowFlags_NoCollapse)) {
                std::error_code ec;
                if (!std::filesystem::exists(pickerDir, ec) || !std::filesystem::is_directory(pickerDir, ec)) {
                    pickerDir = std::filesystem::current_path(ec);
                }

                ImGui::Text("Mode: %s", pickerForSave ? "Save" : "Load");
                ImGui::TextWrapped("Current directory: %s", pickerDir.string().c_str());

                if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf062##scene_picker_up")) && pickerDir.has_parent_path()) {
                    pickerDir = pickerDir.parent_path();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Up one folder");
                }
                ImGui::SameLine();
                if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf0ac##scene_picker_root"))) {
                    pickerDir = std::filesystem::path("/");
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Go to root (/)");
                }
                ImGui::SameLine();
                if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf015##scene_picker_home"))) {
                    const char* home = std::getenv("HOME");
                    if (home && *home) {
                        pickerDir = std::filesystem::path(home);
                    }
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Go to home folder");
                }
                ImGui::SameLine();
                if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf07c##scene_picker_home_scenes"))) {
                    pickerDir = std::filesystem::path("scenes");
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Go to project scenes folder");
                }

                std::vector<std::filesystem::directory_entry> dirs;
                std::vector<std::filesystem::directory_entry> files;
                for (const auto& entry : std::filesystem::directory_iterator(pickerDir, ec)) {
                    if (ec) {
                        break;
                    }
                    if (entry.is_directory(ec)) {
                        dirs.push_back(entry);
                    } else if (entry.is_regular_file(ec)) {
                        files.push_back(entry);
                    }
                }

                auto byName = [](const std::filesystem::directory_entry& a, const std::filesystem::directory_entry& b) {
                    return a.path().filename().string() < b.path().filename().string();
                };
                std::sort(dirs.begin(), dirs.end(), byName);
                std::sort(files.begin(), files.end(), byName);

                auto executePickerSelection = [&]() {
                    std::filesystem::path selected = pickerDir / pickerNameBuf;
                    if (selected.extension() != ".scene") {
                        selected += ".scene";
                    }

                    if (pickerForSave || std::filesystem::exists(selected, ec)) {
                        copyPathToBuffer(selected, sceneFolderBuf, sizeof(sceneFolderBuf));
                        if (pickerForSave) {
                            if (mainScene) {
                                mainScene->save(sceneFolderBuf, &settings);
                            }
                        } else {
                            pendingLoadPath = sceneFolderBuf;
                            loadScenePending = true;
                        }
                        pickerError.clear();
                        ImGui::CloseCurrentPopup();
                    } else {
                        pickerError = "Selected file does not exist.";
                    }
                };

                ImGui::BeginChild("##picker_entries", ImVec2(0.0f, 260.0f), true);
                for (const auto& d : dirs) {
                    std::string label = "[DIR] " + d.path().filename().string();
                    if (ImGui::Selectable(label.c_str(), false)) {
                        pickerDir = d.path();
                    }
                }
                for (const auto& f : files) {
                    const std::string name = f.path().filename().string();
                    const bool isSceneFile = f.path().extension() == ".scene";
                    if (!pickerForSave && !isSceneFile) {
                        continue;
                    }
                    std::string label = name;
                    if (ImGui::Selectable(label.c_str(), false)) {
                        std::string clipped = name;
                        if (clipped.size() >= sizeof(pickerNameBuf)) {
                            clipped.resize(sizeof(pickerNameBuf) - 1);
                        }
                        std::memcpy(pickerNameBuf, clipped.c_str(), clipped.size());
                        pickerNameBuf[clipped.size()] = '\0';
                    }
                    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        executePickerSelection();
                    }
                }
                ImGui::EndChild();

                ImGui::Text("File name:");
                ImGui::InputText("##picker_name", pickerNameBuf, sizeof(pickerNameBuf));

                if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf00c##scene_picker_select"))) {
                    executePickerSelection();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Select file");
                }
                ImGui::SameLine();
                if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf00d##scene_picker_cancel"))) {
                    pickerError.clear();
                    ImGui::CloseCurrentPopup();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Cancel");
                }

                if (!pickerError.empty()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", pickerError.c_str());
                }

                ImGui::EndPopup();
            }

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
                    float gpuTotal = profileShadow + profileMainCull + profileSky +
                                     profileSolidDraw + profileVegetation + profileWater + profileImGui;
                    ImGui::Text("Shadow:        %.2f", profileShadow);
                    ImGui::Text("Main Cull:     %.2f", profileMainCull);
                    ImGui::Text("Sky:           %.2f", profileSky);
                    ImGui::Text("Solid Draw:    %.2f", profileSolidDraw);
                    ImGui::Text("Vegetation:    %.2f", profileVegetation);
                    ImGui::Text("Water:         %.2f", profileWater);
                    ImGui::Text("ImGui:         %.2f", profileImGui);
                    ImGui::Text("GPU Total:     %.2f", gpuTotal);
                    ImGui::Separator();
                    ImGui::Text("--- CPU Timing (ms) ---");
                    ImGui::Text("FPS:           %.1f", profileFps);
                    ImGui::Text("Update:        %.2f", profileCpuUpdate);
                    ImGui::Text("Record:        %.2f", profileCpuRecord);
                }

                ImGui::End();
            }

            // Small top-right overlay under the main menu bar showing gamepad connection
            {
                ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
                ImGui::SetNextWindowBgAlpha(0.35f);
                float padding = 10.0f;
                float y = ImGui::GetFrameHeight() + 6.0f; // position just under the main menu bar
                ImVec2 disp = ImGui::GetIO().DisplaySize;
                // anchor by top-right using pivot (1,0)
                ImGui::SetNextWindowPos(ImVec2(disp.x - padding, y), ImGuiCond_Always, ImVec2(1.0f, 0.0f));

                ImGui::Begin("GamepadOverlay", nullptr, flags);
                bool gamepadConnected = false;
                for (int jid = GLFW_JOYSTICK_1; jid <= GLFW_JOYSTICK_LAST; ++jid) {
                    if (glfwJoystickIsGamepad(jid)) { gamepadConnected = true; break; }
                }

                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 cursor = ImGui::GetCursorScreenPos();
                float iconSize = ImGui::GetFrameHeight() * 0.6f;
                ImVec2 center = ImVec2(cursor.x + iconSize * 0.5f, cursor.y + iconSize * 0.5f);
                ImU32 color = gamepadConnected ? IM_COL32(40,200,40,255) : IM_COL32(200,40,40,255);
                draw_list->AddCircleFilled(center, iconSize * 0.4f, color);
                ImGui::Dummy(ImVec2(iconSize, iconSize));
                ImGui::SameLine();
                ImGui::Text("%s", gamepadConnected ? "Gamepad" : "No Gamepad");
                if (ImGui::IsItemHovered() || ImGui::IsWindowHovered()) ImGui::SetTooltip("Gamepad %s", gamepadConnected ? "connected" : "not connected");

                // Draw controller page indicator (small 3-letter icon) below the gamepad status
                ControllerParameters* cp = controllerManager.getParameters();
                if (cp) {
                    ImGui::Separator();
                    ImDrawList* dl2 = ImGui::GetWindowDrawList();
                    ImVec2 pos = ImGui::GetCursorScreenPos();
                    // Reserve space and draw a small colored circle then the label (same style as GamepadWidget)
                    float pageIconHeight = ImGui::GetFrameHeight();
                    ImGui::Dummy(ImVec2(36.0f, pageIconHeight));
                    // center vertically inside the reserved dummy
                    ImVec2 center2 = ImVec2(pos.x + 10.0f, pos.y + pageIconHeight * 0.5f);
                    ImVec4 col = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
                    const char* label = "?";
                    switch (cp->currentPage) {
                        case ControllerParameters::CAMERA: label = "CAM"; col = ImVec4(0.2f,0.5f,0.9f,1.0f); break;
                        case ControllerParameters::BRUSH_POSITION: label = "POS"; col = ImVec4(0.2f,0.9f,0.3f,1.0f); break;
                        case ControllerParameters::BRUSH_SCALE: label = "SCL"; col = ImVec4(0.95f,0.6f,0.1f,1.0f); break;
                        case ControllerParameters::BRUSH_ROTATION: label = "ROT"; col = ImVec4(0.7f,0.3f,0.9f,1.0f); break;
                        case ControllerParameters::BRUSH_PROPERTIES: label = "PRP"; col = ImVec4(0.6f,0.6f,0.6f,1.0f); break;
                    }
                    dl2->AddCircleFilled(center2, 8.0f, ImGui::ColorConvertFloat4ToU32(col));
                    float textY = pos.y + (pageIconHeight - ImGui::GetFontSize()) * 0.5f;
                    dl2->AddText(ImVec2(pos.x + 22.0f, textY), ImGui::ColorConvertFloat4ToU32(ImVec4(1,1,1,1)), label);
                    if (ImGui::IsItemHovered() || ImGui::IsWindowHovered()) ImGui::SetTooltip("%s", ControllerParameters::pageTypeName(cp->currentPage));
                }

                ImGui::End();
            }
        }

        if (imguiShowDemo) ImGui::ShowDemoWindow(&imguiShowDemo);

        cubeCount = sceneRenderer ? sceneRenderer->getRegisteredModelCount() : 0;

        // Update per-frame widget state (avoid storing VulkanApp* inside widgets)
        if (renderTargetsWidget) renderTargetsWidget->setFrameInfo(getCurrentFrame(), getWidth(), getHeight());
        if (vulkanResourcesManagerWidget) vulkanResourcesManagerWidget->updateWithApp(this);

        // Render all widgets
        widgetManager.renderAll();
    }

    void draw(VkCommandBuffer &commandBuffer) override {
        // Only record draw commands; command buffer and render pass are already active
        if (commandBuffer == VK_NULL_HANDLE) {
            std::cerr << "[MyApp::draw] Error: commandBuffer is VK_NULL_HANDLE, skipping draw." << std::endl;
            return;
        }
        if (!sceneRenderer) {
            std::cerr << "[MyApp::draw] Error: sceneRenderer is nullptr, skipping draw." << std::endl;
            return;
        }
        if (!mainScene) {
            std::cerr << "[MyApp::draw] Error: mainScene is nullptr, skipping draw." << std::endl;
            return;
        }

        // --- SAFETY: Ensure indirect buffers are rebuilt if dirty before first draw ---
        if (sceneRenderer->solidRenderer->getIndirectRenderer().isDirty()) {
            // printf("[MyApp::draw] solidRenderer indirect buffer dirty, rebuilding before draw...\\n");
            sceneRenderer->solidRenderer->getIndirectRenderer().rebuild(this);
        }
        if (sceneRenderer->waterRenderer->getIndirectRenderer().isDirty()) {
            // printf("[MyApp::draw] waterRenderer indirect buffer dirty, rebuilding before draw...\\n");
            sceneRenderer->waterRenderer->getIndirectRenderer().rebuild(this);
        }

        uint32_t frameIdx = getCurrentFrame();
        glm::mat4 viewProj = camera.getViewProjectionMatrix();
        glm::mat4 invViewProj = glm::inverse(viewProj);

        // Composite offscreen scene + water into the swapchain
        if (sceneRenderer && sceneRenderer->postProcessRenderer) {
            VkImageView skyViewPP = sceneRenderer->skyRenderer ? sceneRenderer->skyRenderer->getSkyView(frameIdx) : VK_NULL_HANDLE;
            sceneRenderer->postProcessRenderer->render(
                this,
                commandBuffer,
                sceneRenderer->solidRenderer->getColorView(frameIdx),
                sceneRenderer->solidRenderer->getDepthView(frameIdx),
                sceneRenderer->waterRenderer->getWaterDepthView(frameIdx),
                viewProj,
                invViewProj,
                glm::vec3(uboStatic.viewPos),
                frameIdx,
                skyViewPP);
        }
        //std::cout << "[MyApp::draw] waterPass returned. Rendering ImGui..." << std::endl;
        // ImGui rendering
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (!draw_data) {
            std::cerr << "[MyApp::draw] Warning: ImGui::GetDrawData() returned nullptr, skipping ImGui rendering." << std::endl;
        } else if (commandBuffer == VK_NULL_HANDLE) {
            std::cerr << "[MyApp::draw] Error: commandBuffer is VK_NULL_HANDLE before ImGui rendering, skipping ImGui." << std::endl;
        } else {
            //std::cerr << "[MyApp::draw] ImGui::GetDrawData() valid, calling ImGui_ImplVulkan_RenderDrawData..." << std::endl;
            VkQueryPool qp = queryPools[getCurrentFrame()];
            if (profilingEnabled && qp != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 12);
            ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);
            if (profilingEnabled && qp != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 13);
        }
    }

    void clean() override {
    // Join tessellation thread before tearing down Vulkan resources.
    if (sceneProcessThread.joinable()) sceneProcessThread.join();


        // Free all ImGui descriptor sets owned by the widget while ImGui is still
        // alive. clean() is called before cleanupImGui(), so this is safe.
        // Without this, ~RenderTargetsWidget() (called from ~MyApp() after Vulkan
        // teardown) would try to free descriptors from a destroyed pool.
        if (renderTargetsWidget) {
            renderTargetsWidget->invalidateImGuiDescriptors();
        }

        if (impostorWidget) {
            impostorWidget->cleanup();
        }

        // Cleanup scene renderer and all sub-renderers
        if (sceneRenderer) {
            sceneRenderer->cleanup(this);
        }

        // Destroy timestamp query pools
        for (uint32_t f = 0; f < 2; ++f) {
            if (queryPools[f] != VK_NULL_HANDLE) {
                vkDestroyQueryPool(getDevice(), queryPools[f], nullptr);
                queryPools[f] = VK_NULL_HANDLE;
            }
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

    void onImGuiRecreated() override {
        // Re-create ImGui AddTexture DS for shadow cascades — the old ones used the
        // previous DescriptorSetLayout which was destroyed by ImGui_ImplVulkan_Shutdown.
        if (sceneRenderer && sceneRenderer->shadowMapper) {
            sceneRenderer->shadowMapper->recreateImGuiDescriptors();
        }
        // Invalidate all cached ImGui texture DS in the RenderTargetsWidget so they
        // are re-created with the new DSL on the next frame.
        if (renderTargetsWidget) {
            renderTargetsWidget->invalidateImGuiDescriptors();
        }
    }

    void onEvent(const EventPtr &event) override {
        if (auto closeEvent = std::dynamic_pointer_cast<CloseWindowEvent>(event)) {
            requestClose();
            return;
        }
        if (auto fullscreenEvent = std::dynamic_pointer_cast<ToggleFullscreenEvent>(event)) {
            toggleFullscreen();
            return;
        }
        if (auto rebuildEvent = std::dynamic_pointer_cast<RebuildBrushEvent>(event)) {
            // Defer heavy rebuild to postSubmit() to avoid interfering with
            // command buffer recording and GPU fences.
            brushRebuildPending = true;
            return;
        }
    }
    // Called by VulkanApp after a frame has been submitted
    void postSubmit() override;
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
    // Scene objects, background thread, and brush3dWidget are now set up
    // directly in setup() so the CPU-heavy scene load can run in parallel
    // with texture loading. This stub is kept for call-site compatibility.
}

// Implementation: setup vegetation textures
void MyApp::setupVegetationTextures() {
    // Allocate 3-layer texture arrays for vegetation (foliage, grass, wild)
    vegetationTextureArrayManager.allocate(3, 512, 512, this);
    vegetationAtlasEditor = std::make_shared<VegetationAtlasEditor>(&vegetationTextureArrayManager, &vegetationAtlasManager);
    billboardCreator = std::make_shared<BillboardCreator>(&billboardManager, &vegetationAtlasManager, &vegetationTextureArrayManager);
    // Provide VulkanApp to the creator so it can initialize GPU-backed preview textures
    billboardCreator->setVulkanApp(this);
    impostorWidget = std::make_shared<ImpostorWidget>();
    impostorWidget->setVulkanApp(this);
    impostorWidget->setVegetationRenderer(sceneRenderer->vegetationRenderer.get());
    
    // Load the vegetation atlas textures (albedo, normal, opacity) into the texture array
    std::vector<TextureTriple> vegTriples = {
        { "textures/vegetation/foliage_color.jpg", "textures/vegetation/foliage_normal.jpg", "textures/vegetation/foliage_opacity.jpg" },
        { "textures/vegetation/grass_color.jpg",   "textures/vegetation/grass_normal.jpg",   "textures/vegetation/grass_opacity.jpg" },
        { "textures/vegetation/wild_color.jpg",    "textures/vegetation/wild_normal.jpg",    "textures/vegetation/wild_opacity.jpg" }
    };
    size_t loaded = vegetationTextureArrayManager.loadTriples(this, vegTriples);
    std::cerr << "[MyApp::setupVegetationTextures] Loaded " << loaded << " vegetation texture layers" << std::endl;

    // Auto-detect atlas tiles from opacity maps and populate AtlasManager for each texture
    const char* opacityPaths[3] = { "textures/vegetation/foliage_opacity.jpg", "textures/vegetation/grass_opacity.jpg", "textures/vegetation/wild_opacity.jpg" };
    for (int atlasIndex = 0; atlasIndex < 3; ++atlasIndex) {
        try {
            vegetationAtlasManager.clear(atlasIndex);
            int added = vegetationAtlasManager.autoDetectTiles(atlasIndex, opacityPaths[atlasIndex]);
            std::cerr << "[MyApp::setupVegetationTextures] Atlas " << atlasIndex << ": auto-detected " << added << " tiles" << std::endl;
        } catch (...) {
            std::cerr << "[MyApp::setupVegetationTextures] Atlas " << atlasIndex << ": autoDetectTiles failed" << std::endl;
        }
    }

    // Initialize the editor with 3 vegetation billboards.
    // Each billboard uses only one atlas (its respective texture), and includes
    // all available tiles from that atlas as layers with uniform horizontal offsets.
    billboardManager.clear();

    for (int atlasIndex = 0; atlasIndex < 3; ++atlasIndex) {
        const std::string name = "Vegetation Billboard " + std::to_string(atlasIndex + 1);
        const size_t bidx = billboardManager.createBillboard(name);

        Billboard* billboard = billboardManager.getBillboard(bidx);
        if (billboard) {
            billboard->width = 1.0f;
            billboard->height = 1.5f;
        }

        const size_t tileCount = vegetationAtlasManager.getTileCount(atlasIndex);
        if (tileCount == 0) continue;

        // Repeat tiles until there are at least minSlots to fill the full width.
        const size_t minSlots   = 5;
        const size_t repeats    = (minSlots + tileCount - 1) / tileCount;
        const size_t totalSlots = repeats * tileCount;

        for (size_t slot = 0; slot < totalSlots; ++slot) {
            const size_t tileIndex = slot % tileCount;
            const AtlasTile* tile  = vegetationAtlasManager.getTile(atlasIndex, static_cast<int>(tileIndex));

            BillboardLayer layer;
            layer.atlasIndex = atlasIndex;
            layer.tileIndex  = static_cast<int>(tileIndex);
            // Distribute slots uniformly left-to-right in normalised [-1,+1] space.
            // Centre of slot s of n: (2s+1)/n − 1
            layer.offsetX = (2.0f * static_cast<float>(slot) + 1.0f)
                            / static_cast<float>(totalSlots) - 1.0f;
            layer.offsetY = 0.0f;
            // Scale each tile to exactly fill its 1/totalSlots slot of billboard width.
            // In compositeLayer: visible_width = 2 * scaleX * tile->scaleX, desired = 2/totalSlots.
            layer.scaleX = (tile && tile->scaleX > 1e-6f)
                           ? 1.0f / (tile->scaleX * static_cast<float>(totalSlots))
                           : 1.0f;
            // Scale each tile to fill the full billboard height (100%).
            // denomY = scaleY * tile->scaleY; setting scaleY = 1/tile->scaleY → denomY = 1.0,
            // so the full [-1,+1] output range maps to the full tile height.
            layer.scaleY = (tile && tile->scaleY > 1e-6f)
                           ? 1.0f / tile->scaleY
                           : 1.0f;
            layer.rotation = 0.0f;
            layer.opacity  = 1.0f;
            layer.renderOrder = static_cast<int>(slot);

            billboardManager.addLayer(bidx, layer);
        }
    }

    if (billboardCreator) {
        // Bake authoring billboards into dedicated per-billboard GPU textures.
        billboardCreator->bakeAllBillboards();
    }

    // Notify ImpostorWidget about the freshly baked texture arrays.
    if (impostorWidget && billboardCreator) {
        impostorWidget->setSource(
            billboardCreator->getAlbedoArrayView(),
            billboardCreator->getNormalArrayView(),
            billboardCreator->getOpacityArrayView(),
            billboardCreator->getArraySampler(),
            static_cast<int>(billboardManager.getBillboardCount()));
    }
}

// Implementation: rebuild the brush scene from Brush3dWidget entries
void MyApp::rebuildBrushScene() {
    if (!brushScene || !sceneRenderer || !brush3dWidget) return;

    // Process only the currently-selected brush entry from the manager
    const BrushEntry* selectedEntry = brushManager.getSelectedEntry();
    size_t selCount = selectedEntry ? 1 : 0;
    std::cerr << "[MyApp::rebuildBrushScene] Rebuilding with " << selCount << " selected entries" << std::endl;

    // 1. Remove all existing brush meshes from GPU
    sceneRenderer->clearBrushMeshes();

    // 2. Reset the brush octrees (clears spatial data without change events)
    brushScene->getOpaqueOctree().reset();
    brushScene->transparentOctree.reset();

    if (!selectedEntry) {
        // Nothing to add — just rebuild to commit the removals
        sceneRenderer->solidRenderer->getIndirectRenderer().setDirty(true);
        sceneRenderer->solidRenderer->getIndirectRenderer().rebuild(this);
        sceneRenderer->waterRenderer->getIndirectRenderer().setDirty(true);
        sceneRenderer->waterRenderer->getIndirectRenderer().rebuild(this);
        return;
    }

    // 3. Create brush change handlers (use separate brush chunk maps)
    SolidSpaceChangeHandler brushSolidHandler = sceneRenderer->makeBrushSolidSpaceChangeHandler(brushScene, this);
    LiquidSpaceChangeHandler brushLiquidHandler = sceneRenderer->makeBrushLiquidSpaceChangeHandler(brushScene, this);
    UniqueOctreeChangeHandler uniqueBrushSolidHandler = UniqueOctreeChangeHandler(brushSolidHandler);
    UniqueOctreeChangeHandler uniqueBrushLiquidHandler = UniqueOctreeChangeHandler(brushLiquidHandler);

    Simplifier simplifier(0.99f, 0.1f, true);
    glm::vec4 translate(0.0f);
    glm::vec4 scale(1.0f);

    // 4. Process the selected brush entry only
    const auto& entry = *selectedEntry;
        // Select the target octree and handler based on targetLayer
        Octree& octree = (entry.targetLayer == 0)
            ? brushScene->getOpaqueOctree()
            : brushScene->transparentOctree;
        const OctreeChangeHandler& handler = (entry.targetLayer == 0)
            ? static_cast<const OctreeChangeHandler&>(uniqueBrushSolidHandler)
            : static_cast<const OctreeChangeHandler&>(uniqueBrushLiquidHandler);

        Transformation model(entry.scale, entry.translate, entry.yaw, entry.pitch, entry.roll);
        SimpleBrush brush(entry.materialIndex);

        // Create the base SDF primitive (stack-allocated, octree copies during add)
        // sdfType: 0=Sphere,1=Box,2=Capsule,3=Octahedron,4=Pyramid,5=Torus,6=Cone,7=Cylinder
        // We use a lambda to avoid massive switch duplication for add vs del with optional effects
        auto applyEntry = [&](WrappedSignedDistanceFunction* wrappedFunc) {
            // Optionally wrap in an effect
            // effectType: 0=PerlinDistort, 1=PerlinCarve, 2=SineDistort, 3=VoronoiCarve
            if (entry.useEffect) {
                switch (entry.effectType) {
                    case 0: {
                        WrappedPerlinDistortDistanceEffect effect(wrappedFunc,
                            entry.effectAmplitude, entry.effectFrequency,
                            glm::vec3(0), entry.effectBrightness, entry.effectContrast);
                        if (entry.brushMode == 0)
                            octree.add(&effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        else
                            octree.del(&effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        break;
                    }
                    case 1: {
                        WrappedPerlinCarveDistanceEffect effect(wrappedFunc,
                            entry.effectAmplitude, entry.effectFrequency, entry.effectThreshold,
                            glm::vec3(0), entry.effectBrightness, entry.effectContrast);
                        if (entry.brushMode == 0)
                            octree.add(&effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        else
                            octree.del(&effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        break;
                    }
                    case 2: {
                        WrappedSineDistortDistanceEffect effect(wrappedFunc,
                            entry.effectAmplitude, entry.effectFrequency, glm::vec3(0));
                        if (entry.brushMode == 0)
                            octree.add(&effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        else
                            octree.del(&effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        break;
                    }
                    case 3: {
                        WrappedVoronoiCarveDistanceEffect effect(wrappedFunc,
                            entry.effectAmplitude, entry.effectCellSize,
                            glm::vec3(0), entry.effectBrightness, entry.effectContrast);
                        if (entry.brushMode == 0)
                            octree.add(&effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        else
                            octree.del(&effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        break;
                    }
                    default: {
                        // Fallback: no effect
                        if (entry.brushMode == 0)
                            octree.add(wrappedFunc, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        else
                            octree.del(wrappedFunc, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        break;
                    }
                }
            } else {
                // No effect — use the primitive directly
                if (entry.brushMode == 0)
                    octree.add(wrappedFunc, model, translate, scale, brush, entry.minSize, simplifier, handler);
                else
                    octree.del(wrappedFunc, model, translate, scale, brush, entry.minSize, simplifier, handler);
            }
        };

        switch (entry.sdfType) {
            case 0: { // Sphere
                SphereDistanceFunction fn;
                WrappedSphere wrapped(&fn);
                applyEntry(&wrapped);
                break;
            }
            case 1: { // Box
                BoxDistanceFunction fn;
                WrappedBox wrapped(&fn);
                applyEntry(&wrapped);
                break;
            }
            case 2: { // Capsule
                CapsuleDistanceFunction fn(entry.capsuleA, entry.capsuleB, entry.capsuleRadius);
                WrappedCapsule wrapped(&fn);
                applyEntry(&wrapped);
                break;
            }
            case 3: { // Octahedron
                OctahedronDistanceFunction fn;
                WrappedOctahedron wrapped(&fn);
                applyEntry(&wrapped);
                break;
            }
            case 4: { // Pyramid
                PyramidDistanceFunction fn;
                WrappedPyramid wrapped(&fn);
                applyEntry(&wrapped);
                break;
            }
            case 5: { // Torus
                TorusDistanceFunction fn(entry.torusRadii);
                WrappedTorus wrapped(&fn);
                applyEntry(&wrapped);
                break;
            }
            case 6: { // Cone
                ConeDistanceFunction fn;
                WrappedCone wrapped(&fn);
                applyEntry(&wrapped);
                break;
            }
            case 7: { // Cylinder
                CylinderDistanceFunction fn;
                WrappedCylinder wrapped(&fn);
                applyEntry(&wrapped);
                break;
            }
            default:
                std::cerr << "[rebuildBrushScene] Unknown sdfType " << entry.sdfType << ", skipping" << std::endl;
                break;
            }

    // 5. Flush queued change events (triggers mesh creation via SceneRenderer)
    uniqueBrushSolidHandler.handleEvents();
    uniqueBrushLiquidHandler.handleEvents();

    // 6. Rebuild indirect buffers so new brush meshes appear
    sceneRenderer->solidRenderer->getIndirectRenderer().setDirty(true);
    sceneRenderer->solidRenderer->getIndirectRenderer().rebuild(this);
    sceneRenderer->waterRenderer->getIndirectRenderer().setDirty(true);
    sceneRenderer->waterRenderer->getIndirectRenderer().rebuild(this);

    // std::cerr << "[MyApp::rebuildBrushScene] Done — brush opaque chunks: " << sceneRenderer->brushSolidChunks.size()
    //           << ", brush transparent chunks: " << sceneRenderer->brushTransparentChunks.size() << std::endl;
}

// Ensure pending texture generation requests are flushed after a frame is submitted
// so array-layer transitions happen outside of active draw command buffers.

void MyApp::generateMap() {
    // Join any previous background tessellation thread
    if (sceneProcessThread.joinable()) sceneProcessThread.join();

    // Wait for the GPU to finish all in-flight work before clearing GPU resources
    deviceWaitIdle();

    // Remove all existing GPU-side meshes
    if (sceneRenderer) {
        sceneRenderer->removeAllRegisteredMeshes();
        sceneRenderer->removeAllTransparentMeshes();
        sceneRenderer->nodeDebugCubes.clear();
    }

    // Reset both octrees (frees all node memory, keeps bounds)
    mainScene->getOpaqueOctree().reset();
    mainScene->transparentOctree.reset();

    // Discard any stale pending change events
    sceneUniqueSolidHandler->clear();
    sceneUniqueLiquidHandler->clear();

    // Reset octree explorer ready flag
    if (octreeExplorerWidget)
        octreeExplorerWidget->octreeReady.store(false, std::memory_order_release);

    // Build the octree (CPU only, no tessellation)
    MainSceneLoader loader;
    mainScene->loadScene(loader, *sceneUniqueSolidHandler, *sceneUniqueLiquidHandler);
    std::cout << "[MyApp::generateMap] Octree construction complete\n";

    // Tessellate chunks in a background thread
    sceneProcessThread = std::thread([this]() {
        sceneUniqueSolidHandler->handleEvents();
        sceneUniqueLiquidHandler->handleEvents();
        if (octreeExplorerWidget)
            octreeExplorerWidget->octreeReady.store(true, std::memory_order_release);
        std::cout << "[MyApp::generateMap] Scene chunk tessellation complete\n";
    });
}

void MyApp::loadSceneFromFile(const std::string& path) {
    if (sceneProcessThread.joinable()) sceneProcessThread.join();

    deviceWaitIdle();

    if (sceneRenderer) {
        sceneRenderer->removeAllRegisteredMeshes();
        sceneRenderer->removeAllTransparentMeshes();
        sceneRenderer->nodeDebugCubes.clear();
    }

    mainScene->getOpaqueOctree().reset();
    mainScene->transparentOctree.reset();

    sceneUniqueSolidHandler->clear();
    sceneUniqueLiquidHandler->clear();

    if (octreeExplorerWidget)
        octreeExplorerWidget->octreeReady.store(false, std::memory_order_release);

    mainScene->load(path, *sceneUniqueSolidHandler, *sceneUniqueLiquidHandler, &settings);
    std::cout << "[MyApp::loadSceneFromFile] Octree loaded from '" << path << "'\n";

    sceneProcessThread = std::thread([this]() {
        sceneUniqueSolidHandler->handleEvents();
        sceneUniqueLiquidHandler->handleEvents();
        if (octreeExplorerWidget)
            octreeExplorerWidget->octreeReady.store(true, std::memory_order_release);
        std::cout << "[MyApp::loadSceneFromFile] Scene tessellation complete\n";
    });
}

void MyApp::postSubmit() {
    if (textureMixer) {
        textureMixer->flushPendingRequests(this);
        textureMixer->pollPendingGenerations(this);
    }

    // If a rebuild was requested from the UI (Apply Brush), perform it now
    // after the frame was submitted so GPU fences can be waited on safely.
    if (brushRebuildPending) {
        brushRebuildPending = false;
        rebuildBrushScene();
    }

    if (generateMapPending) {
        generateMapPending = false;
        generateMap();
    }

    if (loadScenePending) {
        loadScenePending = false;
        loadSceneFromFile(pendingLoadPath);
    }
}


