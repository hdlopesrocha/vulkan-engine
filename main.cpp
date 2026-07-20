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
#include <sys/resource.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "vulkan/ubo/UniformObject.hpp"
#include "vulkan/ubo/SkyUniform.hpp"
#include "vulkan/VulkanApp.hpp"
#include "vulkan/renderer/SceneRenderer.hpp"
#include "vulkan/renderer/RendererUtils.hpp"
#include "utils/LocalScene.hpp"
#include "widgets/SettingsWidget.hpp"
#include "widgets/SkyWidget.hpp"
#include "widgets/SkySettings.hpp"
#include "widgets/WaterWidget.hpp"
#include "widgets/RenderTargetsWidget.hpp"
#include "widgets/BillboardCreator.hpp"
#include "widgets/ImpostorWidget.hpp"
#include "services/ImpostorService.hpp"
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
#include "widgets/components/FilePicker.hpp"
#include "utils/MainSceneLoader.hpp"
#include "utils/Settings.hpp"
#include "widgets/WidgetManager.hpp"
#include "math/Camera.hpp"
#include "math/Light.hpp"
#include "events/EventManager.hpp"
#include "events/KeyboardPublisher.hpp"
#include "events/GamepadPublisher.hpp"
#include "events/NunchukPublisher.hpp"
#include "events/CloseWindowEvent.hpp"
#include "events/ToggleFullscreenEvent.hpp"
#include "events/RebuildBrushEvent.hpp"
#include "vulkan/TextureArrayManager.hpp"
#include "vulkan/MaterialManager.hpp"
#include "vulkan/renderer/DescriptorWriter.hpp"
#include "utils/BillboardManager.hpp"
#include "utils/AtlasManager.hpp"
#include "services/TextureMixer.hpp"
#include "services/BillboardService.hpp"
#include "utils/ShadowParams.hpp"
#include "space/ThreadPool.hpp"

class MyApp : public VulkanApp, public IEventHandler {
public:
    Settings settings;
    SceneRenderer * sceneRenderer = nullptr;
    LocalScene * mainScene;
    LocalScene * brushScene = nullptr;
    std::shared_ptr<Brush3dWidget> brush3dWidget;
    // Shared brush entries edited by Brush3dWidget (owned by MyApp)
    Brush3dManager brushManager;
    static constexpr uint32_t QUERY_COUNT = 18; // 9 intervals × 2 timestamps each
    std::array<VkQueryPool, 3> queryPools = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    bool queryPoolReady[2] = {false, false};
    float timestampPeriod = 0.0f;
    bool profilingEnabled = true;
    float profileShadow = 0.0f;
    float profileMainCull = 0.0f;
    float profileSky = 0.0f;
    float profileSolidDraw = 0.0f;
    float profileVegetationReal = 0.0f;
    float profileVegetationImpostor = 0.0f;
    float profileWater = 0.0f;
    float profileImGui = 0.0f;
    float profileSolid360 = 0.0f;
    float profileBackface = 0.0f;
    float profileCpuUpdate = 0.0f;
    float profileCpuRecord = 0.0f;
    float profileFps = 0.0f;
    VkDescriptorSet shadowPassDescriptorSet = VK_NULL_HANDLE;
    UniformObject uboStatic = {};
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    // Track async task fences so we can defer buffer modifications
    // until GPU work from previous-frame async tasks completes.
    std::vector<VkFence> pendingAsyncFences;
    std::mutex asyncFenceMutex;
    std::shared_ptr<SettingsWidget> settingsWidget;
    std::shared_ptr<SkyWidget> skyWidget;
    std::shared_ptr<WaterWidget> waterWidget;
    std::shared_ptr<RenderTargetsWidget> renderTargetsWidget;
    std::shared_ptr<BillboardCreator> billboardCreator;
    std::shared_ptr<ImpostorService> impostorService;
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
    FilePicker scenePicker_{"Scene File Picker", ".scene"};
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
    // Accumulated time driving the animated brush-space sphere trajectory
    float brushAnimTime = 0.0f;
    // Last frame delta, forwarded to postSubmit for the per-frame brush rebuild
    float lastFrameDelta = 0.0f;
    ShadowParams shadowParams;
    // When user clicks "Apply Brush" from ImGui we defer the heavy rebuild
    // until after the current frame is submitted to avoid waiting on fences
    // while the frame is being recorded (causes deadlock). Set by UI,
    // consumed in `postSubmit()`.
    bool brushRebuildPending = false;
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
    NunchukPublisher nunchukPublisher;
    bool sceneLoading = false;

    // Handlers kept alive as members so the tessellation background thread
    // can safely use them after setup() returns.
    std::unique_ptr<SolidSpaceChangeHandler> sceneSolidHandler;
    std::unique_ptr<LiquidSpaceChangeHandler> sceneLiquidHandler;
    std::unique_ptr<UniqueOctreeChangeHandler> sceneUniqueSolidHandler;
    std::unique_ptr<UniqueOctreeChangeHandler> sceneUniqueLiquidHandler;
    std::thread sceneProcessThread; // tessellates chunks after octree is built

    // Pre-allocated descriptor pool+set rings to avoid per-frame create/destroy
    static constexpr uint32_t ASYNC_RING_SIZE = 3;
    struct PoolSetPair { VkDescriptorPool pool; VkDescriptorSet set; };
    PoolSetPair cachedBackfaceCompute[ASYNC_RING_SIZE]{};
    uint32_t ringBackfaceCompute = 0;
    ThreadPool asyncThreadPool{1}; // single worker for per-frame back-face pass

    // Persistent cubemap resources (used inline on main CB, no async race)
    Buffer cube360UBO{};
    Buffer cube360Compact{};
    Buffer cube360Visible{};
    Buffer cube360WaterCompact{};
    Buffer cube360WaterVisible{};
    VkDescriptorSet cube360GfxDs = VK_NULL_HANDLE;
    VkDescriptorSet cube360ComputeDs = VK_NULL_HANDLE;
    VkDescriptorSet cube360WaterComputeDs = VK_NULL_HANDLE;
    uint32_t cube360TexVersion = 0;

    ~MyApp() {
        if (sceneProcessThread.joinable()) sceneProcessThread.join();
    }

    // setupTextures (defined out-of-line to avoid inline/member-definition issues)
    void setupTextures() {
        uint32_t layerCount = 32;

        textureArrayManager.allocate(layerCount, 1024, 1024, this);
        // Use shared TextureTriple defined in TextureArrayManager.hpp
        const std::vector<TextureTriple> textureTriples = {
            { "textures/Wall_Stone_010_basecolor.jpg", "textures/Wall_Stone_010_normal.jpg", "textures/Wall_Stone_010_height.jpg", "textures/Wall_Stone_010_roughness.jpg", "textures/Wall_Stone_010_ambientOcclusion.jpg" },
            { "textures/Ground_Dirt_007_basecolor.jpg", "textures/Ground_Dirt_007_normal.jpg", "textures/Ground_Dirt_007_height.jpg", "textures/Ground_Dirt_007_roughness.jpg", "textures/Ground_Dirt_007_ambientOcclusion.jpg" },
            { "textures/Dead_leaves_001_COLOR.jpg", "textures/Dead_leaves_001_NRM.jpg", "textures/Dead_leaves_001_DISP.jpg", "textures/Dead_leaves_001_SPEC.jpg", "textures/Dead_leaves_001_OCC.jpg" },
            { "textures/Grass_001_COLOR.jpg", "textures/Grass_001_NORM.jpg", "textures/Grass_001_DISP.jpg", "textures/Grass_001_ROUGH.jpg", "textures/Grass_001_OCC.jpg" },
            { "textures/Lava_005_COLOR.jpg", "textures/Lava_005_NORM.jpg", "textures/Lava_005_DISP.jpg", "textures/Lava_005_ROUGH.jpg", "textures/Lava_005_OCC.jpg" },
            { "textures/Metal_Pattern_008_basecolor.jpg", "textures/Metal_Pattern_008_normal.jpg", "textures/Metal_Pattern_008_height.jpg", "textures/Metal_Pattern_008_roughness.jpg", "textures/Metal_Pattern_008_ambientOcclusion.jpg" },
            { "textures/Linoleum_Floor_001_basecolor.jpg", "textures/Linoleum_Floor_001_normal.jpg", "textures/Linoleum_Floor_001_height.jpg", "textures/Linoleum_Floor_001_roughness.jpg", "textures/Linoleum_Floor_001_ambientOcclusion.jpg" },
            { "textures/Rough_rock_021_COLOR.jpg", "textures/Rough_rock_021_NRM.jpg", "textures/Rough_rock_021_DISP.jpg", "textures/Rough_rock_021_SPEC.jpg", "textures/Rough_rock_021_OCC.jpg" },
            { "textures/Sand_007_basecolor.jpg", "textures/Sand_007_normal.jpg", "textures/Sand_007_height.jpg", "textures/Sand_007_roughness.jpg", "textures/Sand_007_ambientOcclusion.jpg" },
            { "textures/Snow_001_COLOR.jpg", "textures/Snow_001_NORM.jpg", "textures/Snow_001_DISP.jpg", "textures/Snow_001_ROUGH.jpg", "textures/Snow_001_OCC.jpg" },
            { "textures/Sand_002_COLOR.jpg", "textures/Sand_002_NRM.jpg", "textures/Sand_002_DISP.jpg", "textures/Sand_002_SPEC.jpg", "textures/Sand_002_OCC.jpg" },
            { "textures/Bark_001_COLOR.jpg", "textures/Bark_001_NORM.jpg", "textures/Bark_001_DISP.jpg", "textures/Bark_001_ROUGH.jpg", "textures/Bark_001_OCC.jpg" },
            { "textures/Concrete_Blocks_013_basecolor.jpg", "textures/Concrete_Blocks_013_normal.jpg", "textures/Concrete_Blocks_013_height.jpg", "textures/Concrete_Blocks_013_roughness.jpg", "textures/Concrete_Blocks_013_ambientOcclusion.jpg" },
            { "textures/Asphalt_001_COLOR.jpg", "textures/Asphalt_001_NRM.jpg", "textures/Asphalt_001_DISP.jpg", "textures/Asphalt_001_SPEC.jpg", "textures/Asphalt_001_OCC.jpg" },
            { "textures/Stone_Floor_002_COLOR.jpg", "textures/Stone_Floor_002_NORM.jpg", "textures/Stone_Floor_002_DISP.jpg", "textures/Stone_Floor_002_SPEC.jpg", "textures/Stone_Floor_002_OCC.jpg" },
            { "textures/Canyon_Rock_001_COLOR.jpg", "textures/Canyon_Rock_001_NORM.jpg", "textures/Canyon_Rock_001_DISP.jpg", "textures/Canyon_Rock_001_ROUGH.jpg", "textures/Canyon_Rock_001_OCC.jpg" },
            { "textures/Sapphire_001_COLOR.jpg", "textures/Sapphire_001_NORM.jpg", "textures/Sapphire_001_DISP.jpg", "textures/Sapphire_001_ROUGH.jpg", "textures/Sapphire_001_OCC.jpg" },
            { "textures/Rough_rock_006_COLOR.jpg", "textures/Rough_rock_006_NRM.jpg", "textures/Rough_rock_006_DISP.jpg", "textures/Rough_rock_006_SPEC.jpg", "textures/Rough_rock_006_OCC.jpg" },
            { "textures/Crystal_Metal_001_COLOR.jpg", "textures/Crystal_Metal_001_NORM.jpg", "textures/Crystal_Metal_001_DISP.jpg", "textures/Crystal_Metal_001_ROUGH.jpg", "textures/Crystal_Metal_001_OCC.jpg" },
            { "textures/Sci-fi_Armor_001_basecolor.jpg", "textures/Sci-fi_Armor_001_normal.jpg", "textures/Sci-fi_Armor_001_height.jpg", "textures/Sci-fi_Armor_001_roughness.jpg", "textures/Sci-fi_Armor_001_ambientOcclusion.jpg" },
            { "textures/Greeble_Techno_002_basecolor.jpg", "textures/Greeble_Techno_002_normal.jpg", "textures/Greeble_Techno_002_height.jpg", "textures/Greeble_Techno_002_roughness.jpg", "textures/Greeble_Techno_002_ambientOcclusion.jpg" },

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
            textureArrayManager.getImTexture(i, 3);
            textureArrayManager.getImTexture(i, 4);
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
        textureArrayManager.getImTexture(editableLayer, 3);
        textureArrayManager.getImTexture(editableLayer, 4);

        // Generate textures for all configured mixer entries (async submissions tracked by TextureMixer)
        textureMixer->generateInitialTextures(mixerParams);
        textureMixerWidget = std::make_shared<TextureMixerWidget>(textureMixer, mixerParams, "Texture Mixer");
        widgetManager.addWidget(textureMixerWidget);

        size_t materialCount = std::max<size_t>(static_cast<size_t>(loadedTextureLayers), static_cast<size_t>(loadedTextureLayers + 1));
        if (materialCount == 0) {
            materialCount = layerCount ? layerCount : 1u;
        }
        materials.assign(materialCount, MaterialProperties{});
          
        materials[0u].mappingMode = true;
        materials[0u].tessLevel = 5.0f;
        materials[0u].tessMinLevel = 2.0f;
        materials[0u].tessMaxLevel = 16.0f;
        materials[0u].tessHeightScale = 8.0f;
        materials[0u].triplanar = true;
        materials[0u].triplanarScaleU = 0.002f;
        materials[0u].triplanarScaleV = 0.002f;
        materials[0u].invertHeight = true;

        materials[5u].mappingMode = true;
        materials[5u].tessLevel = 5.0f;
        materials[5u].tessMinLevel = 2.0f;
        materials[5u].tessMaxLevel = 16.0f;
        materials[5u].tessHeightScale = 2.0f;
        materials[5u].triplanar = true;
        materials[5u].triplanarScaleU = 0.002f;
        materials[5u].triplanarScaleV = 0.002f;
        materials[5u].invertHeight = true;
        materials[5u].reflectionStrength = 0.8f;

        materials[6u].mappingMode = true;
        materials[6u].tessLevel = 1.0f;
        materials[6u].tessMinLevel = 1.0f;
        materials[6u].tessMaxLevel = 1.0f;
        materials[6u].tessHeightScale = 0.0f;
        materials[6u].triplanar = true;
        materials[6u].triplanarScaleU = 0.002f;
        materials[6u].triplanarScaleV = 0.002f;
        materials[6u].invertHeight = true;
        materials[6u].reflectionStrength = 1.0f;

        materials[7u].mappingMode = true;
        materials[7u].tessLevel = 2.0f;
        materials[7u].tessMinLevel = 2.0f;
        materials[7u].tessMaxLevel = 16.0f;
        materials[7u].tessHeightScale = 8.0f;
        materials[7u].triplanar = true;
        materials[7u].triplanarScaleU = 0.01f;
        materials[7u].triplanarScaleV = 0.01f;
        materials[7u].invertHeight = true;
        materials[7u].reflectionStrength = 0.2f;

        materials[12u].mappingMode = true;
        materials[12u].tessLevel = 6.0f;
        materials[12u].tessMinLevel = 2.0f;
        materials[12u].tessMaxLevel = 16.0f;
        materials[12u].tessHeightScale = 32.0f;
        materials[12u].triplanar = true;
        materials[12u].triplanarScaleU = 0.002f;
        materials[12u].triplanarScaleV = 0.002f;
        materials[12u].invertHeight = true;
        materials[12u].reflectionStrength = 0.3f;

        materials[14u].mappingMode = true;
        materials[14u].tessLevel = 5.0f;
        materials[14u].tessMinLevel = 2.0f;
        materials[14u].tessMaxLevel = 16.0f;
        materials[14u].tessHeightScale = 8.0f;
        materials[14u].triplanar = true;
        materials[14u].triplanarScaleU = 0.002f;
        materials[14u].triplanarScaleV = 0.002f;
        materials[14u].invertHeight = true;

        materials[15u].mappingMode = true;
        materials[15u].tessLevel = 5.0f;
        materials[15u].tessMinLevel = 2.0f;
        materials[15u].tessMaxLevel = 16.0f;
        materials[15u].tessHeightScale = 8.0f;
        materials[15u].triplanar = true;
        materials[15u].triplanarScaleU = 0.002f;
        materials[15u].triplanarScaleV = 0.002f;
        materials[15u].invertHeight = true;

        materials[16u].mappingMode = true;
        materials[16u].tessLevel = 5.0f;
        materials[16u].tessMinLevel = 2.0f;
        materials[16u].tessMaxLevel = 16.0f;
        materials[16u].tessHeightScale = 32.0f;
        materials[16u].triplanar = true;
        materials[16u].triplanarScaleU = 0.002f;
        materials[16u].triplanarScaleV = 0.002f;
        materials[16u].invertHeight = true;

        materials[18u].mappingMode = true;
        materials[18u].tessLevel = 5.0f;
        materials[18u].tessMinLevel = 2.0f;
        materials[18u].tessMaxLevel = 16.0f;
        materials[18u].tessHeightScale = 32.0f;
        materials[18u].triplanar = true;
        materials[18u].triplanarScaleU = 0.002f;
        materials[18u].triplanarScaleV = 0.002f;
        materials[18u].invertHeight = true;

        materials[20u].mappingMode = true;
        materials[20u].tessLevel = 5.0f;
        materials[20u].tessMinLevel = 2.0f;
        materials[20u].tessMaxLevel = 16.0f;
        materials[20u].tessHeightScale = 32.0f;
        materials[20u].triplanar = true;
        materials[20u].triplanarScaleU = 0.002f;
        materials[20u].triplanarScaleV = 0.002f;
        materials[20u].invertHeight = true;

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
        for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i)
            shadowParams.shadowMapSizes[i] = sceneRenderer->shadowMapper->getShadowMapSize(i);
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
            wp.tessMinLevel = 2.0f;
            wp.tessMaxLevel = 16.0f;
            wp.reflectionStrength = 0.5f;
            wp.fresnelPower = 1.0f;
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
            wp.tessMinLevel = 1.0f;
            wp.tessMaxLevel = 1.0f;
            waterParams.push_back(wp); // Add a third layer to demonstrate pagination in UI even without texture arrays
        }


        // Create scene objects and build the octree synchronously in setup().
        // Chunk tessellation is deferred and processed on a background thread.
        // Vulkan GPU uploads happen on the main thread via processPendingMeshes().
        scenePicker_.addBookmark(
            reinterpret_cast<const char*>(u8"\uf07c##scene_bm_scenes"),
            "Go to project scenes folder",
            std::filesystem::path("scenes"));
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

        // Init the VegetationRenderer before setupVegetationTextures so that
        // wind params UBO + descriptor set layout exist before captureAll calls
        // setImpostorData().  SceneRenderer::init() is called later after all
        // texture/material setup is complete.
        if (sceneRenderer && sceneRenderer->vegetationRenderer)
            sceneRenderer->vegetationRenderer->init(this);

        setupVegetationTextures();
        setupTextures();

        sceneRenderer->init(this, &textureArrayManager, &materialManager, waterParams);

        // Re-wire impostors now that VegetationRenderer::init() has stored the render pass.
        if (impostorService) impostorService->rewire();

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
        gamepadWidget = std::make_shared<GamepadWidget>(controllerManager.getParameters(), &nunchukPublisher);
        // Auto-connect to a Wiimote (with or without Nunchuk) on startup.
        nunchukPublisher.connect();
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
        shadowParams.update(camera.getPosition(), light, camera.getViewProjectionMatrix(), settings.nearPlane, settings.farPlane);
        
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
                for (uint32_t f = 0; f < 3; ++f) {
                    if (vkCreateQueryPool(getDevice(), &qpci, nullptr, &queryPools[f]) != VK_SUCCESS)
                        throw std::runtime_error("Failed to create timestamp query pool");
                }
            }
        }
        preAllocateAsyncDescriptorPools();
        // Try loading the default scene; fall back to procedural generation if it fails
        const std::string defaultScenePath = "scenes/default.scene";
        if (std::filesystem::exists(defaultScenePath)) {
            pendingLoadPath = defaultScenePath;
            loadScenePending = true;
            std::cout << "[MyApp::setup] Loading default scene from '" << defaultScenePath << "'\n";
        } else {
            generateMapPending = true; // Trigger initial map generation on first frame
            std::cout << "[MyApp::setup] No default scene found, generating procedural map\n";
        }
    }

    // Move vegetation texture setup into its own method for clarity
    void setupVegetationTextures();
    // Move scene-loading into its own method for clarity
    void setupScene();
    // Pre-allocate descriptor pool+set rings for async tasks
    void preAllocateAsyncDescriptorPools();
    // Rebuild the brush preview scene from Brush3dWidget entries
    void rebuildBrushScene();
    // When brush animation is enabled, advance the trajectory time and move the
    // selected brush entry along a circular orbit; the actual brush rebuild is
    // performed by rebuildBrushScene() afterwards.
    void updateBrushAnimation(float deltaTime);
    // Clear GPU meshes, reset octrees and regenerate via MainSceneLoader
    void generateMap();
    void action();
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
        // Poll nunchuk state (if a Wiimote with nunchuk extension is connected)
        nunchukPublisher.update();
        eventManager.processQueued();

        shadowParams.update(camera.getPosition(), light, camera.getViewProjectionMatrix(), settings.nearPlane, settings.farPlane);

        // Drain the pending mesh queue populated by the background scene-loading
        // thread.  GPU uploads happen here on the main thread so newly generated
        // chunks become visible progressively without blocking the render loop.
        // Process pending meshes at a controlled rate (10 per frame).
        // Chunks closest to the camera are uploaded first.
        if (sceneRenderer && !isLoading)
            sceneRenderer->processPendingMeshes(this, camera.getPosition());

        // Brush meshes drain from their OWN queue (decoupled from solid/water).
        if (sceneRenderer && !isLoading)
            sceneRenderer->processPendingBrushMeshes(this, camera.getPosition());

        // Drive the async streaming subsystem each frame. prepareFrameWaits()
        // registers upload completion semaphores with this frame's submit, and
        // processUploads() submits as many queued jobs as staging allows — no
        // fixed per-frame cap. (Terrain/water/brush copies currently still go
        // through IndirectRenderer; this is the integration point for migrating
        // them onto UploadManager.)
        if (sceneRenderer)
            sceneRenderer->streamer.update(this);

        // Drain the CPU vegetation-generation queue so chunkBuffers is
        // populated before preRenderPass records read barriers.  Must
        // happen here because barriers cannot be emitted inside dynamic
        // rendering, and draw() runs inside beginPass/endPass.
        if (sceneRenderer && sceneRenderer->vegetationRenderer) {
            sceneRenderer->vegetationRenderer->processPendingChunks(10);
        }

        mainTime += deltaTime;
        lastFrameDelta = deltaTime;
        if (sceneRenderer && sceneRenderer->vegetationRenderer) {
            sceneRenderer->vegetationRenderer->setWindTime(mainTime);
            sceneRenderer->vegetationRenderer->setImpostorDistance(settings.impostorDistance);
        }
        if (sceneRenderer && sceneRenderer->skyRenderer) {
            sceneRenderer->skyRenderer->update(this);
        }
        profileCpuUpdate = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - cpuUpdateT0).count();
    }

    void preRenderPass(VkCommandBuffer &commandBuffer) override {

        uint32_t frameIdx = getCurrentFrame();

        // Profiling: read previous frame's query results (with availability flag to
        // avoid even partial driver stalls), then reset for this frame.
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE) {
            if (queryPoolReady[frameIdx]) {
                auto msDiff = [&](uint64_t endTs, uint64_t startTs) -> float {
                    return static_cast<float>(endTs - startTs) * timestampPeriod * 1e-6f;
                };
                // Group A: indices 0-9 (always written by main command buffer)
                struct { uint64_t value; uint64_t availability; } tsA[10] = {};
                if (vkGetQueryPoolResults(getDevice(), queryPools[frameIdx], 0, 10,
                        sizeof(tsA), tsA, sizeof(tsA[0]),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS
                        && timestampPeriod > 0.0f) {
                    if (tsA[0].availability) profileShadow        = msDiff(tsA[1].value, tsA[0].value);
                    if (tsA[2].availability) profileMainCull      = msDiff(tsA[3].value, tsA[2].value);
                    if (tsA[4].availability) profileSky           = msDiff(tsA[5].value, tsA[4].value);
                    if (tsA[6].availability) profileSolidDraw     = msDiff(tsA[7].value, tsA[6].value);
                    if (tsA[8].availability) profileVegetationReal = msDiff(tsA[9].value, tsA[8].value);
                }
                // Group B: indices 10-13 (written by main command buffer for color pass)
                struct { uint64_t value; uint64_t availability; } tsB[4] = {};
                if (vkGetQueryPoolResults(getDevice(), queryPools[frameIdx], 10, 4,
                        sizeof(tsB), tsB, sizeof(tsB[0]),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS
                        && timestampPeriod > 0.0f) {
                    if (tsB[0].availability) profileVegetationImpostor = msDiff(tsB[1].value, tsB[0].value);
                }
                // Group C: indices 14-17 (always written by main command buffer)
                struct { uint64_t value; uint64_t availability; } tsC[4] = {};
                if (vkGetQueryPoolResults(getDevice(), queryPools[frameIdx], 14, 4,
                        sizeof(tsC), tsC, sizeof(tsC[0]),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS
                        && timestampPeriod > 0.0f) {
                    if (tsC[0].availability) profileWater = msDiff(tsC[1].value, tsC[0].value);
                    if (tsC[2].availability) profileImGui = msDiff(tsC[3].value, tsC[2].value);
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
        uboStatic.invViewProjection = glm::inverse(viewProj);
        uboStatic.viewPos = glm::vec4(camera.getPosition(), 1.0f);
        uboStatic.lightDir = glm::vec4(light.getDirection(), 0.0f);
        uboStatic.lightColor = glm::vec4(1.0f, 1.0f, 0.9f, 1.0f);
        uboStatic.lightSpaceMatrix  = shadowParams.lightSpaceMatrix[0];
        uboStatic.lightSpaceMatrix1 = shadowParams.lightSpaceMatrix[1];
        uboStatic.lightSpaceMatrix2 = shadowParams.lightSpaceMatrix[2];
        // Encode debug/triplanar/tess parameters into the shared UBO
        uboStatic.debugParams = glm::vec4(static_cast<float>(settings.debugMode), settings.roughnessEnabled ? 1.0f : 0.0f, settings.aoEnabled ? 1.0f : 0.0f, 0.0f);
        uboStatic.triplanarSettings = glm::vec4(settings.triplanarThreshold, settings.triplanarExponent, 0.0f, 0.0f);
        uboStatic.tessParams = glm::vec4(
            settings.tessMinDistance,
            settings.tessMaxDistance,
            settings.tessellationFactor,
            0.0f
        );
        // passParams: x = isShadowPass, y = tessEnabled, z = nearPlane, w = farPlane
        uboStatic.passParams = glm::vec4(0.0f, settings.tessellationEnabled ? 1.0f : 0.0f, settings.nearPlane, settings.farPlane);
        // materialFlags.w = global normal-mapping toggle (shader checks ubo.materialFlags.w > 0.5)
        uboStatic.materialFlags.w = settings.normalMappingEnabled ? 1.0f : 0.0f;
        // shadowEffects.w = global shadow toggle (shader checks ubo.shadowEffects.w > 0.5)
        uboStatic.shadowEffects.w = settings.enableShadows ? 1.0f : 0.0f;

        // Reset command buffer state tracker and wire it to all sub-renderers.
        // NOTE: backFaceRenderer and waterRenderer's IndirectRenderer are deliberately
        // excluded — they are accessed by the async back-face task on a separate thread
        // and keeping cmdState=nullptr for them avoids a data race on frameCmdState.
        // Null checks mirror existing guards in the render code below.
        sceneRenderer->frameCmdState.reset();
        if (sceneRenderer->shadowMapper) sceneRenderer->shadowMapper->setCmdState(&sceneRenderer->frameCmdState);
        if (sceneRenderer->solidRenderer) sceneRenderer->solidRenderer->setCmdState(&sceneRenderer->frameCmdState);
        if (sceneRenderer->skyRenderer) sceneRenderer->skyRenderer->setCmdState(&sceneRenderer->frameCmdState);
        if (sceneRenderer->vegetationRenderer) sceneRenderer->vegetationRenderer->setCmdState(&sceneRenderer->frameCmdState);
        if (sceneRenderer->postProcessRenderer) sceneRenderer->postProcessRenderer->setCmdState(&sceneRenderer->frameCmdState);
        if (sceneRenderer->debugCubeRenderer) sceneRenderer->debugCubeRenderer->setCmdState(&sceneRenderer->frameCmdState);
        if (sceneRenderer->debugSDFRenderer) sceneRenderer->debugSDFRenderer->setCmdState(&sceneRenderer->frameCmdState);
        if (sceneRenderer->solidWireframe) sceneRenderer->solidWireframe->setCmdState(&sceneRenderer->frameCmdState);
        if (sceneRenderer->waterWireframe) sceneRenderer->waterWireframe->setCmdState(&sceneRenderer->frameCmdState);
        if (sceneRenderer->solid360Renderer) sceneRenderer->solid360Renderer->setCmdState(&sceneRenderer->frameCmdState);
        if (sceneRenderer->waterRenderer) sceneRenderer->waterRenderer->setCmdState(&sceneRenderer->frameCmdState);
        if (sceneRenderer->solidRenderer) sceneRenderer->solidRenderer->getIndirectRenderer().setCmdState(&sceneRenderer->frameCmdState);

        // ── GPU culling: must run BEFORE shadow pass so drawPrepared has
        // current-frame compact/visibleCount buffers populated. ──
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 2);
        sceneRenderer->solidRenderer->getIndirectRenderer().acquireBuffers(commandBuffer);
        sceneRenderer->solidRenderer->getIndirectRenderer().prepareCull(commandBuffer, viewProj);
        if (sceneRenderer->vegetationRenderer && settings.vegetationEnabled) {
            sceneRenderer->vegetationRenderer->prepareCull(commandBuffer, viewProj);
        }
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 3);

        // Shadow pass: renders solid geometry into shadow map from light's perspective
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 0);
        if (sceneRenderer) {
            sceneRenderer->shadowPass(this, commandBuffer, getMainDescriptorSet(), frameIdx, sceneRenderer->mainUniformBuffers[frameIdx], uboStatic, settings.enableShadows, settings.vegetationEnabled);
        }
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 1);

        // Upload UBO to GPU (VMA persistently mapped — no map/unmap needed)
        if (sceneRenderer) {
            memcpy(sceneRenderer->mainUniformBuffers[frameIdx].mappedData, &uboStatic, sizeof(UniformObject));
        } else {
            std::cerr << "[MyApp::preRenderPass] sceneRenderer is null, skipping UBO upload\n";
        }

        const bool waterEnabled = settings.waterEnabled;
        const bool vegetationEnabled = settings.vegetationEnabled;

        // ── Cubemap render on main CB (after shadow pass, reads fresh shadow maps) ──
        if (waterEnabled && sceneRenderer && sceneRenderer->solid360Renderer) {
            ensureCubemapResources();

            // Force dummy cubemap into cube360GfxDs binding #11 every frame
            if (cube360GfxDs != VK_NULL_HANDLE && sceneRenderer->solid360Renderer) {
                VkImageView dummyView = sceneRenderer->solid360Renderer->getDummyCubeView();
                VkSampler cubeSamp = sceneRenderer->solid360Renderer->getSolid360Sampler();
                if (dummyView != VK_NULL_HANDLE && cubeSamp != VK_NULL_HANDLE) {
                    DescriptorWriter(getDevice())
                        .writeImage(cube360GfxDs, 11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                    cubeSamp, dummyView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                        .flush();
                }
            }

            UniformObject ubo360 = uboStatic;
            ubo360.materialFlags.x = 1.0f; // skipEnvMap flag

            auto tCubemap = std::chrono::high_resolution_clock::now();
            this->sceneRenderer->solid360Renderer->renderSolid360(
                this, commandBuffer,
                this->sceneRenderer->skyRenderer.get(), this->sceneRenderer->getSkySettings().mode,
                this->sceneRenderer->solidRenderer.get(),
                cube360GfxDs,
                cube360UBO, ubo360,
                cube360ComputeDs,
                cube360Compact.buffer,
                cube360Visible.buffer,
                cube360WaterComputeDs,
                cube360WaterCompact.buffer,
                cube360WaterVisible.buffer);
            this->profileSolid360 = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - tCubemap).count();
        }

        // Render sky + solids/vegetation into the solid offscreen framebuffer (one per frame)

        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 4);
        if (sceneRenderer->skyRenderer) {
            SkySettings::Mode skyMode = sceneRenderer->getSkySettings().mode;
            sceneRenderer->skyRenderer->renderOffscreen(this, commandBuffer, frameIdx,
                getMainDescriptorSet(), sceneRenderer->mainUniformBuffers[frameIdx], uboStatic, viewProj, skyMode);
        }

        VkClearValue colorClear{};
        // Clear solid offscreen color to transparent so composite starts from empty scene
        colorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        VkClearValue depthClear{};
        depthClear.depthStencil = {1.0f, 0};

            // Acquire vegetation instance/indirect buffers before
            // vkCmdBeginRendering (barriers illegal inside dynamic rendering).
            if (vegetationEnabled && sceneRenderer->vegetationRenderer) {
                sceneRenderer->vegetationRenderer->recordReadBarriers(commandBuffer);
            }

            // Ensure the 360° cubemap writes (from the renderSolid360 call above
            // on the same CB) are visible to the solid shader's environment-map
            // sampler.
            if (sceneRenderer && sceneRenderer->solid360Renderer) {
                VkImage cubeImg = sceneRenderer->solid360Renderer->getCube360ColorImage();
                if (cubeImg != VK_NULL_HANDLE) {
                    RendererUtils::transitionImageLayout(
                        commandBuffer, cubeImg,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT, 0, 6);
                }
            }

        // Transition depth to DEPTH_STENCIL_ATTACHMENT_OPTIMAL for Instance 1 below.
        // (The previous frame left it in SHADER_READ_ONLY after the water pass.)
        {
            VkImage solidDepthImg = sceneRenderer->solidRenderer->getDepthImage(frameIdx);
            if (solidDepthImg != VK_NULL_HANDLE) {
                RendererUtils::transitionImageLayout(
                    commandBuffer, solidDepthImg,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);
                setImageLayoutTracked(solidDepthImg, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
            }
        }

        // ── Instance 1: Deferred depth pre-pass (no color attachment) ──
        // Solid + vegetation write depth; impostors use single-pass (depth+color in Instance 2).
        {
            VkRenderingAttachmentInfo depthAtt{};
            depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAtt.imageView = sceneRenderer->solidRenderer->getDepthView(frameIdx);
            depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAtt.clearValue = depthClear;

            VkRenderingInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.offset = {0, 0};
            ri.renderArea.extent = {static_cast<uint32_t>(getWidth()), static_cast<uint32_t>(getHeight())};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 0;
            ri.pColorAttachments = nullptr;
            ri.pDepthAttachment = &depthAtt;

            vkCmdBeginRendering(commandBuffer, &ri);

            {
                VkViewport vp{0.0f, 0.0f, (float)getWidth(), (float)getHeight(), 0.0f, 1.0f};
                vkCmdSetViewport(commandBuffer, 0, 1, &vp);
                VkRect2D sc{{0, 0}, {(uint32_t)getWidth(), (uint32_t)getHeight()}};
                vkCmdSetScissor(commandBuffer, 0, 1, &sc);
            }

            // Solid geometry depth
            sceneRenderer->solidRenderer->drawDepth(commandBuffer, this, getMainDescriptorSet());

            // Vegetation depth (impostors render depth+color in Instance 2)
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 8);
            if (vegetationEnabled && sceneRenderer->vegetationRenderer) {
                sceneRenderer->vegetationRenderer->drawDepth(this, commandBuffer, viewProj, camera.getPosition());
            }
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 9);

            vkCmdEndRendering(commandBuffer);
        }

        // Note: no barrier needed between instances — vkCmdEndRendering makes depth
        // writes available; Instance 2's LOAD_OP_LOAD waits on them.

        // Transition color to COLOR_ATTACHMENT_OPTIMAL for Instance 2 below.
        {
            VkImage solidColorImg = sceneRenderer->solidRenderer->getColorImage(frameIdx);
            if (solidColorImg != VK_NULL_HANDLE) {
                RendererUtils::transitionImageLayout(
                    commandBuffer, solidColorImg,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
                setImageLayoutTracked(solidColorImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, 1);
            }
        }

        // ── Instance 2: Color pass (load depth from prepass, LESS compare for impostors) ──
        {
            VkRenderingAttachmentInfo colorAtt{};
            colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            colorAtt.imageView = sceneRenderer->solidRenderer->getColorView(frameIdx);
            colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAtt.clearValue = colorClear;

            VkRenderingAttachmentInfo depthAtt{};
            depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAtt.imageView = sceneRenderer->solidRenderer->getDepthView(frameIdx);
            depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAtt.clearValue = depthClear;

            VkRenderingInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.offset = {0, 0};
            ri.renderArea.extent = {static_cast<uint32_t>(getWidth()), static_cast<uint32_t>(getHeight())};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &colorAtt;
            ri.pDepthAttachment = &depthAtt;

            vkCmdBeginRendering(commandBuffer, &ri);

            {
                VkViewport vp{0.0f, 0.0f, (float)getWidth(), (float)getHeight(), 0.0f, 1.0f};
                vkCmdSetViewport(commandBuffer, 0, 1, &vp);
                VkRect2D sc{{0, 0}, {(uint32_t)getWidth(), (uint32_t)getHeight()}};
                vkCmdSetScissor(commandBuffer, 0, 1, &sc);
            }

            // Sky first (background, no depth write)
            if (sceneRenderer->skyRenderer) {
                SkySettings::Mode skyMode = sceneRenderer->getSkySettings().mode;
                sceneRenderer->skyRenderer->render(this, commandBuffer, getMainDescriptorSet(),
                    sceneRenderer->mainUniformBuffers[frameIdx], uboStatic, viewProj, skyMode);
                // Restore the main UBO that sky rendering overwrote
                void* data;
                data = sceneRenderer->mainUniformBuffers[frameIdx].map(0);
                memcpy(data, &uboStatic, sizeof(UniformObject));
                sceneRenderer->mainUniformBuffers[frameIdx].unmap(); // VMA persistent mapping
            }
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 5);

            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 6);

            // Solid geometry color (LESS_OR_EQUAL, no depth write)
            sceneRenderer->solidRenderer->drawColor(commandBuffer, this, getMainDescriptorSet());

            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 7);

            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 10);

            // Vegetation color + impostor color+depth (single-pass depth write)
            if (vegetationEnabled && sceneRenderer->vegetationRenderer) {
                sceneRenderer->vegetationRenderer->drawColor(this, commandBuffer, viewProj, camera.getPosition());
            }

            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 11);

            // Debug renders on top
            const bool showOctreeDebug = octreeExplorerWidget && octreeExplorerWidget->getShowDebugCubes();
            if (showOctreeDebug) {
                std::vector<DebugCubeRenderer::CubeWithColor> debugCubes;
                if (showOctreeDebug && octreeExplorerWidget->isVisible()) {
                    const auto& widgetCubes = octreeExplorerWidget->getExpandedCubes();
                    debugCubes.reserve(widgetCubes.size());
                    for (const auto& wc : widgetCubes)
                        debugCubes.push_back({BoundingBox(wc.cube.getMin(), wc.cube.getMax()), wc.color});
                }
                if (showOctreeDebug) {
                    auto nodeCubes = sceneRenderer->getDebugNodeCubes();
                    debugCubes.reserve(debugCubes.size() + nodeCubes.size());
                    for (auto &nc : nodeCubes) debugCubes.push_back(nc);
                }
                if (!debugCubes.empty()) {
                    sceneRenderer->debugCubeRenderer->setCubes(debugCubes);
                    sceneRenderer->debugCubeRenderer->render(this, commandBuffer, getMainDescriptorSet());
                }
            }

            if (settings.showBoundingBoxes && sceneRenderer && sceneRenderer->boundingBoxRenderer) {
                std::vector<DebugCubeRenderer::CubeWithColor> boxes;
                auto gatherBoxesFrom = [&](const IndirectRenderer &ir, const glm::vec3 &color){
                    ir.visitActiveMeshInfos([&](const IndirectRenderer::MeshInfo &mi){
                        boxes.push_back({BoundingBox(glm::vec3(mi.boundsMin), glm::vec3(mi.boundsMax)), color});
                    });
                };
                gatherBoxesFrom(sceneRenderer->solidRenderer->getIndirectRenderer(), glm::vec3(0.0f, 1.0f, 0.0f));
                gatherBoxesFrom(sceneRenderer->waterRenderer->getIndirectRenderer(), glm::vec3(0.0f, 0.5f, 1.0f));
                if (!boxes.empty()) {
                    sceneRenderer->boundingBoxRenderer->setCubes(boxes);
                    sceneRenderer->boundingBoxRenderer->render(this, commandBuffer, getMainDescriptorSet());
                }
            }

            if (settings.showSDFDebug && sceneRenderer && sceneRenderer->debugSDFRenderer) {
                auto sdfCubes = sceneRenderer->getDebugSDFCubes();
                if (!sdfCubes.empty()) {
                    sceneRenderer->debugSDFRenderer->setCubes(sdfCubes);
                    sceneRenderer->debugSDFRenderer->render(this, commandBuffer, getMainDescriptorSet());
                }
            }

            if (settings.wireframeMode && sceneRenderer) {
                sceneRenderer->drawSolidWireframeOverlay(this, commandBuffer, frameIdx, getMainDescriptorSet(), settings.wireframeMode);
            }

            vkCmdEndRendering(commandBuffer);
        }

        // Transition color+depth to SHADER_READ_ONLY for water pass compositing.
        // The water pass's initializeGeomDepthFromSceneDepth expects the scene depth
        // in SHADER_READ_ONLY_OPTIMAL (it transitions to TRANSFER_SRC internally).
        {
            VkImage solidColorImg = sceneRenderer->solidRenderer->getColorImage(frameIdx);
            VkImage solidDepthImg = sceneRenderer->solidRenderer->getDepthImage(frameIdx);
            uint32_t bc = 0;
            VkImageMemoryBarrier2 barriers[2]{};

            if (solidColorImg != VK_NULL_HANDLE) {
                barriers[bc].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barriers[bc].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                barriers[bc].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                barriers[bc].dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                barriers[bc].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                barriers[bc].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                barriers[bc].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barriers[bc].image = solidColorImg;
                barriers[bc].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                ++bc;
            }
            if (solidDepthImg != VK_NULL_HANDLE) {
                barriers[bc].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barriers[bc].srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                barriers[bc].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                barriers[bc].dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                barriers[bc].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                barriers[bc].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                barriers[bc].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barriers[bc].image = solidDepthImg;
                barriers[bc].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
                ++bc;
            }
            if (bc > 0) {
                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = bc;
                dep.pImageMemoryBarriers = barriers;
                vkCmdPipelineBarrier2(commandBuffer, &dep);
            }
            if (solidColorImg != VK_NULL_HANDLE)
                setImageLayoutTracked(solidColorImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
            if (solidDepthImg != VK_NULL_HANDLE)
                setImageLayoutTracked(solidDepthImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
        }

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

        // If water is disabled, clear its offscreen targets here (outside any active
        // dynamic rendering instance) so the post-process compositor won't sample
        // stale content.
        if (!waterEnabled && sceneRenderer && sceneRenderer->waterRenderer) {
            sceneRenderer->waterRenderer->clearRenderTargets(this, commandBuffer, frameIdx);
        }

        // Launch asynchronous recording+submit for independent offscreen passes
        // using a persistent thread pool to avoid per-frame std::thread creation overhead.
        bool launchedBackFace = false;
        VkSemaphore semBackFace = VK_NULL_HANDLE;
        std::future<void> asyncBackFaceFuture;

        // Back-face depth for water
        if (waterEnabled && sceneRenderer && sceneRenderer->backFaceRenderer) {
            launchedBackFace = true;
            asyncBackFaceFuture = asyncThreadPool.enqueue([this, viewProj, frameIdx, &semBackFace]() {
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
                auto lazyComputeSlot = [&](PoolSetPair* ring, uint32_t& idx, VkDescriptorSetLayout layout, const char* label) -> PoolSetPair& {
                    auto& s = ring[idx++ % ASYNC_RING_SIZE];
                    if (s.pool != VK_NULL_HANDLE) return s;
                    if (layout == VK_NULL_HANDLE) return s;
                    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
                    VkDescriptorPoolCreateInfo pci{};
                    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                    pci.poolSizeCount = 1; pci.pPoolSizes = &ps; pci.maxSets = 1;
                    pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
                    vkCreateDescriptorPool(device, &pci, nullptr, &s.pool);
                    app->resources.addDescriptorPool(s.pool, label);
                    VkDescriptorSetAllocateInfo ai{};
                    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    ai.descriptorPool = s.pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &layout;
                    vkAllocateDescriptorSets(device, &ai, &s.set);
                    app->resources.addDescriptorSet(s.set, label);
                    return s;
                };
                VkDescriptorPool taskPool = VK_NULL_HANDLE;
                VkDescriptorSet computeDs = VK_NULL_HANDLE;
                {
                    VkDescriptorSetLayout bfLayout = ind.getComputeDescriptorSetLayout();
                    auto& slot = lazyComputeSlot(cachedBackfaceCompute, ringBackfaceCompute, bfLayout, "Lazy cachedBackfaceCompute");
                    taskPool = slot.pool;
                    computeDs = slot.set;
                }

                // Update descriptor set with buffers: inCmds, outCmds, bounds, visibleCount
                if (computeDs != VK_NULL_HANDLE) {
                    DescriptorWriter(device)
                        .writeBuffer(computeDs, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     ind.getIndirectBuffer().buffer, 0, VK_WHOLE_SIZE)
                        .writeBuffer(computeDs, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     taskCompact.buffer, 0, VK_WHOLE_SIZE)
                        .writeBuffer(computeDs, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     ind.getBoundsBuffer().buffer, 0, VK_WHOLE_SIZE)
                        .writeBuffer(computeDs, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     taskVisible.buffer, 0, VK_WHOLE_SIZE)
                        .flush();
                }

                // Run cull into per-task buffers - only when compute pipeline is ready (meshes loaded)
                if (computeDs != VK_NULL_HANDLE) {
                    ind.prepareCullWithDescriptor(cmd, viewProj, computeDs, taskCompact.buffer, taskVisible.buffer);
                }

                // Render back-face pass using the per-task compact/visible buffers so draws consume the cull results
                auto tBackface = std::chrono::high_resolution_clock::now();
                this->sceneRenderer->backFaceRenderer->renderBackFacePass(app, cmd, frameIdx,
                                            ind,
                                            this->sceneRenderer->waterRenderer->getWaterGeometryPipelineLayout(),
                                            app->getMainDescriptorSet(),
                                            app->getMaterialDescriptorSet(),
                                            this->sceneRenderer->waterRenderer->getWaterDepthDescriptorSet(frameIdx),
                                            (computeDs != VK_NULL_HANDLE) ? taskCompact.buffer : VK_NULL_HANDLE,
                                            (computeDs != VK_NULL_HANDLE) ? taskVisible.buffer : VK_NULL_HANDLE);

                this->profileBackface = std::chrono::duration<float, std::milli>(
                    std::chrono::high_resolution_clock::now() - tBackface).count();
                // Submit and schedule cleanup of temporary resources after fence signals
                VkFence f = app->submitCommandBufferAsync(cmd, &semBackFace);
                { std::lock_guard<std::mutex> lk(asyncFenceMutex); pendingAsyncFences.push_back(f); }
                app->deferDestroyUntilFence(f, [app, taskCompact, taskVisible]() mutable {
                    app->destroyBuffer(taskCompact);
                    app->destroyBuffer(taskVisible);
                });
            });
        }

        // Wait for the async back-face task to complete before water pass so no two
        // threads call vkCmdBindDescriptorSets with the same descriptor set concurrently
        // and to ensure semBackFace is signaled before waterPass uses it.
        if (asyncBackFaceFuture.valid()) asyncBackFaceFuture.wait();

        // Run water geometry pass offscreen and bind scene textures for post-process
        if (waterEnabled) {
            // GPU frustum cull water meshes (must run outside a render pass)
            sceneRenderer->waterRenderer->getIndirectRenderer().acquireBuffers(commandBuffer);
            sceneRenderer->waterRenderer->getIndirectRenderer().prepareCull(commandBuffer, viewProj);
            // Use 360° solid+sky reflection instead of the sky-only equirect view
            VkImageView cubeReflectionView = VK_NULL_HANDLE;
            if (sceneRenderer && sceneRenderer->solid360Renderer) cubeReflectionView = sceneRenderer->solid360Renderer->getSolid360View();
            VkImageView skyView = (sceneRenderer && sceneRenderer->skyRenderer) ? sceneRenderer->skyRenderer->getSkyView(frameIdx) : VK_NULL_HANDLE;
            // If we launched an async back-face submission, tell waterPass to skip issuing it again.
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 14);
            sceneRenderer->waterPass(this, commandBuffer, frameIdx, getMainDescriptorSet(), settings.waterWireframeMode,
                mainTime, launchedBackFace, skyView, cubeReflectionView);
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 15);
        }

        profileCpuRecord = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - cpuRecordT0).count();
    }

    void renderImGui() override {
        static char sceneFolderBuf[512] = "scenes/default.scene";

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Generate Map")) generateMapPending = true;
                if (ImGui::MenuItem("Action")) action();
                ImGui::Separator();
                if (ImGui::MenuItem("Save Scene...")) scenePicker_.open(sceneFolderBuf, true,  "Mode: Save");
                if (ImGui::MenuItem("Load Scene...")) scenePicker_.open(sceneFolderBuf, false, "Mode: Load");
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

            std::filesystem::path chosenScene;
            if (scenePicker_.render(chosenScene)) {
                std::string s = chosenScene.string();
                if (s.size() < sizeof(sceneFolderBuf)) {
                    std::memcpy(sceneFolderBuf, s.c_str(), s.size());
                    sceneFolderBuf[s.size()] = '\0';
                }
                if (scenePicker_.isSaveMode()) {
                    if (mainScene) mainScene->save(sceneFolderBuf, &settings);
                } else {
                    pendingLoadPath = sceneFolderBuf;
                    loadScenePending = true;
                }
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
                    float vegReal = profileVegetationReal - profileVegetationImpostor;
                    if (vegReal < 0.0f) vegReal = 0.0f;
                    float gpuTotal = profileShadow + profileMainCull + profileSky +
                                     profileSolidDraw + profileVegetationReal +
                                     profileWater + profileImGui + profileSolid360 + profileBackface;
                    ImGui::Text("Shadow:        %.2f", profileShadow);
                    ImGui::Text("Main Cull:     %.2f", profileMainCull);
                    ImGui::Text("Sky:           %.2f", profileSky);
                    ImGui::Text("Solid Draw:    %.2f", profileSolidDraw);
                    ImGui::Text("Veg Real:      %.2f", vegReal);
                    ImGui::Text("Veg Impostor:  %.2f", profileVegetationImpostor);
                    ImGui::Text("Water:         %.2f", profileWater);
                    ImGui::Text("Solid360*:     %.2f", profileSolid360);
                    ImGui::Text("Backface*:     %.2f", profileBackface);
                    ImGui::Text("ImGui:         %.2f", profileImGui);
                    ImGui::Text("GPU Total:     %.2f", gpuTotal);
                    ImGui::Text("* = CPU-timed");
                    ImGui::Separator();
                    ImGui::Text("--- CPU Timing (ms) ---");
                    ImGui::Text("FPS:           %.1f", profileFps);
                    ImGui::Text("Update:        %.2f", profileCpuUpdate);
                    ImGui::Text("Record:        %.2f", profileCpuRecord);
                }

                // GPU memory usage (VK_EXT_memory_budget)
                {
                    auto budgets = getMemoryBudgets();
                    if (!budgets.empty()) {
                        ImGui::Separator();
                        ImGui::Text("--- GPU Memory (MB) ---");
                        for (size_t h = 0; h < budgets.size(); ++h) {
                            const auto& b = budgets[h];
                            const char* heapName = (b.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "Device Local" : "Host Visible";
                            float usageMB  = static_cast<float>(b.usage)  / (1024.0f * 1024.0f);
                            float budgetMB = static_cast<float>(b.budget) / (1024.0f * 1024.0f);
                            float totalMB  = static_cast<float>(b.size)   / (1024.0f * 1024.0f);
                            if (b.budget > 0) {
                                ImGui::Text("%s:  %.0f / %.0f MB", heapName, usageMB, budgetMB);
                            } else {
                                ImGui::Text("%s:  %.0f / %.0f MB (total)", heapName, usageMB, totalMB);
                            }
                        }
                    }
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
            sceneRenderer->solidRenderer->getIndirectRenderer().rebuild(this);
        }
        if (sceneRenderer->waterRenderer->getIndirectRenderer().isDirty()) {
            sceneRenderer->waterRenderer->getIndirectRenderer().rebuild(this);
        }

        uint32_t frameIdx = getCurrentFrame();

        // Per-frame cull buffers: each IndirectRenderer needs its own per-frame
        // compact/visibleCount buffer to avoid cross-frame overwrite races.
        sceneRenderer->solidRenderer->getIndirectRenderer().setCullFrame(frameIdx);
        sceneRenderer->waterRenderer->getIndirectRenderer().setCullFrame(frameIdx);

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

        // ImGui rendering
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (!draw_data) {
            std::cerr << "[MyApp::draw] Warning: ImGui::GetDrawData() returned nullptr, skipping ImGui rendering." << std::endl;
        } else if (commandBuffer == VK_NULL_HANDLE) {
            std::cerr << "[MyApp::draw] Error: commandBuffer is VK_NULL_HANDLE before ImGui rendering, skipping ImGui." << std::endl;
        } else {
            VkQueryPool qp = queryPools[getCurrentFrame()];
            if (profilingEnabled && qp != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 16);
            ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);
            if (profilingEnabled && qp != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 17);
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

        if (impostorService) {
            impostorService->cleanup();
        }

        // Cleanup scene renderer and all sub-renderers
        if (sceneRenderer) {
            sceneRenderer->cleanup(this);
        }

        // Destroy timestamp query pools
        for (uint32_t f = 0; f < 3; ++f) {
            if (queryPools[f] != VK_NULL_HANDLE) {
                vkDestroyQueryPool(getDevice(), queryPools[f], nullptr);
                queryPools[f] = VK_NULL_HANDLE;
            }
        }
        // Pre-allocated ring pools are tracked by VulkanResourceManager, so
        // resources.cleanup() will destroy them. Just zero our arrays.
        for (auto& slot : cachedBackfaceCompute) slot = {};

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

    void preImGuiShutdown() override {
        // Free all ImGui descriptor sets BEFORE Shutdown while the old backend data
        // is still alive. This avoids freeing DS allocated with the old descriptor
        // set layout through the new backend data after a Shutdown/Init cycle.
        if (sceneRenderer && sceneRenderer->shadowMapper) {
            sceneRenderer->shadowMapper->freeImGuiDescriptors();
        }
        if (renderTargetsWidget) {
            renderTargetsWidget->invalidateImGuiDescriptors();
        }
        if (billboardCreator) {
            billboardCreator->invalidateImGuiDescriptors();
        }
    }

    void onImGuiRecreated() override {
        // Re-create ImGui AddTexture DS for shadow cascades — the old ones used the
        // previous DescriptorSetLayout which was destroyed by ImGui_ImplVulkan_Shutdown.
        // Old handles were freed by preImGuiShutdown() so recreateImGuiDescriptors
        // will skip the free and go straight to allocation.
        if (sceneRenderer && sceneRenderer->shadowMapper) {
            sceneRenderer->shadowMapper->recreateImGuiDescriptors();
        }
        // Widget handles were nulled by preImGuiShutdown(); this is a no-op.
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
    // Ensure persistent cubemap rendering resources are allocated
    void ensureCubemapResources();

    // Called by VulkanApp after a frame has been submitted
    void postSubmit() override;
};



int main(int argc, char** argv) {
    // Allow threads (RADV driver, validation layers, miniaudio) to use
    // real-time scheduling. Without this, glibc's thread priority protection
    // code (tpp.c) hits an assertion when a library tries to promote a thread
    // to SCHED_FIFO but RLIMIT_RTPRIO is too low.
    struct rlimit rlim;
    if (getrlimit(RLIMIT_RTPRIO, &rlim) == 0) {
        rlim.rlim_cur = rlim.rlim_max;
        setrlimit(RLIMIT_RTPRIO, &rlim);
    }
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
    auto billboardSvc = std::make_shared<BillboardService>();
    billboardCreator = std::make_shared<BillboardCreator>(&billboardManager, &vegetationAtlasManager, &vegetationTextureArrayManager, billboardSvc);
    // Provide VulkanApp to the creator so it can initialize GPU-backed preview textures
    billboardCreator->setVulkanApp(this);
    impostorService = std::make_shared<ImpostorService>();
    impostorService->init(this);
    impostorWidget = std::make_shared<ImpostorWidget>(impostorService);
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

    // Wire freshly baked billboard array textures to VegetationRenderer.
    // The setup() call to setBillboardArrayTextures happens before baking,
    // so the views were VK_NULL_HANDLE. Re-wire now with valid views.
    if (sceneRenderer && sceneRenderer->vegetationRenderer && billboardCreator) {
        sceneRenderer->vegetationRenderer->setBillboardArrayTextures(
            billboardCreator->getAlbedoArrayView(),
            billboardCreator->getNormalArrayView(),
            billboardCreator->getOpacityArrayView(),
            billboardCreator->getArraySampler(),
            this
        );
    }

    // Notify ImpostorService about the freshly baked texture arrays.
    if (impostorService && billboardCreator) {
        impostorService->setSource(
            billboardCreator->getAlbedoArrayView(),
            billboardCreator->getNormalArrayView(),
            billboardCreator->getOpacityArrayView(),
            billboardCreator->getArraySampler(),
            static_cast<int>(billboardManager.getBillboardCount()));
    }
}

// Implementation: pre-allocate descriptor pool+set rings for async tasks
void MyApp::preAllocateAsyncDescriptorPools() {
    VkDevice device = getDevice();

    auto allocateComputeRing = [&](PoolSetPair* ring, VkDescriptorSetLayout dsLayout, const char* label) {
        if (dsLayout == VK_NULL_HANDLE) return;
        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        for (uint32_t i = 0; i < ASYNC_RING_SIZE; ++i) {
            VkDescriptorPool pool;
            if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
                std::cerr << "[Async] Failed to pre-allocate " << label << " pool " << i << "\n";
                ring[i] = {};
                continue;
            }
            { std::string s = std::string(label) + " ring #" + std::to_string(i); resources.addDescriptorPool(pool, s.c_str()); }
            VkDescriptorSet set;
            VkDescriptorSetAllocateInfo ainfo{};
            ainfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ainfo.descriptorPool = pool;
            ainfo.descriptorSetCount = 1;
            ainfo.pSetLayouts = &dsLayout;
            if (vkAllocateDescriptorSets(device, &ainfo, &set) != VK_SUCCESS) {
                resources.removeDescriptorPool(pool);
                vkDestroyDescriptorPool(device, pool, nullptr);
                std::cerr << "[Async] Failed to pre-allocate " << label << " set " << i << "\n";
                ring[i] = {};
                continue;
            }
            { std::string s = std::string(label) + " DS #" + std::to_string(i); resources.addDescriptorSet(set, s.c_str()); }
            ring[i] = {pool, set};
        }
    };

    // Backface compute (same layout as water compute)
    if (sceneRenderer && sceneRenderer->waterRenderer) {
        auto& waterInd = sceneRenderer->waterRenderer->getIndirectRenderer();
        allocateComputeRing(cachedBackfaceCompute, waterInd.getComputeDescriptorSetLayout(), "cachedBackfaceCompute");
    }
}

// Implementation: rebuild the brush scene from Brush3dWidget entries
void MyApp::rebuildBrushScene() {
    if (!brushScene || !sceneRenderer || !brush3dWidget) return;

    // No device-wide stall here. The brush flow no longer writes into the shared
    // vertex/index buffers in place: it batches mesh adds and rebuilds once per
    // layer, and IndirectRenderer::rebuild() double-buffers the merged buffers
    // across a fixed pool — the batch lands in a fresh slot no in-flight frame is
    // reading, and the previous slot is recycled only once its frames retire
    // (frame-fence-gated, non-blocking). In-flight frames keep reading the old
    // buffers safely, so there is no WRITE_AFTER_READ hazard and no need to idle
    // the whole device every drag frame. See SceneRenderer::processPendingBrushMeshes
    // / updateBrushMeshForNode and IndirectRenderer::rebuild().

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

    // angle=0.95 (cos≈18°): normals within 18° → flat surface → full distance tolerance.
    // distance=0.2: flat patches may have up to 20% cube-size SDF error (curved gets 10%).
    Simplifier simplifier(0.95f, 0.2f, true);
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
                            octree.apply(SDF::opUnion, &effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        else
                            octree.apply(SDF::opSubtraction, &effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        break;
                    }
                    case 1: {
                        WrappedPerlinCarveDistanceEffect effect(wrappedFunc,
                            entry.effectAmplitude, entry.effectFrequency, entry.effectThreshold,
                            glm::vec3(0), entry.effectBrightness, entry.effectContrast);
                        if (entry.brushMode == 0)
                            octree.apply(SDF::opUnion, &effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        else
                            octree.apply(SDF::opSubtraction, &effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        break;
                    }
                    case 2: {
                        WrappedSineDistortDistanceEffect effect(wrappedFunc,
                            entry.effectAmplitude, entry.effectFrequency, glm::vec3(0));
                        if (entry.brushMode == 0)
                            octree.apply(SDF::opUnion, &effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        else
                            octree.apply(SDF::opSubtraction, &effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        break;
                    }
                    case 3: {
                        WrappedVoronoiCarveDistanceEffect effect(wrappedFunc,
                            entry.effectAmplitude, entry.effectCellSize,
                            glm::vec3(0), entry.effectBrightness, entry.effectContrast);
                        if (entry.brushMode == 0)
                            octree.apply(SDF::opUnion, &effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        else
                            octree.apply(SDF::opSubtraction, &effect, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        break;
                    }
                    default: {
                        // Fallback: no effect
                        if (entry.brushMode == 0)
                            octree.apply(SDF::opUnion, wrappedFunc, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        else
                            octree.apply(SDF::opSubtraction, wrappedFunc, model, translate, scale, brush, entry.minSize, simplifier, handler);
                        break;
                    }
                }
            } else {
                // No effect — use the primitive directly
                if (entry.brushMode == 0)
                    octree.apply(SDF::opUnion, wrappedFunc, model, translate, scale, brush, entry.minSize, simplifier, handler);
                else
                    octree.apply(SDF::opSubtraction, wrappedFunc, model, translate, scale, brush, entry.minSize, simplifier, handler);
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
            case 8: { // Tapered Cylinder
                TaperedCylinderDistanceFunction fn(entry.taperedCylinderRadii.x, entry.taperedCylinderRadii.y);
                WrappedTaperedCylinder wrapped(&fn);
                applyEntry(&wrapped);
                break;
            }
            case 9: { // Tapered Capsule
                TaperedCapsuleDistanceFunction fn(entry.capsuleA, entry.capsuleB,
                    entry.taperedCapsuleRadii.x, entry.taperedCapsuleRadii.y);
                WrappedTaperedCapsule wrapped(&fn);
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

// Advance the brush-animation clock and move the *selected* brush entry along a
// circular trajectory layered on top of the existing triangle-strip ring
// (MainSceneLoader). Only the entry's translate is changed — its shape, size,
// texture, mode and layer are left intact, so the animation composes with the
// user's brush definition. The entry retains the last animated position when
// animation is disabled, allowing manual editing from there.
void MyApp::updateBrushAnimation(float deltaTime) {
    BrushEntry* entry = brushManager.getSelectedEntry();
    if (!entry) return;

    // Freeze the clock while disabled so re-enabling continues from the same
    // orbit phase (the entry keeps its last animated translate).
    brushAnimTime += deltaTime;

    // Ring parameters (defined in MainSceneLoader): centered at origin, height
    // 800, outer radius = worldScale (1500). The brush orbits that ring.
    constexpr float ringHeight = 800.0f;
    constexpr float ringRadius = 1500.0f;   // worldScale * unitOuter(1.0)
    constexpr float orbitSpeed = 0.5f;      // radians / second

    float angle = brushAnimTime * orbitSpeed;
    entry->translate = glm::vec3(ringRadius * std::cos(angle), ringHeight,
                                 ringRadius * std::sin(angle));
}

// Ensure pending texture generation requests are flushed after a frame is submitted
// so array-layer transitions happen outside of active draw command buffers.

void MyApp::action() {
    // Join any previous background tessellation thread
    if (sceneProcessThread.joinable()) sceneProcessThread.join();

    // Wait for the GPU to finish all in-flight work before clearing GPU resources
    deviceWaitIdle();

    MainSceneLoader loader;
    mainScene->action(loader, *sceneUniqueSolidHandler, *sceneUniqueLiquidHandler);
    std::cout << "[MyApp::action] Octree construction complete\n";

    // Tessellate chunks in a background thread. Solid and water are handled on
    // separate threads so both layers tessellate truly in parallel (water no
    // longer waits for solid to finish).
    sceneProcessThread = std::thread([this]() {
        std::thread solidThread([this]() { sceneUniqueSolidHandler->handleEvents(); });
        std::thread waterThread([this]() { sceneUniqueLiquidHandler->handleEvents(); });
        solidThread.join();
        waterThread.join();
        if (octreeExplorerWidget)
            octreeExplorerWidget->octreeReady.store(true, std::memory_order_release);
        std::cout << "[MyApp::action] Scene chunk tessellation complete\n";
    });
}

void MyApp::generateMap() {
    // Join any previous background tessellation thread
    if (sceneProcessThread.joinable()) sceneProcessThread.join();

    // Wait for the GPU to finish all in-flight work before clearing GPU resources
    deviceWaitIdle();

    // Drain any pending deferred-destroy callbacks while the GPU is idle.
    processPendingCommandBuffers();

    // Remove all existing GPU-side meshes
    if (sceneRenderer) {
        sceneRenderer->removeAllRegisteredMeshes();
        sceneRenderer->removeAllTransparentMeshes();
        sceneRenderer->nodeDebugCubes.clear();
        sceneRenderer->clearDebugSDFCubes();
        // Clear vegetation chunk buffers and pending CPU-generation queue.
        // Stale instance buffers from the previous scene reference freed octree
        // memory and cause GPUVM faults (TCP read permission) on RADV when
        // the next frame's shadow pass binds the vegetation pipeline.
        if (sceneRenderer->vegetationRenderer) {
            sceneRenderer->vegetationRenderer->clearAllInstances();
        }
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

    // Tessellate chunks in a background thread. Solid and water are handled on
    // separate threads so both layers tessellate truly in parallel (water no
    // longer waits for solid to finish).
    sceneProcessThread = std::thread([this]() {
        std::thread solidThread([this]() { sceneUniqueSolidHandler->handleEvents(); });
        std::thread waterThread([this]() { sceneUniqueLiquidHandler->handleEvents(); });
        solidThread.join();
        waterThread.join();
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
        sceneRenderer->clearDebugSDFCubes();
        if (sceneRenderer->vegetationRenderer) {
            sceneRenderer->vegetationRenderer->clearAllInstances();
        }
    }

    mainScene->getOpaqueOctree().reset();
    mainScene->transparentOctree.reset();

    sceneUniqueSolidHandler->clear();
    sceneUniqueLiquidHandler->clear();

    if (octreeExplorerWidget)
        octreeExplorerWidget->octreeReady.store(false, std::memory_order_release);

    mainScene->load(path, *sceneUniqueSolidHandler, *sceneUniqueLiquidHandler, &settings);
    std::cout << "[MyApp::loadSceneFromFile] Octree loaded from '" << path << "'\n";

    // Solid and water tessellate on separate threads so both layers progress
    // truly in parallel (water no longer waits for solid to finish).
    sceneProcessThread = std::thread([this]() {
        std::thread solidThread([this]() { sceneUniqueSolidHandler->handleEvents(); });
        std::thread waterThread([this]() { sceneUniqueLiquidHandler->handleEvents(); });
        solidThread.join();
        waterThread.join();
        if (octreeExplorerWidget)
            octreeExplorerWidget->octreeReady.store(true, std::memory_order_release);
        std::cout << "[MyApp::loadSceneFromFile] Scene tessellation complete\n";
    });
}
void MyApp::ensureCubemapResources() {
    VkDevice device = getDevice();

    // Always write dummy cubemap to cube360GfxDs binding #11,
    // even if the DS was already allocated (e.g. before this fix was compiled).
    if (cube360GfxDs != VK_NULL_HANDLE && sceneRenderer && sceneRenderer->solid360Renderer) {
        VkImageView dummyView = sceneRenderer->solid360Renderer->getDummyCubeView();
        VkSampler cubeSamp = sceneRenderer->solid360Renderer->getSolid360Sampler();
        if (dummyView != VK_NULL_HANDLE && cubeSamp != VK_NULL_HANDLE) {
            DescriptorWriter(device)
                .writeImage(cube360GfxDs, 11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            cubeSamp, dummyView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .flush();
        }
    }

    // 1. UBO buffer
    if (cube360UBO.buffer == VK_NULL_HANDLE) {
        cube360UBO = createBuffer(sizeof(UniformObject),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    // Helper to destroy and recreate a buffer if its size is insufficient
    auto ensureBufferSize = [&](Buffer& buf, VkDeviceSize needed,
                                VkBufferUsageFlags usage, const char* label) {
        if (buf.buffer != VK_NULL_HANDLE) {
            VkMemoryRequirements reqs;
            vkGetBufferMemoryRequirements(device, buf.buffer, &reqs);
            if (reqs.size >= needed) return; // already large enough
            // The old buffer may still be referenced by command buffers that are
            // currently in flight (the cube360 culling pass runs every frame).
            // Destroying it synchronously here is the VUID-vkDestroyBuffer-buffer-00922
            // crash; defer the destruction until the GPU is idle instead.
            Buffer old = buf;
            deferDestroyUntilAllPending([old, this]() mutable {
                destroyBuffer(old);
            });
            buf = Buffer{};
        }
        buf = createBuffer(needed, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    };

    // 2. Solid culling buffers (reallocate if mesh count grew)
    IndirectRenderer &solidInd = sceneRenderer->solidRenderer->getIndirectRenderer();
    uint32_t solidCmds = std::max(static_cast<uint32_t>(solidInd.getMeshCount()), 1u);
    VkDeviceSize compactSize = sizeof(VkDrawIndexedIndirectCommand) * solidCmds;
    ensureBufferSize(cube360Compact, compactSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        "cube360Compact");
    ensureBufferSize(cube360Visible, std::max(sizeof(uint32_t), VkDeviceSize(4)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        "cube360Visible");

    // 3. Water culling buffers (reallocate if mesh count grew)
    IndirectRenderer &waterInd = sceneRenderer->waterRenderer->getIndirectRenderer();
    uint32_t waterCmds = std::max(static_cast<uint32_t>(waterInd.getMeshCount()), 1u);
    VkDeviceSize waterCompactSize = sizeof(VkDrawIndexedIndirectCommand) * waterCmds;
    ensureBufferSize(cube360WaterCompact, waterCompactSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        "cube360WaterCompact");
    ensureBufferSize(cube360WaterVisible, std::max(sizeof(uint32_t), VkDeviceSize(4)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        "cube360WaterVisible");

    // 4. Graphics descriptor set (mirrors main DS but uses cube360UBO)
    if (cube360GfxDs == VK_NULL_HANDLE) {
        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 };
        VkDescriptorPoolSize ps2{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 9 };
        VkDescriptorPoolSize ps3{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
        VkDescriptorPoolSize poolSizes[] = {ps, ps2, ps3};

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        poolInfo.maxSets = 1;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;

        VkDescriptorPool gfxPool;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &gfxPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create cubemap GFX descriptor pool");
        resources.addDescriptorPool(gfxPool, "cubemap gfx pool");

        VkDescriptorSetLayout gfxLayout = getDescriptorSetLayout();
        VkDescriptorSetAllocateInfo ainfo{};
        ainfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ainfo.descriptorPool = gfxPool;
        ainfo.descriptorSetCount = 1;
        ainfo.pSetLayouts = &gfxLayout;
        if (vkAllocateDescriptorSets(device, &ainfo, &cube360GfxDs) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate cubemap GFX descriptor set");
        resources.addDescriptorSet(cube360GfxDs, "cubemap gfx DS");

        // Write descriptor set bindings using DescriptorWriter
        {
            DescriptorWriter gfxWriter(device);
            gfxWriter.writeBuffer(cube360GfxDs, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                  cube360UBO.buffer, 0, sizeof(UniformObject));

            auto addImg = [&](uint32_t binding, VkSampler sampler, VkImageView view, VkImageLayout layout) {
                if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
                gfxWriter.writeImage(cube360GfxDs, binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                     sampler, view, layout);
            };

            if (textureArrayManager.albedoSampler != VK_NULL_HANDLE) {
                addImg(1, textureArrayManager.albedoSampler, textureArrayManager.albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                addImg(2, textureArrayManager.normalSampler, textureArrayManager.normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                addImg(3, textureArrayManager.bumpSampler, textureArrayManager.bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                addImg(12, textureArrayManager.roughnessSampler, textureArrayManager.roughnessArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                addImg(13, textureArrayManager.aoSampler, textureArrayManager.aoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
            addImg(4, sceneRenderer->shadowMapper->getShadowMapSampler(), sceneRenderer->shadowMapper->getShadowMapView(0), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(8, sceneRenderer->shadowMapper->getShadowMapSampler(), sceneRenderer->shadowMapper->getShadowMapView(1), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(9, sceneRenderer->shadowMapper->getShadowMapSampler(), sceneRenderer->shadowMapper->getShadowMapView(2), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            if (sceneRenderer->solid360Renderer) {
                VkImageView dummyCubeView = sceneRenderer->solid360Renderer->getDummyCubeView();
                VkSampler cubeSampler = sceneRenderer->solid360Renderer->getSolid360Sampler();
                if (dummyCubeView != VK_NULL_HANDLE && cubeSampler != VK_NULL_HANDLE)
                    addImg(11, cubeSampler, dummyCubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }

            if (sceneRenderer->materialsBuffer.buffer != VK_NULL_HANDLE)
                gfxWriter.writeBuffer(cube360GfxDs, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      sceneRenderer->materialsBuffer.buffer, 0, VK_WHOLE_SIZE);
            if (sceneRenderer->waterParamsBuffer_.buffer != VK_NULL_HANDLE)
                gfxWriter.writeBuffer(cube360GfxDs, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      sceneRenderer->waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE);
            {
                Buffer skyBuf = sceneRenderer->skyRenderer->getSkyUniformBuffer();
                if (skyBuf.buffer != VK_NULL_HANDLE)
                    gfxWriter.writeBuffer(cube360GfxDs, 6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                          skyBuf.buffer, 0, sizeof(SkyUniform));
            }
            if (sceneRenderer->waterRenderUBOBuffer_.buffer != VK_NULL_HANDLE)
                gfxWriter.writeBuffer(cube360GfxDs, 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                      sceneRenderer->waterRenderUBOBuffer_.buffer, 0, sizeof(WaterRenderUBO));

            gfxWriter.flush();
        }
        cube360TexVersion = textureArrayManager.getVersion();
    }

    // Refresh texture bindings on cube360GfxDs if texture arrays were re-allocated
    // (e.g. after TextureMixer generates new layers). Without this, the cubemap
    // capture would sample stale/deleted image views, producing wrong reflections.
    if (cube360GfxDs != VK_NULL_HANDLE && cube360TexVersion != textureArrayManager.getVersion()) {
        cube360TexVersion = textureArrayManager.getVersion();
        DescriptorWriter texWriter(device);
        auto addImg = [&](uint32_t binding, VkSampler sampler, VkImageView view, VkImageLayout layout) {
            if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
            texWriter.writeImage(cube360GfxDs, binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 sampler, view, layout);
        };
        if (textureArrayManager.albedoSampler != VK_NULL_HANDLE) {
            addImg(1, textureArrayManager.albedoSampler, textureArrayManager.albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(2, textureArrayManager.normalSampler, textureArrayManager.normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(3, textureArrayManager.bumpSampler, textureArrayManager.bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(12, textureArrayManager.roughnessSampler, textureArrayManager.roughnessArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(13, textureArrayManager.aoSampler, textureArrayManager.aoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        if (sceneRenderer && sceneRenderer->shadowMapper) {
            addImg(4, sceneRenderer->shadowMapper->getShadowMapSampler(), sceneRenderer->shadowMapper->getShadowMapView(0), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(8, sceneRenderer->shadowMapper->getShadowMapSampler(), sceneRenderer->shadowMapper->getShadowMapView(1), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(9, sceneRenderer->shadowMapper->getShadowMapSampler(), sceneRenderer->shadowMapper->getShadowMapView(2), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        texWriter.flush();
    }

    // 5. Solid compute descriptor set — allocate lazily, always refresh buffer bindings
    {
        VkDescriptorSetLayout dsLayout = solidInd.getComputeDescriptorSetLayout();
        if (dsLayout != VK_NULL_HANDLE && cube360ComputeDs == VK_NULL_HANDLE) {
            VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
            VkDescriptorPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.poolSizeCount = 1; pci.pPoolSizes = &ps; pci.maxSets = 1;
            pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
            VkDescriptorPool pool;
            if (vkCreateDescriptorPool(device, &pci, nullptr, &pool) == VK_SUCCESS) {
                resources.addDescriptorPool(pool, "cubemap compute pool");
                VkDescriptorSetAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &dsLayout;
                if (vkAllocateDescriptorSets(device, &ai, &cube360ComputeDs) == VK_SUCCESS)
                    resources.addDescriptorSet(cube360ComputeDs, "cubemap compute DS");
            }
        }
        if (cube360ComputeDs != VK_NULL_HANDLE) {
            DescriptorWriter(device)
                .writeBuffer(cube360ComputeDs, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             solidInd.getIndirectBuffer().buffer, 0, VK_WHOLE_SIZE)
                .writeBuffer(cube360ComputeDs, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             cube360Compact.buffer, 0, VK_WHOLE_SIZE)
                .writeBuffer(cube360ComputeDs, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             solidInd.getBoundsBuffer().buffer, 0, VK_WHOLE_SIZE)
                .writeBuffer(cube360ComputeDs, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             cube360Visible.buffer, 0, VK_WHOLE_SIZE)
                .flush();
        }
    }

    // 6. Water compute descriptor set
    {
        VkDescriptorSetLayout wDsLayout = waterInd.getComputeDescriptorSetLayout();
        if (wDsLayout != VK_NULL_HANDLE && cube360WaterComputeDs == VK_NULL_HANDLE) {
            VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
            VkDescriptorPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.poolSizeCount = 1; pci.pPoolSizes = &ps; pci.maxSets = 1;
            pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
            VkDescriptorPool pool;
            if (vkCreateDescriptorPool(device, &pci, nullptr, &pool) == VK_SUCCESS) {
                resources.addDescriptorPool(pool, "cubemap water compute pool");
                VkDescriptorSetAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &wDsLayout;
                if (vkAllocateDescriptorSets(device, &ai, &cube360WaterComputeDs) == VK_SUCCESS)
                    resources.addDescriptorSet(cube360WaterComputeDs, "cubemap water compute DS");
            }
        }
        if (cube360WaterComputeDs != VK_NULL_HANDLE) {
            DescriptorWriter(device)
                .writeBuffer(cube360WaterComputeDs, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             waterInd.getIndirectBuffer().buffer, 0, VK_WHOLE_SIZE)
                .writeBuffer(cube360WaterComputeDs, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             cube360WaterCompact.buffer, 0, VK_WHOLE_SIZE)
                .writeBuffer(cube360WaterComputeDs, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             waterInd.getBoundsBuffer().buffer, 0, VK_WHOLE_SIZE)
                .writeBuffer(cube360WaterComputeDs, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             cube360WaterVisible.buffer, 0, VK_WHOLE_SIZE)
                .flush();
        }
    }
}

void MyApp::postSubmit() {
    if (textureMixer) {
        textureMixer->flushPendingRequests(this);
        textureMixer->pollPendingGenerations(this);
    }

    // If a rebuild was requested from the UI (Apply Brush), perform it now
    // after the frame was submitted so GPU fences can be waited on safely.
    if (!isLoading && settings.animateBrush) {
        // Animate the selected brush entry along a circular trajectory, then
        // rebuild the brush scene from it. Runs in postSubmit (after submit)
        // so the indirect-buffer fill (this frame) is fenced off from the
        // indirect draw (next frame), avoiding a READ_AFTER_WRITE hazard.
        updateBrushAnimation(lastFrameDelta);
        rebuildBrushScene();
    } else if (brushRebuildPending) {
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
