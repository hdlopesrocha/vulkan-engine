#include "vulkan/VulkanApp.hpp"
#include "utils/FileReader.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <chrono>
#include "math/Camera.hpp"
#include "math/Light.hpp"
#include "events/EventManager.hpp"
#include "events/KeyboardPublisher.hpp"
#include "events/GamepadPublisher.hpp"
#include "events/TranslateCameraEvent.hpp"
#include "events/RotateCameraEvent.hpp"
#include "events/CloseWindowEvent.hpp"
#include "events/ToggleFullscreenEvent.hpp"
#include "vulkan/TextureArrayManager.hpp"
#include "vulkan/TextureTriple.hpp"
#include "vulkan/AtlasManager.hpp"
#include "utils/BillboardManager.hpp"
#include "math/SphereModel.hpp"
#include "vulkan/TextureMixer.hpp"
#include "widgets/AnimatedTextureWidget.hpp"
#include "vulkan/ShadowMapper.hpp"
#include "vulkan/ShadowParams.hpp"
#include "vulkan/IndirectRenderer.hpp"
#include "vulkan/WaterRenderer.hpp"
#include "widgets/WidgetManager.hpp"
#include "widgets/CameraWidget.hpp"
#include "widgets/DebugWidget.hpp"
#include "widgets/ShadowMapWidget.hpp"
#include "widgets/TextureViewerWidget.hpp"
#include "widgets/SettingsWidget.hpp"
#include "widgets/LightWidget.hpp"
#include "widgets/SkyWidget.hpp"
#include "widgets/WaterWidget.hpp"
#include "widgets/RenderPassDebugWidget.hpp"
#include "vulkan/SkySphere.hpp"
#include "vulkan/SceneRenderer.hpp"
#include "vulkan/DescriptorSetBuilder.hpp"
#include "vulkan/SkyRenderer.hpp"
#include "widgets/VegetationAtlasEditor.hpp"
#include "widgets/BillboardCreator.hpp"
#include "widgets/VulkanObjectsWidget.hpp"
#include <string>
#include <memory>
#include <iostream>
#include <cmath>
#include <mutex>
#include "utils/LocalScene.hpp"
#include "utils/MainSceneLoader.hpp"

#include "Uniforms.hpp"
#include "vulkan/MaterialManager.hpp"

#include "vulkan/VertexBufferObjectBuilder.hpp"
#include "vulkan/Model3DVersion.hpp"
#include "vulkan/Model3D.hpp"

class MyApp : public VulkanApp, public IEventHandler {
    LocalScene * mainScene;
    SceneRenderer sceneRenderer;

    // Profiling infrastructure
    VkQueryPool queryPool = VK_NULL_HANDLE;
    static constexpr uint32_t QUERY_COUNT = 12; // timestamps for different stages
    float timestampPeriod = 0.0f; // nanoseconds per tick
    // Profiling results (in milliseconds)
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
    bool profilingEnabled = true;

    public:
        MyApp() : sceneRenderer(this) {}


            void postSubmit() override {
                // Read GPU timestamp results from previous frame
                if (profilingEnabled && queryPool != VK_NULL_HANDLE) {
                    uint64_t timestamps[QUERY_COUNT];
                    VkResult result = vkGetQueryPoolResults(
                        getDevice(),
                        queryPool,
                        0, 8,  // Read queries 0-7
                        sizeof(timestamps),
                        timestamps,
                        sizeof(uint64_t),
                        VK_QUERY_RESULT_64_BIT
                    );
                    
                    if (result == VK_SUCCESS) {
                        // Convert timestamp deltas to milliseconds
                        // timestampPeriod is in nanoseconds per tick
                        float nsToMs = timestampPeriod / 1000000.0f;
                        
                        profileShadowCull = (timestamps[1] - timestamps[0]) * nsToMs;
                        profileShadowDraw = (timestamps[2] - timestamps[1]) * nsToMs;
                        profileSky = (timestamps[3] - timestamps[2]) * nsToMs;
                        profileMainCull = (timestamps[4] - timestamps[3]) * nsToMs;
                        profileDepthPrepass = (timestamps[5] - timestamps[4]) * nsToMs;
                        profileMainDraw = (timestamps[6] - timestamps[5]) * nsToMs;
                        profileImGui = (timestamps[7] - timestamps[6]) * nsToMs;
                    }
                }
            }

        // postSubmit() override removed to disable slow per-frame shadow readback
        
    // Pipelines moved into SceneRenderer
    // materials list (replaces legacy TextureManager storage) - each entry corresponds
    // to a layer/index in `textureArrayManager` when arrays are used.
    std::vector<MaterialProperties> materials;
    // Mixer parameters for editable textures (persistent storage — avoid dangling refs)
    std::vector<MixerParameters> mixerParams;
    // GPU-side texture arrays (albedo/normal/bump) for sampler2DArray usage
    TextureArrayManager textureArrayManager;
    
    // GPU-side texture array for vegetation (albedo/normal/opacity)
    TextureArrayManager vegetationTextureArrayManager;
    
    // Atlas manager for vegetation tile definitions
    AtlasManager vegetationAtlasManager;
    
    // Billboard manager for creating billboards from atlas tiles
    BillboardManager billboardManager;
    
    // Widget manager (handles all UI windows)
    WidgetManager widgetManager;
    
    // Widgets (shared pointers for widget manager)
    std::shared_ptr<TextureViewer> textureViewer;
    std::shared_ptr<TextureMixer> textureMixer; // Vulkan compute + textures
    std::shared_ptr<AnimatedTextureWidget> editableTexturesWidget; // ImGui wrapper
    std::shared_ptr<SettingsWidget> settingsWidget;
    std::shared_ptr<BillboardCreator> billboardCreator;
    std::shared_ptr<SkyWidget> skyWidget;

    // UI: currently selected texture triple for preview
    size_t currentTextureIndex = 0;
    size_t editableLayerIndex = SIZE_MAX; // Texture array layer index reserved for editable textures

    // Shadow mapping moved into SceneRenderer
    ShadowParams shadowParams;
    
    // Water rendering moved into SceneRenderer
    WaterParams waterParams;
    std::shared_ptr<WaterWidget> waterWidget;
    std::shared_ptr<RenderPassDebugWidget> renderPassDebugWidget;
    float waterTime = 0.0f;
    float lastDeltaTime = 0.016f;  // Last frame's delta time for use in draw()
    
    // Light (controlled by LightWidget)
    Light light = Light(glm::vec3(-1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f), 1.0f);
    // Camera
    Camera camera = Camera(glm::vec3(3456, 915, -750), Math::eulerToQuat(0, 0, 0));
    // Event manager for app-wide pub/sub
    EventManager eventManager;
    KeyboardPublisher keyboard;
    GamepadPublisher gamepad;
    
    // PassUBO moved into SceneRenderer
    
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet shadowPassDescriptorSet = VK_NULL_HANDLE; // Shadow pass uses this (no shadow map binding)
    MaterialManager materialManager;
    size_t materialCount = 0;
    
    // static parts of the UBO that don't vary per-cube
    UniformObject uboStatic{};
    // textureImage is now owned by TextureManager
    //VertexBufferObject vertexBufferObject; // now owned by CubeMesh

    void setup() override {
        // Subscribe camera and app to event manager
        eventManager.subscribe(&camera);
        eventManager.subscribe(this);

        // Use ImGui dark theme for a darker UI appearance
        ImGui::StyleColorsDark();

        // Convert ImGui style colors from sRGB to linear because swapchain uses VK_FORMAT_B8G8R8A8_SRGB
        // which applies gamma correction on output. ImGui colors are defined in sRGB, so we need to
        // linearize them first to avoid double gamma correction (which makes UI look washed out).
        {
            ImGuiStyle& style = ImGui::GetStyle();
            for (int i = 0; i < ImGuiCol_COUNT; i++) {
                ImVec4& col = style.Colors[i];
                // sRGB -> linear conversion for RGB channels (alpha stays linear)
                auto srgbToLinear = [](float c) {
                    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
                };
                col.x = srgbToLinear(col.x);
                col.y = srgbToLinear(col.y);
                col.z = srgbToLinear(col.z);
                // col.w (alpha) remains unchanged
            }
        }

        // Initialize shadow mapper (moved after pipeline creation below)
        
        // Initialize pipelines and scene renderer internals
        sceneRenderer.createPipelines();
        sceneRenderer.init(this);

        // Now that pipelines (and the app pipeline layout) have been created, SceneRenderer was initialized above

        // initialize texture array manager with room for 24 layers of 1024x1024
        // (we no longer use the legacy TextureManager for storage; `materials` tracks per-layer properties)
        textureArrayManager.allocate(24, 1024, 1024, this);
        std::vector<size_t> loadedIndices;

        // Explicit per-name loads (one-by-one) with realistic material properties
        const std::vector<std::pair<std::array<const char*,3>, MaterialProperties>> specs = {
            {{"textures/pixel_color.jpg", "textures/pixel_normal.jpg", "textures/pixel_bump.jpg"}, MaterialProperties{false, false, 0.01f, 1.0f, 0.15f, 0.3f, 16.0f, 0.0f, 0.0f, true, 0.01f, 0.01f}},
            {{"textures/bricks_color.jpg", "textures/bricks_normal.jpg", "textures/bricks_bump.jpg"}, MaterialProperties{false, false, 0.08f, 4.0f, 0.12f, 0.5f, 32.0f, 0.0f, 0.0f, true, 0.01f, 0.01f}},
            {{"textures/dirt_color.jpg", "textures/dirt_normal.jpg", "textures/dirt_bump.jpg"}, MaterialProperties{false, false, 0.05f, 1.0f, 0.15f, 0.5f, 32.0f, 0.0f, 0.0f, true, 0.01f, 0.01f}},
            {{"textures/forest_color.jpg", "textures/forest_normal.jpg", "textures/forest_bump.jpg"}, MaterialProperties{false, false, 0.06f, 1.0f, 0.18f, 0.5f, 32.0f, 0.0f, 0.0f, true, 0.01f, 0.01f}},
            {{"textures/grass_color.jpg", "textures/grass_normal.jpg", "textures/grass_bump.jpg"}, MaterialProperties{false, false, 0.04f, 1.0f, 0.5f, 0.5f, 32.0f, 0.0f, 0.0f, true, 0.01f, 0.01f}},
            {{"textures/lava_color.jpg", "textures/lava_normal.jpg", "textures/lava_bump.jpg"}, MaterialProperties{false, false, 0.03f, 1.0f, 0.4f, 0.5f, 32.0f, 0.0f, 0.0f, true, 0.01f, 0.01f}},
            {{"textures/metal_color.jpg", "textures/metal_normal.jpg", "textures/metal_bump.jpg"}, MaterialProperties{true, false, 0.5f, 1.0f, 0.1f, 0.8f, 64.0f, 0.0f, 0.0f, true, 0.01f, 0.01f}},
            {{"textures/rock_color.jpg", "textures/rock_normal.jpg", "textures/rock_bump.jpg"}, MaterialProperties{false, false, 0.1f, 1.0f, 0.1f, 0.4f, 32.0f, 0.0f, 0.0f, true, 0.01f, 0.01f}},
            {{"textures/sand_color.jpg", "textures/sand_normal.jpg", "textures/sand_bump.jpg"}, MaterialProperties{false, false, 0.03f, 1.0f, 0.5f, 0.2f, 16.0f, 0.0f, 0.0f, true, 0.01f, 0.01f}},
            {{"textures/snow_color.jpg", "textures/snow_normal.jpg", "textures/snow_bump.jpg"}, MaterialProperties{false, false, 0.04f, 1.0f, 0.1f, 0.1f, 8.0f, 0.0f, 0.0f, true, 0.01f, 0.01f}},
            {{"textures/soft_sand_color.jpg", "textures/soft_sand_normal.jpg", "textures/soft_sand_bump.jpg"}, MaterialProperties{false, false, 0.025f, 1.0f, 0.22f, 0.3f, 16.0f, 0.0f, 0.0f, true, 0.01f, 0.01f}}
        };

        for (const auto &entry : specs) {
            const auto &files = entry.first;
            const auto &matSpec = entry.second;
            // record material properties and load into GPU texture arrays
            size_t idx = materials.size();
            materials.push_back(matSpec);
            loadedIndices.push_back(idx);
            // also load into the new TextureArrayManager (GPU-backed 2D texture arrays)
            textureArrayManager.load(const_cast<char*>(files[0]), const_cast<char*>(files[1]), const_cast<char*>(files[2]));
        }

        // Add editable textures as an entry in `materials` and keep their images managed by the EditableTextureSet
        // Allocate array layers FIRST so MixerParameters.targetLayer contains a valid array layer index
        uint32_t editableLayerA = textureArrayManager.create();
        

        // Apply editable material properties in a single-line initializer
        mixerParams = {
            MixerParameters({ editableLayerA, 4, 8, 8.0f, 8, 0.5f, 2.0f, 0.0f, 5.0f, 42, 0.0f }),
            MixerParameters({ textureArrayManager.create(), 4, 9, 8.0f, 8, 0.5f, 2.0f, 0.0f, 5.0f, 42, 0.0f }),
            MixerParameters({ textureArrayManager.create(), 4, 7, 8.0f, 8, 0.5f, 2.0f, 0.0f, 5.0f, 42, 0.0f }),
            MixerParameters({ textureArrayManager.create(), 7, 9, 8.0f, 8, 0.5f, 2.0f, 0.0f, 5.0f, 42, 0.0f }),
            MixerParameters({ textureArrayManager.create(), 7, 8, 8.0f, 8, 0.5f, 2.0f, 0.0f, 5.0f, 42, 0.0f })        
        };
        materials.push_back(MaterialProperties{ false, false, 0.2f, 16.0f, 0.5f, 0.05f, 32.0f, 0.0f, 0.0f, true, 0.01f, 0.01f});
        materials.push_back(MaterialProperties{ false, false, 0.3f, 17.0f, 0.3f, 0.03f, 16.0f, 0.0f, 0.0f, true, 0.01f, 0.01f});
        materials.push_back(MaterialProperties{ false, false, 0.3f, 17.0f, 0.3f, 0.03f, 16.0f, 0.0f, 0.0f, true, 0.01f, 0.01f});
        materials.push_back(MaterialProperties{ false, false, 0.3f, 17.0f, 0.3f, 0.03f, 16.0f, 0.0f, 0.0f, true, 0.01f, 0.01f});
        materials.push_back(MaterialProperties{ false, false, 0.3f, 17.0f, 0.3f, 0.03f, 16.0f, 0.0f, 0.0f, true, 0.01f, 0.01f});

        // Initialize Vulkan-side editable textures BEFORE creating descriptor sets
        textureMixer = std::make_shared<TextureMixer>();
        textureMixer->init(this, 1024, 1024, &textureArrayManager);
        // Generate initial textures directly into the allocated array layers
        textureMixer->generateInitialTextures(mixerParams);


        // Load vegetation textures (albedo/normal/opacity triples) and initialize MaterialProperties
        // Note: We use the height slot for opacity masks
        const std::vector<std::pair<std::array<const char*,3>, MaterialProperties>> vegSpecs = {
            {{"textures/vegetation/foliage_color.jpg", "textures/vegetation/foliage_normal.jpg", "textures/vegetation/foliage_opacity.jpg"}, MaterialProperties{false, false, 0.0f, 1.0f, 0.3f, 0.3f, 8.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
            {{"textures/vegetation/grass_color.jpg",   "textures/vegetation/grass_normal.jpg",   "textures/vegetation/grass_opacity.jpg"},   MaterialProperties{false, false, 0.0f, 1.0f, 0.35f, 0.3f, 8.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
            {{"textures/vegetation/wild_color.jpg",    "textures/vegetation/wild_normal.jpg",    "textures/vegetation/wild_opacity.jpg"},    MaterialProperties{false, false, 0.0f, 1.0f, 0.32f, 0.3f, 8.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}}
        };

        // allocate vegetation GPU texture arrays sized to the number of veg layers
        vegetationTextureArrayManager.allocate(static_cast<uint32_t>(vegSpecs.size()), 1024, 1024, this);
        // append vegetation materials to `materials` and load texture data into vegetation arrays
        for (const auto &entry : vegSpecs) {
            const auto &files = entry.first;
            const auto &mp = entry.second;
            materials.push_back(mp);
            vegetationTextureArrayManager.load(const_cast<char*>(files[0]), const_cast<char*>(files[1]), const_cast<char*>(files[2]));
        }

        // Auto-detect tiles from vegetation opacity maps
        std::cout << "Auto-detecting vegetation tiles from opacity maps..." << std::endl;
        int foliageTiles = vegetationAtlasManager.autoDetectTiles(0, "textures/vegetation/foliage_opacity.jpg", 10);
        std::cout << "  Foliage: detected " << foliageTiles << " tiles" << std::endl;
        
        int grassTiles = vegetationAtlasManager.autoDetectTiles(1, "textures/vegetation/grass_opacity.jpg", 10);
        std::cout << "  Grass: detected " << grassTiles << " tiles" << std::endl;
        
        int wildTiles = vegetationAtlasManager.autoDetectTiles(2, "textures/vegetation/wild_opacity.jpg", 10);
        std::cout << "  Wild: detected " << wildTiles << " tiles" << std::endl;

        // Create ImGui widget that wraps the Vulkan editable textures and add it to widget manager
        editableTexturesWidget = std::make_shared<AnimatedTextureWidget>(textureMixer, mixerParams, "Editable Textures");
        widgetManager.addWidget(editableTexturesWidget);
        // Show the editable textures widget by default so users see the previews on startup
        //editableTexturesWidget->show();

        

        // We already allocated array layers (`editableLayerA`/`editableLayerB`) before
        // generating initial textures. Use the first allocated layer as the default
        // editable layer for the widget previews.
        editableLayerIndex = editableLayerA;
        // Inform TextureMixer which layer to use for ImGui previews
        textureMixer->setEditableLayer(editableLayerIndex);
        // remove any zeros that might come from failed loads (TextureManager may throw or return an index; assume valid indices)
        if (!loadedIndices.empty()) currentTextureIndex = loadedIndices[0];
        
        std::cout << "[DEBUG] Created mainPassUBO.buffer: handle=" << sceneRenderer.mainPassUBO.buffer.buffer 
              << " memory=" << sceneRenderer.mainPassUBO.buffer.memory 
              << " size=" << sizeof(UniformObject) << std::endl;

        // Initialize mainPassUBO with uboStatic so descriptor sets see valid data
        sceneRenderer.mainPassUBO.data = uboStatic;
        updateUniformBuffer(sceneRenderer.mainPassUBO.buffer, &sceneRenderer.mainPassUBO.data, sizeof(UniformObject));
        std::cout << "[DEBUG] Initialized mainPassUBO with uboStatic" << std::endl;
        
        // create descriptor sets - one per texture triple
        size_t tripleCount = materials.size();
        if (tripleCount == 0) tripleCount = 1; // ensure at least one
        // Create/upload GPU-side material buffer (packed vec4-friendly struct)
        updateMaterials();
        // Delegate descriptor pool/allocation and writes to SceneRenderer
        sceneRenderer.createDescriptorSets(materialManager, textureArrayManager, descriptorSet, shadowPassDescriptorSet, tripleCount);
        
        // Per-instance descriptor sets removed — use per-material/global descriptor sets instead.
        // Material SSBO is bound into a single global descriptor set; updateMaterials() will
        // refresh that set if it exists.
        updateMaterials();
                
        // Initialize widgets
        textureViewer = std::make_shared<TextureViewer>();
        textureViewer->init(&textureArrayManager, &materials);
        // Rebuild GPU material buffer when materials are modified via the texture viewer
        textureViewer->setOnMaterialChanged([this](size_t idx) {
            (void)idx; // currently we rebuild entire buffer
            updateMaterials();
        });
        widgetManager.addWidget(textureViewer);
        
        // Hook Vulkan editable texture regeneration callback so materials are refreshed
        textureMixer->setOnTextureGenerated([this]() {
            printf("Editable textures regenerated (layer %u)\n", static_cast<unsigned int>(editableLayerIndex));
            // Textures are written directly into the texture arrays by the compute shader
            updateMaterials();
        });
        
        // Create other widgets
        auto cameraWidget = std::make_shared<CameraWidget>(&camera);
        widgetManager.addWidget(cameraWidget);
        
        auto debugWidget = std::make_shared<DebugWidget>(&materials, &camera, &currentTextureIndex);
        widgetManager.addWidget(debugWidget);
        
        auto shadowWidget = std::make_shared<ShadowMapWidget>(&sceneRenderer.shadowMapper, &shadowParams);
        widgetManager.addWidget(shadowWidget);
        
        // Create settings widget
        settingsWidget = std::make_shared<SettingsWidget>();
        // Hook debug UI button to trigger a shadow depth readback
        settingsWidget->setDumpShadowDepthCallback([this]() {
            sceneRenderer.shadowMapper.readbackShadowDepth();
            std::cerr << "[Debug] Wrote bin/shadow_depth.pgm (manual trigger).\n";
        });
        widgetManager.addWidget(settingsWidget);
        
        // Create light control widget
        auto lightWidget = std::make_shared<LightWidget>(&light);
        widgetManager.addWidget(lightWidget);
        // Create sky widget (controls colors and parameters)
        skyWidget = std::make_shared<SkyWidget>();
        widgetManager.addWidget(skyWidget);
        // Create water widget (controls water rendering parameters)
        waterWidget = std::make_shared<WaterWidget>(&waterParams);
        widgetManager.addWidget(waterWidget);
        
        renderPassDebugWidget = std::make_shared<RenderPassDebugWidget>(this, &sceneRenderer.waterRenderer);
        widgetManager.addWidget(renderPassDebugWidget);
        // Vulkan objects widget
        auto vulkanObjectsWidget = std::make_shared<VulkanObjectsWidget>(this);
        widgetManager.addWidget(vulkanObjectsWidget);
        if (skyWidget) {
            // Sky UBO is managed by SkySphere; SkyWidget values will be read and uploaded there.
        }

        // Initialize sky manager which creates and binds the sky UBO into descriptor sets
        sceneRenderer.init(nullptr, skyWidget.get(), descriptorSet);
        
        // Create vegetation atlas editor widget
        auto vegAtlasEditor = std::make_shared<VegetationAtlasEditor>(&vegetationTextureArrayManager, &vegetationAtlasManager);
        widgetManager.addWidget(vegAtlasEditor);
        
        // Create billboard creator widget
        billboardCreator = std::make_shared<BillboardCreator>(&billboardManager, &vegetationAtlasManager, &vegetationTextureArrayManager);
        billboardCreator->setVulkanApp(this);
        widgetManager.addWidget(billboardCreator);
        
        createCommandBuffers();
        mainScene = new LocalScene();

        MainSceneLoader mainSceneLoader = MainSceneLoader(&mainScene->transparentLayerChangeHandler, &mainScene->opaqueLayerChangeHandler);
        mainScene->loadScene(mainSceneLoader);

        // Load all chunk meshes into IndirectRenderer after scene loading
        std::cout << "[Setup] Loading all chunk meshes into IndirectRenderer..." << std::endl;
        auto meshLoadStart = std::chrono::steady_clock::now();
        size_t meshCount = 0;
        
        mainScene->forEachChunkNode(Layer::LAYER_OPAQUE, [this, &meshCount](std::vector<OctreeNodeData>& nodes) {
            for (auto& node : nodes) {
                NodeID id = reinterpret_cast<NodeID>(node.node);
                uint ver = node.node->version;
                
                // Generate mesh synchronously during setup
                mainScene->requestModel3D(Layer::LAYER_OPAQUE, node, [this, id, ver, &meshCount](const Geometry& mesh) {
                    uint32_t meshId = sceneRenderer.indirectRenderer.addMesh(this, mesh, glm::mat4(1.0f));
                    sceneRenderer.nodeModelVersions[id] = { meshId, ver };
                    meshCount++;
                });
            }
        });
        
        // Rebuild buffers and update descriptor once after all meshes are added
        sceneRenderer.indirectRenderer.rebuild(this);
        sceneRenderer.indirectRenderer.updateModelsDescriptorSet(this, VK_NULL_HANDLE);
        
        auto meshLoadEnd = std::chrono::steady_clock::now();
        double meshLoadTime = std::chrono::duration<double>(meshLoadEnd - meshLoadStart).count();
        std::cout << "[Setup] Loaded " << meshCount << " opaque meshes in " << meshLoadTime << "s" << std::endl;

        // Load all water (transparent) chunk meshes into WaterRenderer
        std::cout << "[Setup] Loading all water meshes into WaterRenderer..." << std::endl;
        auto waterMeshLoadStart = std::chrono::steady_clock::now();
        size_t waterMeshCount = 0;
        
        mainScene->forEachChunkNode(Layer::LAYER_TRANSPARENT, [this, &waterMeshCount](std::vector<OctreeNodeData>& nodes) {
            for (auto& node : nodes) {
                NodeID id = reinterpret_cast<NodeID>(node.node);
                uint ver = node.node->version;
                
                // Generate mesh synchronously during setup
                mainScene->requestModel3D(Layer::LAYER_TRANSPARENT, node, [this, id, ver, &waterMeshCount](const Geometry& mesh) {
                    uint32_t meshId = sceneRenderer.waterRenderer.getIndirectRenderer().addMesh(this, mesh, glm::mat4(1.0f));
                    sceneRenderer.waterNodeModelVersions[id] = { meshId, ver };
                    waterMeshCount++;
                });
            }
        });
        
        // Rebuild water buffers and update descriptor once after all meshes are added
        sceneRenderer.waterRenderer.getIndirectRenderer().rebuild(this);
        sceneRenderer.waterRenderer.getIndirectRenderer().updateModelsDescriptorSet(this, VK_NULL_HANDLE);
        
        auto waterMeshLoadEnd = std::chrono::steady_clock::now();
        double waterMeshLoadTime = std::chrono::duration<double>(waterMeshLoadEnd - waterMeshLoadStart).count();
        std::cout << "[Setup] Loaded " << waterMeshCount << " water meshes in " << waterMeshLoadTime << "s" << std::endl;

        // Create GPU timestamp query pool for profiling
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(getPhysicalDevice(), &props);
            timestampPeriod = props.limits.timestampPeriod; // nanoseconds per tick
            
            VkQueryPoolCreateInfo queryInfo{};
            queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
            queryInfo.queryCount = QUERY_COUNT;
            
            if (vkCreateQueryPool(getDevice(), &queryInfo, nullptr, &queryPool) != VK_SUCCESS) {
                std::cerr << "Warning: Failed to create timestamp query pool\n";
                queryPool = VK_NULL_HANDLE;
            }
        }
        
    };

    // Upload all materials into a GPU storage buffer (called once during setup)
    void updateMaterials() {
        materialCount = materials.size();
        if (materialCount == 0) return;

        // Allocate GPU buffer via MaterialManager and upload per-index materials
        materialManager.allocate(materialCount, this);
        for (size_t mi = 0; mi < materialCount; ++mi) {
            const MaterialProperties &mat = materials[mi];
            materialManager.update(mi, mat, this);
        }

        // Rebind material buffer into the SINGLE global material descriptor set (binding 5)
        // so shaders can index `materials[...]` and we avoid per-instance descriptor updates.
        VkDescriptorSet matDS = getMaterialDescriptorSet();
        if (matDS != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo materialBufInfo{ materialManager.getBuffer().buffer, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet matWrite{}; matWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; matWrite.dstSet = matDS; matWrite.dstBinding = 5; matWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; matWrite.descriptorCount = 1; matWrite.pBufferInfo = &materialBufInfo;
            updateDescriptorSet(matDS, { matWrite });
        }

    }

    // Sky updates are handled by SkySphere

    // IEventHandler: handle top-level window events like close/fullscreen
    void onEvent(const EventPtr &event) override {
        if (!event) return;
        if (std::dynamic_pointer_cast<CloseWindowEvent>(event)) {
            requestClose();
            return;
        }
        if (std::dynamic_pointer_cast<ToggleFullscreenEvent>(event)) {
            toggleFullscreen();
            return;
        }
    }

    // build ImGui UI (moved from VulkanApp)
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

            // Small top-left overlay under the main menu bar showing FPS and visible count (one per line)
            {
                ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
                ImGui::SetNextWindowBgAlpha(0.35f);
                float padding = 10.0f;
                float y = ImGui::GetFrameHeight() + 6.0f; // position just under the main menu bar
                ImGui::SetNextWindowPos(ImVec2(padding, y), ImGuiCond_Always);
                if (ImGui::Begin("##stats_overlay", nullptr, flags)) {
                    ImGui::Text("FPS: %.1f (%.2f ms)", imguiFps, imguiFps > 0 ? 1000.0f / imguiFps : 0.0f);
                    ImGui::Text("Loaded: %zu meshes", sceneRenderer.nodeModelVersions.size());
                    
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
                }
                ImGui::End();
            }
        }

        if (imguiShowDemo) ImGui::ShowDemoWindow(&imguiShowDemo);

        // Render all widgets
        widgetManager.renderAll();
    }

    void update(float deltaTime) override {
        auto cpuUpdateStart = std::chrono::high_resolution_clock::now();
        lastDeltaTime = deltaTime;  // Store for use in draw()
        
        // Process queued events (dispatch to handlers)
        eventManager.processQueued();
        
        // Check and apply V-Sync setting changes from UI
        if (settingsWidget) {
            setVSyncEnabled(settingsWidget->getVSyncEnabled());
        }

        float aspect = (float)getWidth() / (float) getHeight();
        glm::mat4 proj = glm::perspective(45.0f * 3.1415926f / 180.0f, aspect, 0.1f, 8192.0f);
        // flip Y for Vulkan clip space
        proj[1][1] *= -1.0f;
        camera.setProjection(proj);

        GLFWwindow* win = getWindow();
        if (win) {
            // Apply sensitivity settings from UI
            keyboard.setMoveSpeed(settingsWidget->getMoveSpeed());
            keyboard.setAngularSpeed(settingsWidget->getAngularSpeedDeg());
            keyboard.update(win, &eventManager, camera, deltaTime, settingsWidget->getFlipKeyboardRotation());
        }
        // Poll gamepad and publish camera movement events (apply sensitivity)
        gamepad.setMoveSpeed(settingsWidget->getMoveSpeed());
        gamepad.setAngularSpeed(settingsWidget->getAngularSpeedDeg());
        gamepad.update(&eventManager, camera, deltaTime, settingsWidget->getFlipGamepadRotation());

        // prepare static parts of the UBO (viewPos, light, material flags) - model and viewProjection will be set per-cube in draw()
        uboStatic.viewPos = glm::vec4(camera.getPosition(), 1.0f);
        // The UI light direction represents a vector TO the light. Shaders expect a light vector that points FROM the
        // light TOWARD the surface when performing lighting/shadow calculations. Send the direction to the GPU
        // so both lighting and shadow projection use the same convention.
        uboStatic.lightDir = glm::vec4(light.getDirection(), 0.0f);
        uboStatic.lightColor = glm::vec4(light.getColor() * light.getIntensity(), 1.0f);
        // Tessellation parameters from UI: near/far distances and min/max tess levels
        if (settingsWidget) {
            uboStatic.tessParams = glm::vec4(
                settingsWidget->getTessMinDistance(),
                settingsWidget->getTessMaxDistance(),
                settingsWidget->getTessMinLevel(),
                settingsWidget->getTessMaxLevel()
            );
        } else {
            uboStatic.tessParams = glm::vec4(10.0f, 200.0f, 1.0f, 32.0f);
        }

        // Compute light space matrix for shadow mapping using ShadowParams
        shadowParams.update(camera.getPosition(), light);
        uboStatic.lightSpaceMatrix = shadowParams.lightSpaceMatrix;
        
        auto cpuUpdateEnd = std::chrono::high_resolution_clock::now();
        profileCpuUpdate = std::chrono::duration<float, std::milli>(cpuUpdateEnd - cpuUpdateStart).count();
    };

    void draw(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo) override {
        uint32_t frameIdx = getCurrentFrame();
        auto cpuRecordStart = std::chrono::high_resolution_clock::now();
                
        // Rebuild renderer if any meshes were added/removed
        sceneRenderer.indirectRenderer.rebuild(this);

        
        // Rebuild water renderer if any meshes were added/removed
        sceneRenderer.waterRenderer.getIndirectRenderer().rebuild(this);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        
        // Reset and write timestamp queries for profiling
        if (queryPool != VK_NULL_HANDLE) {
            vkCmdResetQueryPool(commandBuffer, queryPool, 0, QUERY_COUNT);
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0); // Start
        }
    
        // First pass: Render shadow map (skip if shadows globally disabled)
        if (!settingsWidget || settingsWidget->getShadowsEnabled()) {
            // Prepare GPU cull before starting the shadow render pass (use light's view-projection for shadow frustum)
            sceneRenderer.indirectRenderer.prepareCull(commandBuffer, light.getViewProjectionMatrix());
            
            if (queryPool != VK_NULL_HANDLE) {
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 1); // After shadow cull
            }
            
            sceneRenderer.shadowMapper.beginShadowPass(commandBuffer, uboStatic.lightSpaceMatrix);
            
            // Update uniform buffer for shadow pass: set viewProjection = lightSpaceMatrix
            sceneRenderer.shadowPassUBO.data = uboStatic;
            sceneRenderer.shadowPassUBO.data.viewProjection = uboStatic.lightSpaceMatrix;
            sceneRenderer.shadowPassUBO.data.materialFlags = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
            sceneRenderer.shadowPassUBO.data.passParams = glm::vec4(1.0f, settingsWidget ? (settingsWidget->getShadowTessellationEnabled() ? 1.0f : 0.0f) : 0.0f, 0.0f, 0.0f);
            updateUniformBuffer(sceneRenderer.shadowPassUBO.buffer, &sceneRenderer.shadowPassUBO.data, sizeof(UniformObject));
            

            // Bind material/per-texture descriptor sets for shadow pass
            // Use shadowPassDescriptorSet which has ATTACHMENT_OPTIMAL layout for shadow map
            {
                VkDescriptorSet setsToBind[2];
                uint32_t bindCount = 0;
                VkDescriptorSet matDs = getMaterialDescriptorSet();
                if (matDs != VK_NULL_HANDLE) setsToBind[bindCount++] = matDs;
                if (shadowPassDescriptorSet != VK_NULL_HANDLE) setsToBind[bindCount++] = shadowPassDescriptorSet;
                if (bindCount > 0) vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipelineLayout(), 0, bindCount, setsToBind, 0, nullptr);
            }

            // Issue compacted draws after GPU frustum culling
            sceneRenderer.indirectRenderer.drawPrepared(commandBuffer, this);

            sceneRenderer.shadowMapper.endShadowPass(commandBuffer);
            
            if (queryPool != VK_NULL_HANDLE) {
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 2); // After shadow draw
            }
        } else {
            // Write placeholder timestamps if shadows disabled
            if (queryPool != VK_NULL_HANDLE) {
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 1);
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 2);
            }
        }
        
        // Update uniform buffer for main pass BEFORE starting the render pass
        // Include water time in passParams for water geometry shader
        waterTime += lastDeltaTime * waterParams.waveSpeed;
        waterParams.time = waterTime;
        
        sceneRenderer.mainPassUBO.data = uboStatic;
        sceneRenderer.mainPassUBO.data.viewProjection = camera.getViewProjectionMatrix();
        sceneRenderer.mainPassUBO.data.materialFlags.w = settingsWidget->getNormalMappingEnabled() ? 1.0f : 0.0f;
        sceneRenderer.mainPassUBO.data.shadowEffects = glm::vec4(0.0f, 0.0f, 0.0f,  settingsWidget->getShadowsEnabled() ? 1.0f : 0.0f);
        sceneRenderer.mainPassUBO.data.debugParams = glm::vec4((float)settingsWidget->getDebugMode(), 0.0f, 0.0f, 0.0f);
        sceneRenderer.mainPassUBO.data.triplanarSettings = glm::vec4(settingsWidget->getTriplanarThreshold(), settingsWidget->getTriplanarExponent(), 0.0f, 0.0f);
        // passParams: x=waterTime, y=tessEnabled, z=waveScale, w=noiseScale
        sceneRenderer.mainPassUBO.data.passParams = glm::vec4(
            waterTime,
            settingsWidget ? (settingsWidget->getTessellationEnabled() ? 1.0f : 0.0f) : 0.0f,
            waterParams.waveScale,
            waterParams.noiseScale
        );
        // DEBUG: Print passParams and timing every 30 frames
        static int frameCount = 0;
        if (frameCount++ % 30 == 0) {
            std::cout << "[DEBUG] waterTime=" << waterTime
                      << " lastDeltaTime=" << lastDeltaTime
                      << " ubo.passParams.time=" << sceneRenderer.mainPassUBO.data.passParams.x 
                      << " waveScale=" << sceneRenderer.mainPassUBO.data.passParams.z 
                      << " noiseScale=" << sceneRenderer.mainPassUBO.data.passParams.w 
                      << " | mainPassUBO.buffer.buffer=" << sceneRenderer.mainPassUBO.buffer.buffer << std::endl;
        }
        // Store additional water params in tessParams: x=waveSpeed, y=refractionStrength, z=fresnelPower, w=transparency
        sceneRenderer.mainPassUBO.data.tessParams = glm::vec4(
            waterParams.waveSpeed,
            waterParams.refractionStrength,
            waterParams.fresnelPower,
            waterParams.transparency
        );
        updateUniformBuffer(sceneRenderer.mainPassUBO.buffer, &sceneRenderer.mainPassUBO.data, sizeof(UniformObject));
        
        // Update water params UBO
        sceneRenderer.waterPassUBO.data.params1 = glm::vec4(
            waterParams.refractionStrength,
            waterParams.fresnelPower,
            waterParams.transparency,
            waterParams.foamDepthThreshold
        );
        sceneRenderer.waterPassUBO.data.params2 = glm::vec4(
            waterParams.waterTint,
            1.0f / waterParams.noiseScale,
            static_cast<float>(waterParams.noiseOctaves),
            waterParams.noisePersistence
        );
        sceneRenderer.waterPassUBO.data.params3 = glm::vec4(waterParams.noiseTimeSpeed, waterTime, waterParams.shoreStrength, waterParams.shoreFalloff);
        sceneRenderer.waterPassUBO.data.shallowColor = glm::vec4(waterParams.shallowColor, 0.0f);
        sceneRenderer.waterPassUBO.data.deepColor = glm::vec4(waterParams.deepColor, waterParams.foamIntensity);
        sceneRenderer.waterPassUBO.data.foamParams = glm::vec4(1.0f/waterParams.foamNoiseScale, static_cast<float>(waterParams.foamNoiseOctaves), waterParams.foamNoisePersistence, waterParams.foamTintIntensity);
        sceneRenderer.waterPassUBO.data.foamParams2 = glm::vec4(waterParams.foamBrightness, waterParams.foamContrast, 0.0f, 0.0f);
        sceneRenderer.waterPassUBO.data.foamTint = waterParams.foamTint;
        updateUniformBuffer(sceneRenderer.waterPassUBO.buffer, &sceneRenderer.waterPassUBO.data, sizeof(WaterParamsGPU));
        
        // DEBUG: Print UBO passParams after write
        //ubo.printPassParams();
        
        // Add a memory barrier to ensure uniform buffer write is visible to GPU
        VkMemoryBarrier memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            1, &memBarrier,
            0, nullptr,
            0, nullptr
        );

        // Prepare GPU cull for main pass BEFORE the render pass (vkCmdFillBuffer/barriers not allowed inside render pass)
        sceneRenderer.indirectRenderer.prepareCull(commandBuffer, camera.getViewProjectionMatrix());
        
        // Also prepare water cull before render pass
        if (sceneRenderer.waterRenderer.getIndirectRenderer().getMeshCount() > 0) {
            sceneRenderer.waterRenderer.getIndirectRenderer().prepareCull(commandBuffer, camera.getViewProjectionMatrix());
        }
        
        if (queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 4); // After main cull
        }

        // Check if we need offscreen rendering for water
        bool hasWater = sceneRenderer.waterRenderer.getIndirectRenderer().getMeshCount() > 0;
        
        // If we have water, render main scene to offscreen targets for sampling
        // Otherwise render directly to swapchain
        VkRenderPassBeginInfo mainPassInfo = renderPassInfo;
        VkFramebuffer targetFramebuffer = renderPassInfo.framebuffer;
        
        if (hasWater) {
            // Initialize per-frame offscreen render targets if needed
            sceneRenderer.waterRenderer.createRenderTargets(getWidth(), getHeight());
            
            // Use offscreen framebuffer for main scene (per-frame)
            mainPassInfo.renderPass = sceneRenderer.waterRenderer.getSceneRenderPass();
            mainPassInfo.framebuffer = sceneRenderer.waterRenderer.getSceneFramebuffer(frameIdx);
            mainPassInfo.renderArea.extent = {getWidth(), getHeight()};
        }

        // Begin main scene render pass (either offscreen or swapchain)
        vkCmdBeginRenderPass(commandBuffer, &mainPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        
        // set dynamic viewport/scissor to match current swapchain extent
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)getWidth();
        viewport.height = (float)getHeight();
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        // --- Render sky sphere first: large sphere centered at camera ---
        if (sceneRenderer.skyRenderer && descriptorSet != VK_NULL_HANDLE) {
            if (sceneRenderer.skySphere) sceneRenderer.skySphere->update();
            SkyMode skyMode = skyWidget ? skyWidget->getSkyMode() : SkyMode::Gradient;
            sceneRenderer.skyRenderer->render(commandBuffer, sceneRenderer.skyVBO, descriptorSet, sceneRenderer.mainPassUBO.buffer, sceneRenderer.mainPassUBO.data, camera.getViewProjectionMatrix(), skyMode);
        }
        
        if (queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, queryPool, 3); // After sky
        }


        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = { (uint32_t)getWidth(), (uint32_t)getHeight() };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        
        // Bind vertex/index buffers once for all subsequent draw calls in this render pass
        sceneRenderer.indirectRenderer.bindBuffers(commandBuffer);

        // Bind descriptor sets if available (material/global + per-texture)
        { 
            VkDescriptorSet setsToBind[2];
            uint32_t bindCount = 0;
            VkDescriptorSet matDs = getMaterialDescriptorSet();
            if (matDs != VK_NULL_HANDLE) setsToBind[bindCount++] = matDs;
            VkDescriptorSet perTexDs = VK_NULL_HANDLE;
            if (descriptorSet != VK_NULL_HANDLE) perTexDs = descriptorSet;
            if (perTexDs != VK_NULL_HANDLE) 
                setsToBind[bindCount++] = perTexDs;
            if (bindCount > 0) 
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipelineLayout(), 0, bindCount, setsToBind, 0, nullptr);
        }

        // --- Depth pre-pass: fill depth buffer with geometry depths (color writes disabled) ---
        if (sceneRenderer.depthPrePassPipeline != VK_NULL_HANDLE) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneRenderer.depthPrePassPipeline);
            // Ensure material/models descriptor is bound (models SSBO expected in material set)
            VkDescriptorSet matDs = getMaterialDescriptorSet();
            if (matDs != VK_NULL_HANDLE) {
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipelineLayout(), 0, 1, &matDs, 0, nullptr);
            }
            // Reuse culled draw commands - buffers already bound above
            sceneRenderer.indirectRenderer.drawIndirectOnly(commandBuffer, this);
        }
        
        if (queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, queryPool, 5); // After depth prepass
        }


        // Bind descriptor set and draw
        // Use a single pipeline (tessellation is enabled/disabled in the shader using mappingMode)
        if (settingsWidget->getWireframeEnabled() && sceneRenderer.graphicsPipelineWire != VK_NULL_HANDLE) 
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneRenderer.graphicsPipelineWire);
        else 
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneRenderer.graphicsPipeline);
     
        // Bind global material descriptor set (set 0) and per-material descriptor set (set 1)  
        {
            VkDescriptorSet setsToBind[2];
            uint32_t bindCount = 0;
            VkDescriptorSet matDs = getMaterialDescriptorSet();
            if (matDs != VK_NULL_HANDLE) {
                setsToBind[bindCount++] = matDs;
            }
            // Bind per-material descriptor set (fallback to instance descriptor set if needed)
            VkDescriptorSet perTexDs = VK_NULL_HANDLE;
            if (descriptorSet != VK_NULL_HANDLE) perTexDs = descriptorSet;
            // bind the single per-texture descriptor set if available
            if (perTexDs != VK_NULL_HANDLE) setsToBind[bindCount++] = perTexDs;
            if (bindCount == 2) {
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipelineLayout(), 0, 2, setsToBind, 0, nullptr);
            } else if (bindCount == 1) {
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipelineLayout(), 0, 1, setsToBind, 0, nullptr);
            }
        }
        
        // Render compacted draw commands - buffers already bound, just issue draw
        sceneRenderer.indirectRenderer.drawIndirectOnly(commandBuffer, this);

        // Timestamp 6: After main draw
        if (profilingEnabled) {
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 6);
        }

        // --- Handle water rendering (transparent, rendered in separate pass with scene sampling) ---
        VkPipeline waterGeomPipeline = sceneRenderer.waterRenderer.getWaterGeometryPipeline();
        if (hasWater) {
            auto waterStart = std::chrono::high_resolution_clock::now();
            
            // End offscreen render pass (scene is now in per-frame sceneColorImages[frameIdx]/sceneDepthImages[frameIdx])
            // The render pass finalLayout transitions them to SHADER_READ_ONLY_OPTIMAL
            vkCmdEndRenderPass(commandBuffer);
            
            // Blit per-frame scene to swapchain for display
            // Scene color and depth are in SHADER_READ_ONLY_OPTIMAL from render pass finalLayout
            
            // Find the current swapchain image
            uint32_t imageIndex = 0;
            const auto& framebuffers = getSwapchainFramebuffers();
            for (uint32_t i = 0; i < framebuffers.size(); ++i) {
                if (framebuffers[i] == renderPassInfo.framebuffer) {
                    imageIndex = i;
                    break;
                }
            }
            VkImage currentSwapchainImage = getSwapchainImages()[imageIndex];
            
            // Transition per-frame scene images from SHADER_READ_ONLY to TRANSFER_SRC
            VkImageMemoryBarrier sceneColorBarrier{};
            sceneColorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            sceneColorBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            sceneColorBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            sceneColorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            sceneColorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            sceneColorBarrier.image = sceneRenderer.waterRenderer.getSceneColorImage(frameIdx);
            sceneColorBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            sceneColorBarrier.subresourceRange.baseMipLevel = 0;
            sceneColorBarrier.subresourceRange.levelCount = 1;
            sceneColorBarrier.subresourceRange.baseArrayLayer = 0;
            sceneColorBarrier.subresourceRange.layerCount = 1;
            sceneColorBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            sceneColorBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            
            VkImageMemoryBarrier sceneDepthBarrier{};
            sceneDepthBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            sceneDepthBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            sceneDepthBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            sceneDepthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            sceneDepthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            sceneDepthBarrier.image = sceneRenderer.waterRenderer.getSceneDepthImage(frameIdx);
            sceneDepthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            sceneDepthBarrier.subresourceRange.baseMipLevel = 0;
            sceneDepthBarrier.subresourceRange.levelCount = 1;
            sceneDepthBarrier.subresourceRange.baseArrayLayer = 0;
            sceneDepthBarrier.subresourceRange.layerCount = 1;
            sceneDepthBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            sceneDepthBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            
            // Transition swapchain image for transfer dst
            VkImageMemoryBarrier swapBarrier{};
            swapBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            swapBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            swapBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            swapBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapBarrier.image = currentSwapchainImage;
            swapBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            swapBarrier.subresourceRange.baseMipLevel = 0;
            swapBarrier.subresourceRange.levelCount = 1;
            swapBarrier.subresourceRange.baseArrayLayer = 0;
            swapBarrier.subresourceRange.layerCount = 1;
            swapBarrier.srcAccessMask = 0;
            swapBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            
            // Transition main depth buffer for transfer dst
            VkImageMemoryBarrier mainDepthBarrier{};
            mainDepthBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            mainDepthBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            mainDepthBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            mainDepthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            mainDepthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            mainDepthBarrier.image = getDepthImage();
            mainDepthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            mainDepthBarrier.subresourceRange.baseMipLevel = 0;
            mainDepthBarrier.subresourceRange.levelCount = 1;
            mainDepthBarrier.subresourceRange.baseArrayLayer = 0;
            mainDepthBarrier.subresourceRange.layerCount = 1;
            mainDepthBarrier.srcAccessMask = 0;
            mainDepthBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            
            std::array<VkImageMemoryBarrier, 4> preBlitBarriers = {sceneColorBarrier, sceneDepthBarrier, swapBarrier, mainDepthBarrier};
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,  // Scene was written by color/depth output
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr,
                static_cast<uint32_t>(preBlitBarriers.size()), preBlitBarriers.data());
            
            // Blit scene to swapchain
            VkImageBlit blitRegion{};
            blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blitRegion.srcSubresource.mipLevel = 0;
            blitRegion.srcSubresource.baseArrayLayer = 0;
            blitRegion.srcSubresource.layerCount = 1;
            blitRegion.srcOffsets[0] = {0, 0, 0};
            blitRegion.srcOffsets[1] = {(int32_t)getWidth(), (int32_t)getHeight(), 1};
            blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blitRegion.dstSubresource.mipLevel = 0;
            blitRegion.dstSubresource.baseArrayLayer = 0;
            blitRegion.dstSubresource.layerCount = 1;
            blitRegion.dstOffsets[0] = {0, 0, 0};
            blitRegion.dstOffsets[1] = {(int32_t)getWidth(), (int32_t)getHeight(), 1};
            
            vkCmdBlitImage(commandBuffer,
                sceneRenderer.waterRenderer.getSceneColorImage(frameIdx), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                currentSwapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blitRegion, VK_FILTER_NEAREST);
            
            // Copy scene depth to main depth buffer for water depth testing
            VkImageBlit depthBlitRegion{};
            depthBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            depthBlitRegion.srcSubresource.mipLevel = 0;
            depthBlitRegion.srcSubresource.baseArrayLayer = 0;
            depthBlitRegion.srcSubresource.layerCount = 1;
            depthBlitRegion.srcOffsets[0] = {0, 0, 0};
            depthBlitRegion.srcOffsets[1] = {(int32_t)getWidth(), (int32_t)getHeight(), 1};
            depthBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            depthBlitRegion.dstSubresource.mipLevel = 0;
            depthBlitRegion.dstSubresource.baseArrayLayer = 0;
            depthBlitRegion.dstSubresource.layerCount = 1;
            depthBlitRegion.dstOffsets[0] = {0, 0, 0};
            depthBlitRegion.dstOffsets[1] = {(int32_t)getWidth(), (int32_t)getHeight(), 1};
            
            vkCmdBlitImage(commandBuffer,
                sceneRenderer.waterRenderer.getSceneDepthImage(frameIdx), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                getDepthImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &depthBlitRegion, VK_FILTER_NEAREST);
            
            // Transition swapchain and main depth for rendering
            swapBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            swapBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            swapBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            swapBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            
            mainDepthBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            mainDepthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            mainDepthBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mainDepthBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            
            // Transition scene images back to SHADER_READ_ONLY for water sampling
            VkImageMemoryBarrier sceneColorReadBarrier{};
            sceneColorReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            sceneColorReadBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            sceneColorReadBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            sceneColorReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            sceneColorReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            sceneColorReadBarrier.image = sceneRenderer.waterRenderer.getSceneColorImage(frameIdx);
            sceneColorReadBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            sceneColorReadBarrier.subresourceRange.baseMipLevel = 0;
            sceneColorReadBarrier.subresourceRange.levelCount = 1;
            sceneColorReadBarrier.subresourceRange.baseArrayLayer = 0;
            sceneColorReadBarrier.subresourceRange.layerCount = 1;
            sceneColorReadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            sceneColorReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            
            VkImageMemoryBarrier sceneDepthReadBarrier{};
            sceneDepthReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            sceneDepthReadBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            sceneDepthReadBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            sceneDepthReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            sceneDepthReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            sceneDepthReadBarrier.image = sceneRenderer.waterRenderer.getSceneDepthImage(frameIdx);
            sceneDepthReadBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            sceneDepthReadBarrier.subresourceRange.baseMipLevel = 0;
            sceneDepthReadBarrier.subresourceRange.levelCount = 1;
            sceneDepthReadBarrier.subresourceRange.baseArrayLayer = 0;
            sceneDepthReadBarrier.subresourceRange.layerCount = 1;
            sceneDepthReadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            sceneDepthReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            
            std::array<VkImageMemoryBarrier, 4> postBlitBarriers = {swapBarrier, mainDepthBarrier, sceneColorReadBarrier, sceneDepthReadBarrier};
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr,
                static_cast<uint32_t>(postBlitBarriers.size()), postBlitBarriers.data());
            
            // Begin swapchain render pass for water (load existing color, load depth)
            VkRenderPassBeginInfo waterRenderPassInfo{};
            waterRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            waterRenderPassInfo.renderPass = getContinuationRenderPass();
            waterRenderPassInfo.framebuffer = renderPassInfo.framebuffer;
            waterRenderPassInfo.renderArea.offset = {0, 0};
            waterRenderPassInfo.renderArea.extent = {getWidth(), getHeight()};
            waterRenderPassInfo.clearValueCount = 0;
            waterRenderPassInfo.pClearValues = nullptr;
            
            vkCmdBeginRenderPass(commandBuffer, &waterRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
            
            // Reset viewport and scissor
            VkViewport waterViewport{};
            waterViewport.x = 0.0f;
            waterViewport.y = 0.0f;
            waterViewport.width = (float)getWidth();
            waterViewport.height = (float)getHeight();
            waterViewport.minDepth = 0.0f;
            waterViewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &waterViewport);
            
            VkRect2D waterScissor{};
            waterScissor.offset = {0, 0};
            waterScissor.extent = {getWidth(), getHeight()};
            vkCmdSetScissor(commandBuffer, 0, 1, &waterScissor);
            
            // Optionally draw water geometry depending on pipeline and debug mode
            int debugMode = int(sceneRenderer.mainPassUBO.data.debugParams.x + 0.5f);
            if (waterGeomPipeline != VK_NULL_HANDLE && (debugMode == 0 || debugMode == 32)) {
                // Bind water pipeline
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, waterGeomPipeline);
                sceneRenderer.waterRenderer.getIndirectRenderer().bindBuffers(commandBuffer);
                
                // Bind all 3 descriptor sets at their correct indices:
                // Set 0: Materials SSBO
                // Set 1: UBO
                // Set 2: Scene color and depth textures (per-frame)
                {
                    VkDescriptorSet matDs = getMaterialDescriptorSet();
                    VkDescriptorSet sceneDs = sceneRenderer.waterRenderer.getWaterDepthDescriptorSet(frameIdx);
                    
                    // DEBUG: Print descriptor set handles
                    static int printCount = 0;
                    if (printCount++ % 120 == 0) {
                        std::cout << "[DEBUG] Water descriptor bindings: matDs=" << matDs 
                                  << " descriptorSet=" << descriptorSet 
                                  << " sceneDs=" << sceneDs << std::endl;
                    }
                    
                    // Bind sets 0 and 1 together (material + UBO)
                    if (matDs != VK_NULL_HANDLE && descriptorSet != VK_NULL_HANDLE) {
                        VkDescriptorSet sets01[2] = { matDs, descriptorSet };
                        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, 
                            sceneRenderer.waterRenderer.getWaterGeometryPipelineLayout(), 0, 2, sets01, 0, nullptr);
                    }
                    
                    // Bind set 2 (scene textures) at the correct index
                    if (sceneDs != VK_NULL_HANDLE) {
                        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, 
                            sceneRenderer.waterRenderer.getWaterGeometryPipelineLayout(), 2, 1, &sceneDs, 0, nullptr);
                    }
                }
                
                // Draw water geometry
                sceneRenderer.waterRenderer.getIndirectRenderer().drawIndirectOnly(commandBuffer, this);
                
                auto waterEnd = std::chrono::high_resolution_clock::now();
                profileWater = std::chrono::duration<float, std::milli>(waterEnd - waterStart).count();
            } else {
                // Not drawing water geometry in this configuration; zero out profile for consistency
                profileWater = 0.0f;
            }
        }

        // Update debug widget descriptors before ImGui renders
        if (renderPassDebugWidget && hasWater) {
            renderPassDebugWidget->updateDescriptors(frameIdx);
        }

        // render ImGui draw data inside the same command buffer (must be inside render pass)
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (draw_data) ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);

        // Timestamp 7: After ImGui
        if (profilingEnabled) {
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 7);
        }

        vkCmdEndRenderPass(commandBuffer);

        // Transition depth image to READ_ONLY_OPTIMAL for sampling in shaders
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = getDepthImage();
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
        
        auto cpuRecordEnd = std::chrono::high_resolution_clock::now();
        profileCpuRecord = std::chrono::duration<float, std::milli>(cpuRecordEnd - cpuRecordStart).count();
            
    };

    void clean() override {
        // Unsubscribe handlers from event manager to avoid callbacks during teardown
        eventManager.unsubscribe(&camera);
        eventManager.unsubscribe(this);
        // Clear any queued events
        eventManager.processQueued();

        // Cleanup profiling query pool
        if (queryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(getDevice(), queryPool, nullptr);
            queryPool = VK_NULL_HANDLE;
        }

        if (sceneRenderer.graphicsPipeline != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), sceneRenderer.graphicsPipeline, nullptr);
        if (sceneRenderer.graphicsPipelineWire != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), sceneRenderer.graphicsPipelineWire, nullptr);
        if (sceneRenderer.depthPrePassPipeline != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), sceneRenderer.depthPrePassPipeline, nullptr);
        if (sceneRenderer.waterPipeline != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), sceneRenderer.waterPipeline, nullptr);
        // Tessellation pipelines removed earlier; nothing to destroy here.
        sceneRenderer.cleanup();
        // texture cleanup via managers
        // legacy texture managers removed; arrays handle GPU textures now
        textureArrayManager.destroy(this);
        vegetationTextureArrayManager.destroy(this);

        
        // editable textures cleanup
        if (textureMixer) textureMixer->cleanup();
        
        // billboard creator cleanup
        if (billboardCreator) billboardCreator->cleanup();
        
        // material manager cleanup
        materialManager.destroy(this);
    }

};

int main() {
    MyApp app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

