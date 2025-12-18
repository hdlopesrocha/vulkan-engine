#include "vulkan/VulkanApp.hpp"
#include "utils/FileReader.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include "math/Camera.hpp"
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
#include "widgets/WidgetManager.hpp"
#include "widgets/CameraWidget.hpp"
#include "widgets/DebugWidget.hpp"
#include "widgets/ShadowMapWidget.hpp"
#include "widgets/TextureViewerWidget.hpp"
#include "widgets/SettingsWidget.hpp"
#include "widgets/LightWidget.hpp"
#include "widgets/SkyWidget.hpp"
#include "vulkan/SkySphere.hpp"
#include "vulkan/DescriptorSetBuilder.hpp"
#include "vulkan/SkyRenderer.hpp"
#include "widgets/VegetationAtlasEditor.hpp"
#include "widgets/BillboardCreator.hpp"
#include "widgets/VulkanObjectsWidget.hpp"
#include <string>
#include <memory>
#include <iostream>
#include <cmath>
#include "utils/LocalScene.hpp"
#include "utils/MainSceneLoader.hpp"

#include "Uniforms.hpp"
#include "vulkan/MaterialManager.hpp"

#include "vulkan/VertexBufferObjectBuilder.hpp"
#include "vulkan/Model3DVersion.hpp"
#include "vulkan/Model3D.hpp"

class MyApp : public VulkanApp, public IEventHandler {
    LocalScene * mainScene;
    std::unordered_map<OctreeNode*, Model3DVersion> nodeModelVersions;
    std::vector<Model3D*> visibleModels;

    public:
        MyApp() : shadowMapper(this, 8192) {}


            void postSubmit() override {
                
            }

        // postSubmit() override removed to disable slow per-frame shadow readback
        
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    
    // GPU buffers for the built meshes (parallel to `meshes`)
    VertexBufferObject skyVBO;
    VkPipeline graphicsPipelineWire = VK_NULL_HANDLE;
    std::unique_ptr<SkyRenderer> skyRenderer;
    std::unique_ptr<SkySphere> skySphere;
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

    // Shadow mapping
    ShadowMapper shadowMapper;
    
    // Light direction (controlled by LightWidget)
    glm::vec3 lightDirection = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));
    // Camera
    Camera camera = Camera(glm::vec3(3456, 915, -750), Math::eulerToQuat(0, 0, 0));
    // Event manager for app-wide pub/sub
    EventManager eventManager;
    KeyboardPublisher keyboard;
    GamepadPublisher gamepad;
    Buffer mainUniform;
    Buffer shadowUniform;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet shadowDescriptorSet = VK_NULL_HANDLE;
    MaterialManager materialManager;
    size_t materialCount = 0;
    
    // store projection and view for per-cube MVP computation in draw()
    glm::mat4 projMat = glm::mat4(1.0f);
    glm::mat4 viewMat = glm::mat4(1.0f);
    // static parts of the UBO that don't vary per-cube (we'll set model/mvp per draw)
    UniformObject uboStatic{};
    // textureImage is now owned by TextureManager
    //VertexBufferObject vertexBufferObject; // now owned by CubeMesh

    void setup() override {
        // Subscribe camera and app to event manager
        eventManager.subscribe(&camera);
        eventManager.subscribe(this);
        visibleModels.reserve(1000);

        // Use ImGui dark theme for a darker UI appearance
        ImGui::StyleColorsDark();

        // Initialize shadow mapper (moved after pipeline creation below)
        
        // Graphics Pipeline
        {
            ShaderStage vertexShader = ShaderStage(
                createShaderModule(FileReader::readFile("shaders/main.vert.spv")), 
                VK_SHADER_STAGE_VERTEX_BIT
            );

            ShaderStage fragmentShader = ShaderStage(
                createShaderModule(FileReader::readFile("shaders/main.frag.spv")), 
                VK_SHADER_STAGE_FRAGMENT_BIT
            );
            // Load tessellation shaders so pipeline always contains TCS/TES stages.
            ShaderStage tescShader = ShaderStage(
                createShaderModule(FileReader::readFile("shaders/main.tesc.spv")),
                VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
            );
            ShaderStage teseShader = ShaderStage(
                createShaderModule(FileReader::readFile("shaders/main.tese.spv")),
                VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
            );

            graphicsPipeline = createGraphicsPipeline(
                {
                    vertexShader.info,
                    tescShader.info,
                    teseShader.info,
                    fragmentShader.info
                },
                VkVertexInputBindingDescription { 
                    0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX 
                },
                {
                    VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
                    VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                    VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
                    VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                    VkVertexInputAttributeDescription { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent) },
                    VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
                }
            );

            // Create wireframe variant (if device supports it), also includes tessellation stages
            graphicsPipelineWire = createGraphicsPipeline(
                {
                    vertexShader.info,
                    tescShader.info,
                    teseShader.info,
                    fragmentShader.info
                },
                VkVertexInputBindingDescription { 
                    0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX 
                },
                {
                    VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
                    VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                    VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
                    VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                    VkVertexInputAttributeDescription { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent) },
                    VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
                },
                VK_POLYGON_MODE_LINE
            );

            // Tessellation pipeline removed; tessellation is controlled inside the shader (main.tesc)

            // Destroy tessellation shader modules and vertex/fragment modules after pipeline creation
            vkDestroyShaderModule(getDevice(), teseShader.info.module, nullptr);
            vkDestroyShaderModule(getDevice(), tescShader.info.module, nullptr);
            vkDestroyShaderModule(getDevice(), fragmentShader.info.module, nullptr);
            vkDestroyShaderModule(getDevice(), vertexShader.info.module, nullptr);

            // Create sky pipeline via SkyRenderer
            skyRenderer = std::make_unique<SkyRenderer>(this);
            skyRenderer->init();
        }

        // Now that pipelines (and the app pipeline layout) have been created, initialize shadow mapper
        shadowMapper.init();

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
        
        // create three global uniform buffers: main, shadow and sky
        mainUniform = createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        shadowUniform = createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        // create descriptor sets - one per texture triple
        size_t tripleCount = materials.size();
        if (tripleCount == 0) tripleCount = 1; // ensure at least one
        // Create/upload GPU-side material buffer (packed vec4-friendly struct)
        updateMaterials();
        // Allocate descriptor pool sized for descriptor sets. We need room for per-texture sets and
        // per-model-instance sets (we'll create one descriptor set per Model3D instance later).
        // Estimate: per-texture sets = tripleCount * 4, per-instance sets = (1 + 2*tripleCount) * 2 (main+shadow), add a small slack.
        uint32_t estimatedPerTextureSets = static_cast<uint32_t>(tripleCount * 4);
        uint32_t estimatedModelObjects = static_cast<uint32_t>(1 + 2 * tripleCount);
        uint32_t estimatedPerInstanceSets = estimatedModelObjects * 2;
        // Reserve one extra set for the global material descriptor
        uint32_t totalSets = estimatedPerTextureSets + estimatedPerInstanceSets + 4 + 1;
        // each descriptor set writes up to 4 combined image samplers (albedo/normal/height/shadow)
        createDescriptorPool(totalSets, totalSets * 4);

        // Create a global material descriptor set (binding 5) so we can bind the whole
        // Materials SSBO once (set 0) and avoid per-instance/per-texture material bindings.
        VkDescriptorSet globalMatDS = VK_NULL_HANDLE;
        if (materialManager.count() > 0) {
            globalMatDS = createDescriptorSet(getMaterialDescriptorSetLayout());
            VkDescriptorBufferInfo materialBufInfo{ materialManager.getBuffer().buffer, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet matWrite{}; matWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; matWrite.dstSet = globalMatDS; matWrite.dstBinding = 5; matWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; matWrite.descriptorCount = 1; matWrite.pBufferInfo = &materialBufInfo;
            updateDescriptorSet(globalMatDS, { matWrite });
            setMaterialDescriptorSet(globalMatDS);
            registerDescriptorSet(globalMatDS);
        }

        descriptorSet = VK_NULL_HANDLE;
        shadowDescriptorSet = VK_NULL_HANDLE;

        // Use DescriptorSetBuilder to reduce repeated code
        DescriptorSetBuilder dsBuilder(this, &shadowMapper);
        for (size_t i = 0; i < tripleCount; ++i) {
            // Construct a transient Triple that points into the global texture arrays so
            // DescriptorSetBuilder can create descriptor sets. Bindings will be overwritten
            // with array descriptors below when arrays are present.
            Triple tr;
            tr.albedo.view = textureArrayManager.albedoArray.view;
            tr.albedoSampler = textureArrayManager.albedoSampler;
            tr.normal.view = textureArrayManager.normalArray.view;
            tr.normalSampler = textureArrayManager.normalSampler;
            tr.height.view = textureArrayManager.bumpArray.view;
            tr.heightSampler = textureArrayManager.bumpSampler;
            // main descriptor set
            VkDeviceSize matElemSize = sizeof(glm::vec4) * 4; // size of MaterialGPU
            // create main descriptor set (create once, reuse)
            if (descriptorSet == VK_NULL_HANDLE) {
                VkDescriptorSet ds = dsBuilder.createMainDescriptorSet(tr, mainUniform, false, nullptr, 0);
                descriptorSet = ds;
                registerDescriptorSet(descriptorSet);
            }

            // shadow descriptor set (create only once, reuse globally)
            if (shadowDescriptorSet == VK_NULL_HANDLE) {
                VkDescriptorSet sds = dsBuilder.createShadowDescriptorSet(tr, shadowUniform, false, nullptr, 0);
                shadowDescriptorSet = sds;
                registerDescriptorSet(shadowDescriptorSet);
            }
        }

        // If texture arrays are allocated, overwrite bindings 1..3 in descriptor sets
        // to point to the global texture arrays (sampler2DArray) so shaders can index by layer.
        if (textureArrayManager.layerAmount > 0) {
            VkDescriptorImageInfo albedoArrayInfo{ textureArrayManager.albedoSampler, textureArrayManager.albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo normalArrayInfo{ textureArrayManager.normalSampler, textureArrayManager.normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo bumpArrayInfo{ textureArrayManager.bumpSampler, textureArrayManager.bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

            if (descriptorSet != VK_NULL_HANDLE) {
                VkWriteDescriptorSet w1{}; w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w1.dstSet = descriptorSet; w1.dstBinding = 1; w1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w1.descriptorCount = 1; w1.pImageInfo = &albedoArrayInfo;
                VkWriteDescriptorSet w2{}; w2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w2.dstSet = descriptorSet; w2.dstBinding = 2; w2.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w2.descriptorCount = 1; w2.pImageInfo = &normalArrayInfo;
                VkWriteDescriptorSet w3{}; w3.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w3.dstSet = descriptorSet; w3.dstBinding = 3; w3.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w3.descriptorCount = 1; w3.pImageInfo = &bumpArrayInfo;
                updateDescriptorSet(descriptorSet, { w1, w2, w3 });
            }
            if (shadowDescriptorSet != VK_NULL_HANDLE) {
                VkWriteDescriptorSet w1{}; w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w1.dstSet = shadowDescriptorSet; w1.dstBinding = 1; w1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w1.descriptorCount = 1; w1.pImageInfo = &albedoArrayInfo;
                VkWriteDescriptorSet w2{}; w2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w2.dstSet = shadowDescriptorSet; w2.dstBinding = 2; w2.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w2.descriptorCount = 1; w2.pImageInfo = &normalArrayInfo;
                VkWriteDescriptorSet w3{}; w3.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w3.dstSet = shadowDescriptorSet; w3.dstBinding = 3; w3.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w3.descriptorCount = 1; w3.pImageInfo = &bumpArrayInfo;
                updateDescriptorSet(shadowDescriptorSet, { w1, w2, w3 });
            }
        }

        // Build sphere mesh for this material
        auto sphere = SphereModel(0.5f, 32, 16, 0);
        skyVBO = VertexBufferObjectBuilder::create(this, sphere);
        
        // Per-instance descriptor sets removed — use per-material/global descriptor sets instead.
        // Material SSBO is bound into a single global descriptor set; updateMaterials() will
        // refresh that set if it exists.
        updateMaterials();
        
        // Initialize light space matrix BEFORE creating command buffers
        // UI `lightDirection` is the direction FROM the camera/world TO the light (TO the light).
        // For constructing the light view (positioning the light), we need a vector FROM the light TOWARD the scene center,
        // so negate the UI direction when computing light position for shadow mapping.
        glm::vec3 lightDirToLight = -glm::normalize(lightDirection);
        // Center shadow ortho on camera XZ so it follows camera movement
        glm::vec3 camPosInit = camera.getPosition();
        glm::vec3 sceneCenter = glm::vec3(camPosInit.x, -0.75f, camPosInit.z);
        glm::vec3 lightPos = sceneCenter + lightDirToLight * 20.0f;
        glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
        if (glm::abs(glm::dot(lightDirToLight, worldUp)) > 0.9f) {
            worldUp = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        glm::mat4 lightView = glm::lookAt(lightPos, sceneCenter, worldUp);
        // Increase ortho size significantly to ensure all cubes and plane are captured
        float orthoSize = 25.0f; // Much larger to be safe
        glm::mat4 lightProjection = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 1.0f, 50.0f);
        uboStatic.lightSpaceMatrix = lightProjection * lightView;
        
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
            // Update texture arrays so shaders sample the new images
            if (editableLayerIndex != SIZE_MAX) {
                textureArrayManager.updateLayerFromEditableMap(static_cast<uint32_t>(editableLayerIndex), textureMixer->getAlbedo(), 0);
                textureArrayManager.updateLayerFromEditableMap(static_cast<uint32_t>(editableLayerIndex), textureMixer->getNormal(), 1);
                textureArrayManager.updateLayerFromEditableMap(static_cast<uint32_t>(editableLayerIndex), textureMixer->getBump(), 2);
            }
            updateMaterials();
        });
        
        // Create other widgets
        auto cameraWidget = std::make_shared<CameraWidget>(&camera);
        widgetManager.addWidget(cameraWidget);
        
        auto debugWidget = std::make_shared<DebugWidget>(&materials, &camera, &currentTextureIndex);
        widgetManager.addWidget(debugWidget);
        
        auto shadowWidget = std::make_shared<ShadowMapWidget>(&shadowMapper);
        widgetManager.addWidget(shadowWidget);
        
        // Create settings widget
        settingsWidget = std::make_shared<SettingsWidget>();
        // Hook debug UI button to trigger a shadow depth readback
        settingsWidget->setDumpShadowDepthCallback([this]() {
            shadowMapper.readbackShadowDepth();
            std::cerr << "[Debug] Wrote bin/shadow_depth.pgm (manual trigger).\n";
        });
        widgetManager.addWidget(settingsWidget);
        
        // Create light control widget
        auto lightWidget = std::make_shared<LightWidget>(&lightDirection);
        widgetManager.addWidget(lightWidget);
        // Create sky widget (controls colors and parameters)
        skyWidget = std::make_shared<SkyWidget>();
        widgetManager.addWidget(skyWidget);
        // Vulkan objects widget
        auto vulkanObjectsWidget = std::make_shared<VulkanObjectsWidget>(this);
        widgetManager.addWidget(vulkanObjectsWidget);
        if (skyWidget) {
            // Sky UBO is managed by SkySphere; SkyWidget values will be read and uploaded there.
        }

        // Initialize sky manager which creates and binds the sky UBO into descriptor sets
        skySphere = std::make_unique<SkySphere>(this);
        skySphere->init(skyWidget.get(), descriptorSet, shadowDescriptorSet);
        
        // Create vegetation atlas editor widget
        auto vegAtlasEditor = std::make_shared<VegetationAtlasEditor>(&vegetationTextureArrayManager, &vegetationAtlasManager);
        widgetManager.addWidget(vegAtlasEditor);
        
        // Create billboard creator widget
        billboardCreator = std::make_shared<BillboardCreator>(&billboardManager, &vegetationAtlasManager, &vegetationTextureArrayManager);
        billboardCreator->setVulkanApp(this);
        widgetManager.addWidget(billboardCreator);
        
        createCommandBuffers();
        mainScene = new LocalScene();

        MainSceneLoader mainSceneLoader = MainSceneLoader();
        mainScene->loadScene(mainSceneLoader);

        
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
                    ImGui::Text("FPS: %.1f", imguiFps);
                    ImGui::Text("Visible: %zu", visibleModels.size());
                }
                ImGui::End();
            }
        }

        if (imguiShowDemo) ImGui::ShowDemoWindow(&imguiShowDemo);

        // Render all widgets
        widgetManager.renderAll();
    }

    void update(float deltaTime) override {
        // Process queued events (dispatch to handlers)
        eventManager.processQueued();
        // compute viewProjection = proj * view (model set per-object)
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 mvp = glm::mat4(1.0f);

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

        // build view matrix from camera state (no cube rotation)
        glm::mat4 view = camera.getViewMatrix();

        // keep model identity (stop cube rotation)
        model = glm::mat4(1.0f);
        mvp = proj * view * model;

        // store projection and view for per-cube viewProjection computation in draw()
        projMat = proj;
        viewMat = view;

        // prepare static parts of the UBO (viewPos, light, material flags) - model and viewProjection will be set per-cube in draw()
        uboStatic.viewPos = glm::vec4(camera.getPosition(), 1.0f);
        // The UI `lightDirection` represents a vector TO the light. Shaders expect a light vector that points FROM the
        // light TOWARD the surface when performing lighting/shadow calculations. Send the negated direction to the GPU
        // so both lighting and shadow projection use the same convention.
        glm::vec3 lightDir = glm::normalize(lightDirection);
        uboStatic.lightDir = glm::vec4(lightDir, 0.0f);
        uboStatic.lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
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
        // Note: material flags and specularParams are set per-instance from material properties
        // Sky UBO updates are handled by SkySphere (reads SkyWidget directly)
        
        // Compute light space matrix for shadow mapping
        // Center shadow ortho on camera XZ so the shadow projection follows camera movement
        glm::vec3 camPos = camera.getPosition();
        glm::vec3 sceneCenter = glm::vec3(camPos.x, -0.75f, camPos.z);

        // Diagonal light direction matching setup(): use negated UI direction so the light position and view
        // are consistent with the negated `uboStatic.lightDir` sent to shaders.
        glm::vec3 shadowLightDir = glm::normalize(lightDirection);
        glm::vec3 lightPos = sceneCenter - shadowLightDir * 20.0f;
        
        glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
        if (glm::abs(glm::dot(shadowLightDir, worldUp)) > 0.9f) {
            worldUp = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        
        glm::mat4 lightView = glm::lookAt(lightPos, sceneCenter, worldUp);
        
        float orthoSize = 15.0f;
        // Use 0.1 to 50.0 for near/far to ensure good depth range
        glm::mat4 lightProjection = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 0.1f, 50.0f);
        
        uboStatic.lightSpaceMatrix = lightProjection * lightView;
    };

    void draw(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo) override {
        visibleModels.clear();
        // Request visible octree nodes from the main scene for the current camera view            
        mainScene->requestVisibleNodes(Layer::LAYER_OPAQUE, camera.getViewProjectionMatrix(), [this](const OctreeNodeData& data){ 
            // Capture node/version locally to ensure lifetime for the async request callback
            OctreeNode* node = data.node;
            // skip null nodes
            if (!node) return;
            unsigned int version = node->version;

            // Find existing entry and remove it if version changed
            auto it = nodeModelVersions.find(node);
            if (it != nodeModelVersions.end()) {
                if (it->second.version != version) {
                    if (it->second.model) delete it->second.model;
                    nodeModelVersions.erase(it);
                    it = nodeModelVersions.end();
                }
            }

            if (it == nodeModelVersions.end()) {
                // Insert placeholder and request async model load
                auto em = nodeModelVersions.emplace(node, Model3DVersion{});
                it = em.first;
                // requestModel3D expects a non-const reference; cast away const here
                mainScene->requestModel3D(Layer::LAYER_OPAQUE, const_cast<OctreeNodeData&>(data), [this, node, version](const Geometry& mesh) {
                    Model3D * model = new Model3D(
                        VertexBufferObjectBuilder::create(this, mesh), 
                        glm::mat4(1.0f)
                    );
                    nodeModelVersions[node] = { model, version };
                    std::cout << "[Model3D] Loaded model for node " << node << " (version " << version << ")\n";
                });
            }

            // Only add fully-loaded models to visibleModels
            Model3D* loadedModel = nodeModelVersions[node].model;
            if (loadedModel) visibleModels.push_back(loadedModel);
            
        });
        
        
        
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
    

        // Calculate grid layout for cubes
        size_t n = materials.size();
        float spacing = 2.5f;
        if (n == 0) n = 1;
        int cols = static_cast<int>(std::ceil(std::sqrt((float)n)));
        int rows = static_cast<int>(std::ceil((float)n / cols));
        float halfW = (cols - 1) * 0.5f * spacing;
        float halfH = (rows - 1) * 0.5f * spacing;
        

        // First pass: Render shadow map (skip if shadows globally disabled)
        if (!settingsWidget || settingsWidget->getShadowsEnabled()) {
            shadowMapper.beginShadowPass(commandBuffer, uboStatic.lightSpaceMatrix);
            
            // Render all instances to shadow map
            for (const auto& instance : visibleModels) {
                // Update uniform buffer for shadow pass: set viewProjection = lightSpaceMatrix
                UniformObject shadowUbo = uboStatic;
                shadowUbo.viewProjection = uboStatic.lightSpaceMatrix;
                shadowUbo.materialFlags = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
                // Per-material mapping/mappingParams are read from the Materials SSBO in shaders.
                shadowUbo.passParams = glm::vec4(1.0f, settingsWidget ? (settingsWidget->getShadowTessellationEnabled() ? 1.0f : 0.0f) : 0.0f, 0.0f, 0.0f);

                // Use a global (per-material) shadow uniform buffer and descriptor set instead
                // of the per-instance buffer. Determine material index from the mesh's texIndex.
                // single global shadow descriptor set is used; no per-instance selection needed

                updateUniformBuffer(shadowUniform, &shadowUbo, sizeof(UniformObject));
                // Push model matrix for this draw into the pipeline via push constants
                vkCmdPushConstants(commandBuffer, getPipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, 0, sizeof(glm::mat4), &instance->model);

                shadowMapper.renderObject(commandBuffer, instance->vbo, shadowDescriptorSet);
            }

            shadowMapper.endShadowPass(commandBuffer);
        }
        
        // Second pass: Render scene with shadows
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        
        // set dynamic viewport/scissor to match current swapchain extent
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)getWidth();
        viewport.height = (float)getHeight();
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = { (uint32_t)getWidth(), (uint32_t)getHeight() };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // --- Render sky sphere first: large sphere centered at camera ---
        if (skyRenderer && descriptorSet != VK_NULL_HANDLE) {
            if (skySphere) skySphere->update();
            skyRenderer->render(commandBuffer, skyVBO, descriptorSet, mainUniform, uboStatic, projMat, viewMat);
        }

        // bind pipeline
        // We'll bind the appropriate pipeline per-instance (regular or tessellated)

        // Render all instances with main pass
        for (const auto& instance : visibleModels) {
            const auto& vbo = instance->vbo;
            
            // Bind vertex and index buffers
            VkBuffer vertexBuffers[] = { vbo.vertexBuffer.buffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, vbo.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            
            // Update uniform buffer for this instance: set viewProjection and model separately
            UniformObject ubo = uboStatic;
            ubo.viewProjection = projMat * viewMat;
            
            // Per-material tess level is provided by the Materials SSBO; no per-instance override here.
            // (temporary debug logging removed)
            // Global normal mapping toggle (separate from tessellation/mappingMode)
            if (settingsWidget) {
                ubo.materialFlags.w = settingsWidget->getNormalMappingEnabled() ? 1.0f : 0.0f;
            } else {
                ubo.materialFlags.w = 1.0f;
            }
            
            // Apply shadow settings: 
            ubo.shadowEffects = glm::vec4(
                0.0f, 
                0.0f, // shadow displacement disabled
                0.0f, 
                settingsWidget->getShadowsEnabled() ? 1.0f : 0.0f  // global shadows enabled
            );
            // Debug visualization mode (set by SettingsWidget)
            if (settingsWidget) ubo.debugParams = glm::vec4((float)settingsWidget->getDebugMode(), 0.0f, 0.0f, 0.0f);
            else ubo.debugParams = glm::vec4(0.0f);
            // Triplanar settings from UI
            if (settingsWidget) ubo.triplanarSettings = glm::vec4(settingsWidget->getTriplanarThreshold(), settingsWidget->getTriplanarExponent(), 0.0f, 0.0f);
            else ubo.triplanarSettings = glm::vec4(0.12f, 3.0f, 0.0f, 0.0f);
            // isShadowPass = 0.0; second component stores global tessellation enabled flag (1.0 = enabled)
            ubo.passParams = glm::vec4(0.0f, settingsWidget ? (settingsWidget->getTessellationEnabled() ? 1.0f : 0.0f) : 0.0f, 0.0f, 0.0f);
 
            updateUniformBuffer(mainUniform, &ubo, sizeof(UniformObject));
            
            // Bind descriptor set and draw
            // Use a single pipeline (tessellation is enabled/disabled in the shader using mappingMode)
            bool wire = settingsWidget ? settingsWidget->getWireframeEnabled() : false;
            if (wire && graphicsPipelineWire != VK_NULL_HANDLE) vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineWire);
            else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            // Bind global material descriptor set (set 0) and per-material descriptor set (set 1)
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
            // Push per-draw model matrix via push constants (visible to vertex + tessellation stages)
            vkCmdPushConstants(commandBuffer, getPipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, 0, sizeof(glm::mat4), &instance->model);
            vkCmdDrawIndexed(commandBuffer, vbo.indexCount, 1, 0, 0, 0);
        }

        // render ImGui draw data inside the same command buffer (must be inside render pass)
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (draw_data) ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
            
    };

    void clean() override {
        // Unsubscribe handlers from event manager to avoid callbacks during teardown
        eventManager.unsubscribe(&camera);
        eventManager.unsubscribe(this);
        // Clear any queued events
        eventManager.processQueued();

        if (graphicsPipeline != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), graphicsPipeline, nullptr);
        if (graphicsPipelineWire != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), graphicsPipelineWire, nullptr);
        // Tessellation pipelines removed earlier; nothing to destroy here.
        if (skyRenderer) skyRenderer->cleanup();
        // texture cleanup via managers
        // legacy texture managers removed; arrays handle GPU textures now
        textureArrayManager.destroy(this);
        vegetationTextureArrayManager.destroy(this);
 
        // destroy any dynamically created VBOs from async-loaded models
       // for (auto &pv : dynamicMeshVBOs) {
       //     if (pv) pv->destroy(getDevice());
       // }
       // dynamicMeshVBOs.clear();

        // delete any heap-allocated Model3D instances created for async nodes
        for (auto &entry : nodeModelVersions) {
            if (entry.second.model) delete entry.second.model;
        }
        nodeModelVersions.clear();

        // global uniform buffers cleanup (main, shadow, sky)
        if (mainUniform.buffer != VK_NULL_HANDLE) vkDestroyBuffer(getDevice(), mainUniform.buffer, nullptr);
        if (mainUniform.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), mainUniform.memory, nullptr);
        if (shadowUniform.buffer != VK_NULL_HANDLE) vkDestroyBuffer(getDevice(), shadowUniform.buffer, nullptr);
        if (shadowUniform.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), shadowUniform.memory, nullptr);
        // Sky UBO is managed by SkySphere and cleaned up there.
        
        // editable textures cleanup
        if (textureMixer) textureMixer->cleanup();
        
        // billboard creator cleanup
        if (billboardCreator) billboardCreator->cleanup();
        
        // shadow map cleanup
        shadowMapper.cleanup();
        // material manager cleanup
        materialManager.destroy(this);
        // sky resources cleanup via SkySphere
        if (skySphere) skySphere->cleanup();
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

