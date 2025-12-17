#include "vulkan/VulkanApp.hpp"
#include "vulkan/FileReader.hpp"

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
#include "vulkan/TextureManager.hpp"
#include "vulkan/TextureArrayManager.hpp"
#include "vulkan/AtlasManager.hpp"
#include "utils/BillboardManager.hpp"
#include "math/CubeModel.hpp"
#include "math/PlaneModel.hpp"
#include "math/SphereModel.hpp"
#include "vulkan/EditableTextureSet.hpp"
#include "vulkan/ShadowMapper.hpp"
#include "vulkan/ModelManager.hpp"
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
#include "utils/Model3DVersion.hpp"
#include "vulkan/Model3D.hpp"

class MyApp : public VulkanApp, public IEventHandler {
    LocalScene * mainScene;
    std::unordered_map<OctreeNode*, Model3DVersion> nodeModelVersions;
    std::vector<Mesh3D*> visibleModels;

    public:
        MyApp() : shadowMapper(this, 8192) {}


            void postSubmit() override {
                
            }

        // postSubmit() override removed to disable slow per-frame shadow readback
        
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    
    // GPU buffers for the built meshes (parallel to `meshes`)
    std::vector<VertexBufferObject> meshVBOs;
    VkPipeline graphicsPipelineWire = VK_NULL_HANDLE;
    std::unique_ptr<SkyRenderer> skyRenderer;
    std::unique_ptr<SkySphere> skySphere;
    // texture manager handles albedo/normal/height triples
    TextureManager textureManager;
    
    // Vegetation texture manager for billboard vegetation (albedo/normal/opacity)
    TextureManager vegetationTextureManager;
    
    // Atlas manager for vegetation tile definitions
    AtlasManager vegetationAtlasManager;
    
    // Billboard manager for creating billboards from atlas tiles
    BillboardManager billboardManager;
    
    // Widget manager (handles all UI windows)
    WidgetManager widgetManager;
    
    // Widgets (shared pointers for widget manager)
    std::shared_ptr<TextureViewer> textureViewer;
    std::shared_ptr<EditableTextureSet> editableTextures;
    std::shared_ptr<SettingsWidget> settingsWidget;
    std::shared_ptr<BillboardCreator> billboardCreator;
    std::shared_ptr<SkyWidget> skyWidget;
    
    // Model manager to handle all renderable objects
    ModelManager modelManager;
    
    // Store meshes (owned by app but not direct attributes)
    std::vector<std::unique_ptr<Mesh3D>> meshes;
    // Store simple model wrappers (mesh pointer + VBO reference + model matrix)
    std::vector<std::unique_ptr<Model3D>> modelObjects;
    
    // UI: currently selected texture triple for preview
    size_t currentTextureIndex = 0;
    size_t editableTextureIndex = 0;  // Index of the editable texture triple in TextureManager
    // Cached material for the ground plane (captured at setup and updated on texture regenerate)
    MaterialProperties planeMaterial;
    
    // Shadow mapping
    ShadowMapper shadowMapper;
    
    // Light direction (controlled by LightWidget)
    glm::vec3 lightDirection = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));
    
    // Camera
    // start the camera further back so multiple cubes are visible
    Camera camera = Camera(glm::vec3(0.0f, 0.0f, 8.0f));
    // Event manager for app-wide pub/sub
    EventManager eventManager;
    KeyboardPublisher keyboard;
    GamepadPublisher gamepad;
    // (managed by TextureManager)
        std::vector<VkDescriptorSet> descriptorSets;
        std::vector<Buffer> uniforms; // One uniform buffer per cube
        // Per-model-instance descriptor sets and uniform buffers (one UBO + set per Model3D)
        std::vector<VkDescriptorSet> instanceDescriptorSets;
        std::vector<Buffer> instanceUniforms;
        std::vector<VkDescriptorSet> instanceShadowDescriptorSets;
        std::vector<Buffer> instanceShadowUniforms;
        // Additional per-texture uniform buffers and descriptor sets for sphere instances
        std::vector<VkDescriptorSet> sphereDescriptorSets;
        std::vector<Buffer> sphereUniforms;
        // Shadow pass buffers and descriptor sets
        std::vector<Buffer> shadowUniforms;
        std::vector<VkDescriptorSet> shadowDescriptorSets;
        std::vector<Buffer> shadowSphereUniforms;
        std::vector<VkDescriptorSet> shadowSphereDescriptorSets;
        // Material manager for GPU-side packed materials
        MaterialManager materialManager;
        size_t materialCount = 0;
        // Small UBO for sky parameters is managed by SkySphere
        
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

            // initialize texture manager and explicitly load known albedo/normal/height triples by filename
            textureManager.init(this);
            std::vector<size_t> loadedIndices;

            // Explicit per-name loads (one-by-one) with realistic material properties
            const std::vector<std::pair<std::array<const char*,3>, MaterialProperties>> specs = {
                {{"textures/bricks_color.jpg", "textures/bricks_normal.jpg", "textures/bricks_bump.jpg"}, MaterialProperties{true, false, 0.08f, 4.0f, 0.12f, 0.5f, 32.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
                {{"textures/dirt_color.jpg", "textures/dirt_normal.jpg", "textures/dirt_bump.jpg"}, MaterialProperties{true, false, 0.05f, 1.0f, 0.15f, 0.5f, 32.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
                {{"textures/forest_color.jpg", "textures/forest_normal.jpg", "textures/forest_bump.jpg"}, MaterialProperties{true, false, 0.06f, 1.0f, 0.18f, 0.5f, 32.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
                {{"textures/grass_color.jpg", "textures/grass_normal.jpg", "textures/grass_bump.jpg"}, MaterialProperties{true, false, 0.04f, 1.0f, 0.5f, 0.5f, 32.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
                {{"textures/lava_color.jpg", "textures/lava_normal.jpg", "textures/lava_bump.jpg"}, MaterialProperties{true, false, 0.03f, 1.0f, 0.4f, 0.5f, 32.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
                {{"textures/metal_color.jpg", "textures/metal_normal.jpg", "textures/metal_bump.jpg"}, MaterialProperties{true, false, 0.02f, 1.0f, 0.1f, 0.8f, 64.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
                {{"textures/pixel_color.jpg", "textures/pixel_normal.jpg", "textures/pixel_bump.jpg"}, MaterialProperties{true, false, 0.01f, 1.0f, 0.15f, 0.3f, 16.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
                {{"textures/rock_color.jpg", "textures/rock_normal.jpg", "textures/rock_bump.jpg"}, MaterialProperties{true, false, 0.1f, 1.0f, 0.1f, 0.4f, 32.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
                {{"textures/sand_color.jpg", "textures/sand_normal.jpg", "textures/sand_bump.jpg"}, MaterialProperties{true, false, 0.03f, 1.0f, 0.5f, 0.2f, 16.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
                {{"textures/snow_color.jpg", "textures/snow_normal.jpg", "textures/snow_bump.jpg"}, MaterialProperties{true, false, 0.04f, 1.0f, 0.1f, 0.1f, 8.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
                {{"textures/soft_sand_color.jpg", "textures/soft_sand_normal.jpg", "textures/soft_sand_bump.jpg"}, MaterialProperties{true, false, 0.025f, 1.0f, 0.22f, 0.3f, 16.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}}
            };

            for (const auto &entry : specs) {
                const auto &files = entry.first;
                const auto &matSpec = entry.second;
                size_t idx = textureManager.loadTriple(files[0], files[1], files[2]);
                auto &mat = textureManager.getMaterial(idx);
                mat = matSpec;
                loadedIndices.push_back(idx);
            }

            // Disable mapping for all loaded materials: set mappingMode to false
            for (size_t idx : loadedIndices) {
                auto &m = textureManager.getMaterial(idx);
                m.mappingMode = false; // none
            }

            // Initialize vegetation texture manager for billboard vegetation
            vegetationTextureManager.init(this);

            // Load vegetation textures (albedo/normal/opacity triples) and initialize MaterialProperties
            // Note: We use the height slot for opacity masks
            const std::vector<std::pair<std::array<const char*,3>, MaterialProperties>> vegSpecs = {
                {{"textures/vegetation/foliage_color.jpg", "textures/vegetation/foliage_normal.jpg", "textures/vegetation/foliage_opacity.jpg"}, MaterialProperties{true, false, 0.0f, 1.0f, 0.3f, 0.3f, 8.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
                {{"textures/vegetation/grass_color.jpg",   "textures/vegetation/grass_normal.jpg",   "textures/vegetation/grass_opacity.jpg"},   MaterialProperties{true, false, 0.0f, 1.0f, 0.35f, 0.3f, 8.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}},
                {{"textures/vegetation/wild_color.jpg",    "textures/vegetation/wild_normal.jpg",    "textures/vegetation/wild_opacity.jpg"},    MaterialProperties{true, false, 0.0f, 1.0f, 0.32f, 0.3f, 8.0f, 0.0f, 0.0f, false, 1.0f, 1.0f}}
            };

            for (const auto &entry : vegSpecs) {
                const auto &files = entry.first;
                const auto &mp = entry.second;
                size_t idx = vegetationTextureManager.loadTriple(files[0], files[1], files[2]);
                auto &mat = vegetationTextureManager.getMaterial(idx);
                mat = mp;
            }

            for (size_t vi = 0; vi < vegetationTextureManager.count(); ++vi) {
                auto &vm = vegetationTextureManager.getMaterial(vi);
                    vm.mappingMode = false;
            }

            // Auto-detect tiles from vegetation opacity maps
            std::cout << "Auto-detecting vegetation tiles from opacity maps..." << std::endl;
            int foliageTiles = vegetationAtlasManager.autoDetectTiles(0, "textures/vegetation/foliage_opacity.jpg", 10);
            std::cout << "  Foliage: detected " << foliageTiles << " tiles" << std::endl;
            
            int grassTiles = vegetationAtlasManager.autoDetectTiles(1, "textures/vegetation/grass_opacity.jpg", 10);
            std::cout << "  Grass: detected " << grassTiles << " tiles" << std::endl;
            
            int wildTiles = vegetationAtlasManager.autoDetectTiles(2, "textures/vegetation/wild_opacity.jpg", 10);
            std::cout << "  Wild: detected " << wildTiles << " tiles" << std::endl;

            // Initialize and add editable textures BEFORE creating descriptor sets
            editableTextures = std::make_shared<EditableTextureSet>();
            editableTextures->init(this, 1024, 1024, "Editable Textures");
            editableTextures->setTextureManager(&textureManager);
            editableTextures->generateInitialTextures();
            
            // Add editable textures to TextureManager as a regular triple
            size_t editableIndex = textureManager.addTriple(
                editableTextures->getAlbedo().getTextureImage(),
                editableTextures->getAlbedo().getSampler(),
                editableTextures->getNormal().getTextureImage(),
                editableTextures->getNormal().getSampler(),
                editableTextures->getBump().getTextureImage(),
                editableTextures->getBump().getSampler()
            );
            // Apply editable material properties in a single-line initializer
            textureManager.getMaterial(editableIndex) = MaterialProperties{
                false,  // mappingMode (none)
                false, // invertHeight
                0.2f,  // tessHeightScale (unused)
                16.0f, // tessLevel (unused)
                0.5f, // ambientFactor
                0.05f,  // specularStrength
                32.0f, // shininess
                0.0f,  // padding1
                0.0f,  // padding2
                false, // triplanar
                1.0f,  // triplanarScaleU
                1.0f   // triplanarScaleV
            };
            editableTextureIndex = editableIndex;  // Store for plane rendering
            // Initialize cached plane material from the editable texture triple
            planeMaterial = textureManager.getMaterial(editableIndex);

            // remove any zeros that might come from failed loads (TextureManager may throw or return an index; assume valid indices)
            if (!loadedIndices.empty()) currentTextureIndex = loadedIndices[0];
            
            // create uniform buffers - one per texture (including editable texture)
            uniforms.resize(textureManager.count());
            for (size_t i = 0; i < uniforms.size(); ++i) {
                uniforms[i] = createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            }
            // create shadow uniform buffers
            shadowUniforms.resize(textureManager.count());
            for (size_t i = 0; i < shadowUniforms.size(); ++i) {
                shadowUniforms[i] = createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            }
            
            // create descriptor sets - one per texture triple
            size_t tripleCount = textureManager.count();
            if (tripleCount == 0) tripleCount = 1; // ensure at least one
            // Create/upload GPU-side material buffer (packed vec4-friendly struct)
            updateMaterials();
            // Allocate descriptor pool sized for descriptor sets. We need room for per-texture sets and
            // per-model-instance sets (we'll create one descriptor set per Model3D instance later).
            // Estimate: per-texture sets = tripleCount * 4, per-instance sets = (1 + 2*tripleCount) * 2 (main+shadow), add a small slack.
            uint32_t estimatedPerTextureSets = static_cast<uint32_t>(tripleCount * 4);
            uint32_t estimatedModelObjects = static_cast<uint32_t>(1 + 2 * tripleCount);
            uint32_t estimatedPerInstanceSets = estimatedModelObjects * 2;
            uint32_t totalSets = estimatedPerTextureSets + estimatedPerInstanceSets + 4;
            // each descriptor set writes up to 4 combined image samplers (albedo/normal/height/shadow)
            createDescriptorPool(totalSets, totalSets * 4);

            descriptorSets.resize(tripleCount, VK_NULL_HANDLE);
            shadowDescriptorSets.resize(tripleCount, VK_NULL_HANDLE);

            // Use DescriptorSetBuilder to reduce repeated code
            DescriptorSetBuilder dsBuilder(this, &textureManager, &shadowMapper);
            for (size_t i = 0; i < tripleCount; ++i) {
                const auto &tr = textureManager.getTriple(i);
                // main descriptor set
                VkDeviceSize matElemSize = sizeof(glm::vec4) * 4; // size of MaterialGPU
                VkDescriptorSet ds = dsBuilder.createMainDescriptorSet(tr, uniforms[i], materialManager.count() > 0, materialManager.count() > 0 ? &materialManager.getBuffer() : nullptr, materialManager.count() > 0 ? static_cast<VkDeviceSize>(i) * matElemSize : 0);
                descriptorSets[i] = ds;

                // shadow descriptor set
                VkDescriptorSet sds = dsBuilder.createShadowDescriptorSet(tr, shadowUniforms[i], materialManager.count() > 0, materialManager.count() > 0 ? &materialManager.getBuffer() : nullptr, materialManager.count() > 0 ? static_cast<VkDeviceSize>(i) * matElemSize : 0);
                shadowDescriptorSets[i] = sds;
            }

            // Create per-texture descriptor sets and uniform buffers for sphere instances (one per texture triple)
            sphereDescriptorSets.resize(tripleCount, VK_NULL_HANDLE);
            sphereUniforms.resize(tripleCount);
            shadowSphereUniforms.resize(tripleCount);
            shadowSphereDescriptorSets.resize(tripleCount, VK_NULL_HANDLE);
            for (size_t i = 0; i < tripleCount; ++i) {
                // create uniform buffer for sphere instance
                sphereUniforms[i] = createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                shadowSphereUniforms[i] = createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                const auto &tr = textureManager.getTriple(i);
                VkDeviceSize matElemSize = sizeof(glm::vec4) * 4;
                VkDescriptorSet ds = dsBuilder.createSphereDescriptorSet(tr, sphereUniforms[i], materialManager.count() > 0, materialManager.count() > 0 ? &materialManager.getBuffer() : nullptr, materialManager.count() > 0 ? static_cast<VkDeviceSize>(i) * matElemSize : 0);
                sphereDescriptorSets[i] = ds;
                VkDescriptorSet sds = dsBuilder.createShadowSphereDescriptorSet(tr, shadowSphereUniforms[i], materialManager.count() > 0, materialManager.count() > 0 ? &materialManager.getBuffer() : nullptr, materialManager.count() > 0 ? static_cast<VkDeviceSize>(i) * matElemSize : 0);
                shadowSphereDescriptorSets[i] = sds;
            }

        

            // build ground plane mesh (20x20 units) and keep editable texture index
            auto plane = std::make_unique<PlaneModel>();
            // Build plane using the editable texture index (pass as last parameter)
            plane->build(20.0f, 20.0f, 1, 1, static_cast<float>(editableIndex));
            VertexBufferObject planeVbo = VertexBufferObjectBuilder::create(this, *plane);

            // Store plane mesh/vbo first so index 0/1 are reserved
            meshes.push_back(std::move(plane));
            meshVBOs.push_back(planeVbo);

            // Create per-material cube and sphere meshes (one pair per texture triple)
            size_t n = descriptorSets.size();
            if (n == 0) n = 1;
            float spacing = 2.5f;
            int cols = static_cast<int>(std::ceil(std::sqrt((float)n)));
            int rows = static_cast<int>(std::ceil((float)n / cols));
            float halfW = (cols - 1) * 0.5f * spacing;
            float halfH = (rows - 1) * 0.5f * spacing;

            modelObjects.clear();
            modelObjects.reserve(n * 2 + 1);

            // Reserve space for meshes and VBOs to avoid reallocation invalidating references
            meshVBOs.reserve(1 + 2 * n);
            meshes.reserve(1 + 2 * n);

            // Add plane as a Model3D wrapper (uses editable texture index)
            glm::mat4 planeModel = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.5f, 0.0f));
            modelObjects.emplace_back(std::make_unique<Model3D>(meshes[0].get(), meshVBOs[0], planeModel));

            for (size_t i = 0; i < n; ++i) {
                // Build a cube mesh whose per-face tex indices point to this material index
                auto cube = std::make_unique<CubeModel>();
                std::vector<float> faceTex(6, static_cast<float>(i));
                cube->build(faceTex);
                VertexBufferObject cubeVbo = VertexBufferObjectBuilder::create(this, *cube);

                Mesh3D* cubePtr = cube.get();
                meshes.push_back(std::move(cube));
                meshVBOs.push_back(cubeVbo);

                // Position cube in a grid
                int col = static_cast<int>(i % cols);
                int row = static_cast<int>(i / cols);
                float x = col * spacing - halfW;
                float y = -1.0f; // Above the plane
                float z = row * spacing - halfH;
                glm::mat4 cubeModel = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
                modelObjects.emplace_back(std::make_unique<Model3D>(cubePtr, meshVBOs.back(), cubeModel));

                // Build sphere mesh for this material
                auto sphere = std::make_unique<SphereModel>();
                sphere->build(0.5f, 32, 16, static_cast<float>(i));
                VertexBufferObject sphereVbo = VertexBufferObjectBuilder::create(this, *sphere);
                Mesh3D* spherePtr = sphere.get();
                meshes.push_back(std::move(sphere));
                meshVBOs.push_back(sphereVbo);

                // Place sphere above cube
                float sphereRadius = 0.5f;
                float cubeHalfHeight = 0.5f;
                float gap = 0.5f;
                float sphereCenterY = y + cubeHalfHeight + sphereRadius + gap;
                glm::mat4 sphereModel = glm::translate(glm::mat4(1.0f), glm::vec3(x, sphereCenterY, z));
                modelObjects.emplace_back(std::make_unique<Model3D>(spherePtr, meshVBOs.back(), sphereModel));
            }
            // Create per-model-instance uniform buffers and descriptor sets so each instance has its own UBO
            instanceDescriptorSets.clear();
            instanceUniforms.clear();
            instanceShadowDescriptorSets.clear();
            instanceShadowUniforms.clear();
            instanceDescriptorSets.resize(modelObjects.size(), VK_NULL_HANDLE);
            instanceUniforms.resize(modelObjects.size());
            instanceShadowDescriptorSets.resize(modelObjects.size(), VK_NULL_HANDLE);
            instanceShadowUniforms.resize(modelObjects.size());
            {
                DescriptorSetBuilder dsBuilder(this, &textureManager, &shadowMapper);
                VkDeviceSize matElemSize = sizeof(glm::vec4) * 4;
                for (size_t mi = 0; mi < modelObjects.size(); ++mi) {
                    Model3D* m = modelObjects[mi].get();
                    if (!m || !m->mesh) continue;
                    // determine triple index from mesh vertex texIndex (plane was built with editableIndex)
                    int texIdx = 0;
                    if (!m->mesh->getVertices().empty()) texIdx = m->mesh->getVertices()[0].texIndex;
                    if (texIdx < 0 || static_cast<size_t>(texIdx) >= textureManager.count()) texIdx = 0;
                    const auto &tr = textureManager.getTriple(static_cast<size_t>(texIdx));
                    // create per-instance uniform buffers
                    instanceUniforms[mi] = createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    instanceShadowUniforms[mi] = createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    // create descriptor sets bound to the per-instance buffers
                    instanceDescriptorSets[mi] = dsBuilder.createMainDescriptorSet(tr, instanceUniforms[mi], materialManager.count() > 0, materialManager.count() > 0 ? &materialManager.getBuffer() : nullptr, materialManager.count() > 0 ? static_cast<VkDeviceSize>(texIdx) * matElemSize : 0);
                    instanceShadowDescriptorSets[mi] = dsBuilder.createShadowDescriptorSet(tr, instanceShadowUniforms[mi], materialManager.count() > 0, materialManager.count() > 0 ? &materialManager.getBuffer() : nullptr, materialManager.count() > 0 ? static_cast<VkDeviceSize>(texIdx) * matElemSize : 0);
                }
            }
            
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
            textureViewer->init(&textureManager);
            // Rebuild GPU material buffer when materials are modified via the texture viewer
            textureViewer->setOnMaterialChanged([this](size_t idx) {
                (void)idx; // currently we rebuild entire buffer
                updateMaterials();
            });
            widgetManager.addWidget(textureViewer);
            
            // Add editable textures widget (already created above)
            editableTextures->setOnTextureGenerated([this, editableIndex]() {
                // Material properties persist, just textures get updated
                printf("Editable textures regenerated (index %zu)\n", editableIndex);
                // refresh GPU-side material buffer to pick up any editable material changes
                updateMaterials();
                // Update cached plane material to reflect regenerated editable textures
                planeMaterial = textureManager.getMaterial(editableIndex);
            });
            widgetManager.addWidget(editableTextures);
            
            // Create other widgets
            auto cameraWidget = std::make_shared<CameraWidget>(&camera);
            widgetManager.addWidget(cameraWidget);
            
            auto debugWidget = std::make_shared<DebugWidget>(&textureManager, &camera, &currentTextureIndex);
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
            skySphere->init(skyWidget.get(), descriptorSets, shadowDescriptorSets, sphereDescriptorSets, shadowSphereDescriptorSets);
            
            // Create vegetation atlas editor widget
            auto vegAtlasEditor = std::make_shared<VegetationAtlasEditor>(&vegetationTextureManager, &vegetationAtlasManager);
            widgetManager.addWidget(vegAtlasEditor);
            
            // Create billboard creator widget
            billboardCreator = std::make_shared<BillboardCreator>(&billboardManager, &vegetationAtlasManager, &vegetationTextureManager);
            billboardCreator->setVulkanApp(this);
            widgetManager.addWidget(billboardCreator);
            
            createCommandBuffers();
            mainScene = new LocalScene();

            MainSceneLoader mainSceneLoader = MainSceneLoader();
            mainScene->loadScene(mainSceneLoader);

            
        };

        // Upload all materials into a GPU storage buffer (called once during setup)
        void updateMaterials() {
            materialCount = textureManager.count();
            if (materialCount == 0) return;

            // Allocate GPU buffer via MaterialManager and upload per-index materials
            materialManager.allocate(materialCount, this);
            for (size_t mi = 0; mi < materialCount; ++mi) {
                const MaterialProperties &mat = textureManager.getMaterial(mi);
                materialManager.update(mi, mat, this);
            }

            // Rebind material buffer into descriptor sets so shaders read the new GPU buffer
            if (!descriptorSets.empty()) {
                VkDeviceSize matElemSize = sizeof(glm::vec4) * 4; // size of MaterialGPU
                DescriptorSetBuilder dsBuilder(this, &textureManager, &shadowMapper);
                dsBuilder.updateMaterialBinding(descriptorSets, materialManager.getBuffer(), matElemSize);
                dsBuilder.updateMaterialBinding(shadowDescriptorSets, materialManager.getBuffer(), matElemSize);
                dsBuilder.updateMaterialBinding(sphereDescriptorSets, materialManager.getBuffer(), matElemSize);
                dsBuilder.updateMaterialBinding(shadowSphereDescriptorSets, materialManager.getBuffer(), matElemSize);
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
                unsigned int version = node ? node->version : 0u;

                // If we don't have a Model3DVersion for this node yet, insert a default placeholder
                if (nodeModelVersions.find(node) == nodeModelVersions.end()) {
                    nodeModelVersions.try_emplace(node, Model3DVersion{});

                    // Request the Model3D and fill the stored Model3DVersion when available
                    // requestModel3D expects a non-const reference; cast away const here
                    mainScene->requestModel3D(Layer::LAYER_OPAQUE, const_cast<OctreeNodeData&>(data), [this, node, version](Mesh3D& model) {
                        // Store a heap-allocated copy to match Model3DVersion::model (pointer)
                        Mesh3D* stored = new Mesh3D(model);
                        nodeModelVersions[node] = { stored, version };
                        std::cout << "[Model3D] Loaded model for node " << node << " (version " << version << ")\n";
                    });
                } else {
                    visibleModels.push_back(nodeModelVersions[node].model);
                }
            });
            
            
            
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);
            
            // Clear previous frame's instances
            modelManager.clear();
 
            // Calculate grid layout for cubes
            size_t n = descriptorSets.size();
            float spacing = 2.5f;
            if (n == 0) n = 1;
            int cols = static_cast<int>(std::ceil(std::sqrt((float)n)));
            int rows = static_cast<int>(std::ceil((float)n / cols));
            float halfW = (cols - 1) * 0.5f * spacing;
            float halfH = (rows - 1) * 0.5f * spacing;
            
            // Add instances from the simple Model3D wrappers created at setup
            for (size_t mi = 0; mi < modelObjects.size(); ++mi) {
                Model3D* m = modelObjects[mi].get();
                if (!m || !m->mesh) continue;
                VertexBufferObject &vbo = m->getVBO();
                glm::mat4 transform = m->getModel();

                // Select a descriptor/uniform set for the object. Plane uses the editable texture index,
                // other objects use the first available texture descriptor (index 0) if present.
                // Use per-instance descriptor set / uniform buffer created during setup
                VkDescriptorSet ds = instanceDescriptorSets.size() > mi ? instanceDescriptorSets[mi] : VK_NULL_HANDLE;
                Buffer* ub = instanceUniforms.size() > mi ? &instanceUniforms[mi] : nullptr;
                VkDescriptorSet sds = instanceShadowDescriptorSets.size() > mi ? instanceShadowDescriptorSets[mi] : VK_NULL_HANDLE;
                Buffer* sub = instanceShadowUniforms.size() > mi ? &instanceShadowUniforms[mi] : nullptr;
                // material index is encoded in mesh vertex `texIndex`; pass instance without materialIndex
                modelManager.addInstance(m->mesh, vbo, transform, ds, ub, sds, sub);
            }
            
            // First pass: Render shadow map (skip if shadows globally disabled)
            if (!settingsWidget || settingsWidget->getShadowsEnabled()) {
                shadowMapper.beginShadowPass(commandBuffer, uboStatic.lightSpaceMatrix);
                
                // Render all instances to shadow map
                for (const auto& instance : modelManager.getInstances()) {
                    // Update uniform buffer for shadow pass: set viewProjection = lightSpaceMatrix
                        UniformObject shadowUbo = uboStatic;
                        shadowUbo.viewProjection = uboStatic.lightSpaceMatrix;
                    shadowUbo.materialFlags = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
                    // Per-material mapping/mappingParams are read from the Materials SSBO in shaders.
                    shadowUbo.passParams = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f); // isShadowPass = 1.0
    
                    updateUniformBuffer(*instance.shadowUniformBuffer, &shadowUbo, sizeof(UniformObject));
                    // Push model matrix for this draw into the pipeline via push constants
                    vkCmdPushConstants(commandBuffer, getPipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, 0, sizeof(glm::mat4), &instance.transform);
                    shadowMapper.renderObject(commandBuffer, instance.transform, instance.vbo,
                                            instance.shadowDescriptorSet);
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
            if (skyRenderer && meshes.size() > 2 && meshVBOs.size() > 2 && !sphereDescriptorSets.empty() && !sphereUniforms.empty()) {
                if (skySphere) skySphere->update();
                const auto &vbo = meshVBOs[2];
                skyRenderer->render(commandBuffer, vbo, sphereDescriptorSets[0], sphereUniforms[0], uboStatic, projMat, viewMat);
            }

            // bind pipeline
            // We'll bind the appropriate pipeline per-instance (regular or tessellated)

            // Render all instances with main pass
            for (const auto& instance : modelManager.getInstances()) {
                const auto& vbo = instance.vbo;
                
                // Bind vertex and index buffers
                VkBuffer vertexBuffers[] = { vbo.vertexBuffer.buffer };
                VkDeviceSize offsets[] = { 0 };
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
                vkCmdBindIndexBuffer(commandBuffer, vbo.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
                
                // Update uniform buffer for this instance: set viewProjection and model separately
                UniformObject ubo = uboStatic;
                ubo.viewProjection = projMat * viewMat;
                
                // Static material properties are read from the GPU-side material buffer (SSBO).
                // Determine material index from the mesh vertex `texIndex` and use it for per-instance overrides.
                const MaterialProperties* instMat = nullptr;
                if (instance.model && !instance.model->getVertices().empty()) {
                    int idx = instance.model->getVertices()[0].texIndex;
                    if (idx >= 0 && static_cast<size_t>(idx) < textureManager.count()) {
                        instMat = &textureManager.getMaterial(static_cast<size_t>(idx));
                    }
                }
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
                ubo.passParams = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f); // isShadowPass = 0.0
                updateUniformBuffer(*instance.uniformBuffer, &ubo, sizeof(UniformObject));
                
                // Bind descriptor set and draw
                // Use a single pipeline (tessellation is enabled/disabled in the shader using mappingMode)
                bool wire = settingsWidget ? settingsWidget->getWireframeEnabled() : false;
                if (wire && graphicsPipelineWire != VK_NULL_HANDLE) vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineWire);
                else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                       getPipelineLayout(), 0, 1, &instance.descriptorSet, 0, nullptr);
                // Push per-draw model matrix via push constants (visible to vertex + tessellation stages)
                vkCmdPushConstants(commandBuffer, getPipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, 0, sizeof(glm::mat4), &instance.transform);
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
            // texture cleanup via manager
            textureManager.destroyAll();
            // vertex/index buffers cleanup - destroy GPU buffers created from the meshes
            for (auto& vbo : meshVBOs) {
                vbo.destroy(getDevice());
            }
            meshVBOs.clear();
            meshes.clear();

            // uniform buffers cleanup
            for (auto& uniformBuffer : uniforms) {
                if (uniformBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(getDevice(), uniformBuffer.buffer, nullptr);
            }
            for (auto& uniformBuffer : uniforms) {
                if (uniformBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), uniformBuffer.memory, nullptr);
            }
            // per-instance uniform buffers cleanup
            for (auto& uniformBuffer : instanceUniforms) {
                if (uniformBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(getDevice(), uniformBuffer.buffer, nullptr);
            }
            for (auto& uniformBuffer : instanceUniforms) {
                if (uniformBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), uniformBuffer.memory, nullptr);
            }
            // sphere uniform buffers cleanup
            for (auto& uniformBuffer : sphereUniforms) {
                if (uniformBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(getDevice(), uniformBuffer.buffer, nullptr);
            }
            for (auto& uniformBuffer : sphereUniforms) {
                if (uniformBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), uniformBuffer.memory, nullptr);
            }
            
            // editable textures cleanup
            if (editableTextures) editableTextures->cleanup();
            
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

