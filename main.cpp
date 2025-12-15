#include "vulkan/VulkanApp.hpp"
#include "vulkan/FileReader.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include "utils/Camera.hpp"
#include "events/EventManager.hpp"
#include "events/WindowEvents.hpp"
#include "events/KeyboardPublisher.hpp"
#include "events/GamepadPublisher.hpp"
#include "vulkan/TextureManager.hpp"
#include "events/CameraEvents.hpp"
#include "vulkan/AtlasManager.hpp"
#include "utils/BillboardManager.hpp"
#include "utils/CubeMesh.hpp"
#include "utils/PlaneMesh.hpp"
#include "utils/SphereMesh.hpp"
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
#include "widgets/VegetationAtlasEditor.hpp"
#include "widgets/BillboardCreator.hpp"
#include <string>
#include <memory>
#include <iostream>
#include <cmath>

struct UniformObject {
    glm::mat4 mvp;
    glm::mat4 model;
    glm::vec4 viewPos; // world-space camera position
    glm::vec4 lightDir; // xyz = direction, w = unused (padding)
    glm::vec4 lightColor; // rgb = color, w = intensity
    // Material flags
    glm::vec4 materialFlags; // x=unused, y=unused, z=ambient, w=unused
    glm::vec4 mappingParams; // x=mappingEnabled (0=off,1=on) toggles tessellation + bump mapping, y/z/w unused
    glm::vec4 specularParams; // x=specularStrength, y=shininess, z=unused, w=unused
    glm::vec4 triplanarParams; // x=scaleU, y=scaleV, z=enabled(1.0), w=unused
    glm::mat4 lightSpaceMatrix; // for shadow mapping
    glm::vec4 shadowEffects; // x=enableSelfShadow, y=enableShadowDisplacement, z=selfShadowQuality, w=unused
    glm::vec4 debugParams; // x=debugMode (0=normal,1=normalVec,2=normalMap,3=uv,4=tangent,5=bitangent)
    glm::vec4 skyHorizon; // rgb = horizon color, a = unused
    glm::vec4 skyZenith;  // rgb = zenith color, a = unused
    glm::vec4 skyParams;  // x = warmth, y = exponent, z/w = unused
    glm::vec4 nightHorizon; // rgb = night horizon color
    glm::vec4 nightZenith;  // rgb = night zenith color
    glm::vec4 nightParams;  // x = night intensity, y = starIntensity
    glm::vec4 passParams;   // x = isShadowPass (1.0 for shadow pass, 0.0 for main pass)
    
    // Set material properties from MaterialProperties struct
        void setMaterial(const MaterialProperties& mat) {
        materialFlags = glm::vec4(0.0f, 0.0f, mat.ambientFactor, 0.0f);
        mappingParams = glm::vec4(mat.mappingMode ? 1.0f : 0.0f, mat.tessLevel, mat.invertHeight ? 1.0f : 0.0f, mat.tessHeightScale);
        specularParams = glm::vec4(mat.specularStrength, mat.shininess, 0.0f, 0.0f);
        triplanarParams = glm::vec4(mat.triplanarScaleU, mat.triplanarScaleV, mat.triplanar ? 1.0f : 0.0f, 0.0f);
    }
};

#include "vulkan/VertexBufferObjectBuilder.hpp"

class MyApp : public VulkanApp, public IEventHandler {
    public:
        MyApp() : shadowMapper(this, 8192) {}


            void postSubmit() override {
                
            }

        // postSubmit() override removed to disable slow per-frame shadow readback
        
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    
    // GPU buffers for the built meshes (parallel to `meshes`)
    std::vector<VertexBufferObject> meshVBOs;
    VkPipeline graphicsPipelineWire = VK_NULL_HANDLE;
    VkPipeline graphicsPipelineTess = VK_NULL_HANDLE;
    VkPipeline graphicsPipelineTessWire = VK_NULL_HANDLE;
    VkPipeline skyPipeline = VK_NULL_HANDLE;
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
    std::vector<std::unique_ptr<Model3D>> meshes;
    
    // UI: currently selected texture triple for preview
    size_t currentTextureIndex = 0;
    size_t editableTextureIndex = 0;  // Index of the editable texture triple in TextureManager
    
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
        // Additional per-texture uniform buffers and descriptor sets for sphere instances
        std::vector<VkDescriptorSet> sphereDescriptorSets;
        std::vector<Buffer> sphereUniforms;
        // Shadow pass buffers and descriptor sets
        std::vector<Buffer> shadowUniforms;
        std::vector<VkDescriptorSet> shadowDescriptorSets;
        std::vector<Buffer> shadowSphereUniforms;
        std::vector<VkDescriptorSet> shadowSphereDescriptorSets;
        
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

                graphicsPipeline = createGraphicsPipeline(
                    {
                        vertexShader.info, 
                        fragmentShader.info
                    },
                    VkVertexInputBindingDescription { 
                        0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX 
                    },
                    {
                        VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
                        VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                        VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) },
                        VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                        VkVertexInputAttributeDescription { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent) },
                        VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SFLOAT, offsetof(Vertex, texIndex) }
                    }
                );

                // Create wireframe variant (if device supports it)
                graphicsPipelineWire = createGraphicsPipeline(
                    {
                        vertexShader.info,
                        fragmentShader.info
                    },
                    VkVertexInputBindingDescription { 
                        0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX 
                    },
                    {
                        VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
                        VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                        VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) },
                        VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                        VkVertexInputAttributeDescription { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent) },
                        VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SFLOAT, offsetof(Vertex, texIndex) }
                    },
                    VK_POLYGON_MODE_LINE
                );

                // create a tessellated pipeline as well (vertex + tesc + tese + fragment)
                ShaderStage tescShader = ShaderStage(
                    createShaderModule(FileReader::readFile("shaders/main.tesc.spv")),
                    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
                );
                ShaderStage teseShader = ShaderStage(
                    createShaderModule(FileReader::readFile("shaders/main.tese.spv")),
                    VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
                );

                // Build tessellated pipeline
                graphicsPipelineTess = createGraphicsPipeline(
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
                        VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
                        VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                        VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) },
                        VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                        VkVertexInputAttributeDescription { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent) },
                        VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SFLOAT, offsetof(Vertex, texIndex) }
                    }
                );

                // Build tessellated wireframe pipeline
                graphicsPipelineTessWire = createGraphicsPipeline(
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
                        VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
                        VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                        VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) },
                        VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                        VkVertexInputAttributeDescription { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent) },
                        VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SFLOAT, offsetof(Vertex, texIndex) }
                    },
                    VK_POLYGON_MODE_LINE
                );

                // Destroy tess shader modules after pipeline creation
                vkDestroyShaderModule(getDevice(), tescShader.info.module, nullptr);
                vkDestroyShaderModule(getDevice(), teseShader.info.module, nullptr);

                // Destroy vertex/fragment modules used to create the pipelines
                vkDestroyShaderModule(getDevice(), fragmentShader.info.module, nullptr);
                vkDestroyShaderModule(getDevice(), vertexShader.info.module, nullptr);

                // Create sky pipeline (renders large inside-out sphere with no depth writes)
                {
                    ShaderStage skyVert = ShaderStage(
                        createShaderModule(FileReader::readFile("shaders/sky.vert.spv")),
                        VK_SHADER_STAGE_VERTEX_BIT
                    );
                    ShaderStage skyFrag = ShaderStage(
                        createShaderModule(FileReader::readFile("shaders/sky.frag.spv")),
                        VK_SHADER_STAGE_FRAGMENT_BIT
                    );

                    skyPipeline = createGraphicsPipeline(
                        { skyVert.info, skyFrag.info },
                        VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX },
                        {
                            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
                            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                            VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) },
                            VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                            VkVertexInputAttributeDescription { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent) },
                            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SFLOAT, offsetof(Vertex, texIndex) }
                        },
                        VK_POLYGON_MODE_FILL, VK_CULL_MODE_FRONT_BIT, false
                    );

                    vkDestroyShaderModule(getDevice(), skyFrag.info.module, nullptr);
                    vkDestroyShaderModule(getDevice(), skyVert.info.module, nullptr);
                }
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
            // Allocate descriptor pool sized for both cube and sphere descriptor sets (4 sets per texture triple)
            // uboCount = number of uniform descriptors (one per descriptor set), samplerCount = total combined image sampler descriptors
            createDescriptorPool(static_cast<uint32_t>(tripleCount * 4), static_cast<uint32_t>(tripleCount * 16));

            descriptorSets.resize(tripleCount, VK_NULL_HANDLE);
            shadowDescriptorSets.resize(tripleCount, VK_NULL_HANDLE);
            for (size_t i = 0; i < tripleCount; ++i) {
                VkDescriptorSet ds = createDescriptorSet(getDescriptorSetLayout());
                descriptorSets[i] = ds;

                // uniform buffer descriptor (binding 0) - use separate uniform buffer per cube
                VkDescriptorBufferInfo bufferInfo { uniforms[i].buffer, 0 , sizeof(UniformObject) };

                VkWriteDescriptorSet uboWrite{};
                uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                uboWrite.dstSet = ds;
                uboWrite.dstBinding = 0;
                uboWrite.dstArrayElement = 0;
                uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                uboWrite.descriptorCount = 1;
                uboWrite.pBufferInfo = &bufferInfo;

                // bind combined image sampler descriptors for texture arrays (single descriptor each)
                const auto &tr = textureManager.getTriple(i);
                VkDescriptorImageInfo imageInfo { tr.albedoSampler, tr.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                VkWriteDescriptorSet samplerWrite{};
                samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                samplerWrite.dstSet = ds;
                samplerWrite.dstBinding = 1;
                samplerWrite.dstArrayElement = 0;
                samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                samplerWrite.descriptorCount = 1;
                samplerWrite.pImageInfo = &imageInfo;

                VkDescriptorImageInfo normalInfo { tr.normalSampler, tr.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                VkWriteDescriptorSet normalWrite{};
                normalWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                normalWrite.dstSet = ds;
                normalWrite.dstBinding = 2;
                normalWrite.dstArrayElement = 0;
                normalWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                normalWrite.descriptorCount = 1;
                normalWrite.pImageInfo = &normalInfo;

                VkDescriptorImageInfo heightInfo { tr.heightSampler, tr.height.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                VkWriteDescriptorSet heightWrite{};
                heightWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                heightWrite.dstSet = ds;
                heightWrite.dstBinding = 3;
                heightWrite.dstBinding = 3;
                heightWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                heightWrite.descriptorCount = 1;
                heightWrite.pImageInfo = &heightInfo;
                
                // Shadow map descriptor (binding 4)
                VkDescriptorImageInfo shadowInfo { shadowMapper.getShadowMapSampler(), shadowMapper.getShadowMapView(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
                VkWriteDescriptorSet shadowWrite{};
                shadowWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                shadowWrite.dstSet = ds;
                shadowWrite.dstBinding = 4;
                shadowWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                shadowWrite.descriptorCount = 1;
                shadowWrite.pImageInfo = &shadowInfo;

                updateDescriptorSet(
                    ds,
                    { uboWrite, samplerWrite, normalWrite, heightWrite, shadowWrite }
                );

                // Create shadow descriptor set
                VkDescriptorSet sds = createDescriptorSet(getDescriptorSetLayout());
                shadowDescriptorSets[i] = sds;

                // uniform buffer descriptor (binding 0) - use shadow uniform buffer
                VkDescriptorBufferInfo shadowBufferInfo { shadowUniforms[i].buffer, 0 , sizeof(UniformObject) };
                VkWriteDescriptorSet shadowUboWrite{};
                shadowUboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                shadowUboWrite.dstSet = sds;
                shadowUboWrite.dstBinding = 0;
                shadowUboWrite.dstArrayElement = 0;
                shadowUboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                shadowUboWrite.descriptorCount = 1;
                shadowUboWrite.pBufferInfo = &shadowBufferInfo;

                // Ensure sampler/texture writes target the shadow descriptor set as well
                samplerWrite.dstSet = sds;
                normalWrite.dstSet = sds;
                heightWrite.dstSet = sds;
                shadowWrite.dstSet = sds;

                updateDescriptorSet(
                    sds,
                    { shadowUboWrite, samplerWrite, normalWrite, heightWrite, shadowWrite }
                );
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
                VkDescriptorSet ds = createDescriptorSet(getDescriptorSetLayout());
                sphereDescriptorSets[i] = ds;
                VkDescriptorSet sds = createDescriptorSet(getDescriptorSetLayout());
                shadowSphereDescriptorSets[i] = sds;

                // reuse image samplers from texture triple i
                const auto &tr = textureManager.getTriple(i);
                VkDescriptorBufferInfo bufferInfo { sphereUniforms[i].buffer, 0 , sizeof(UniformObject) };
                VkDescriptorImageInfo imageInfo { tr.albedoSampler, tr.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                VkDescriptorImageInfo normalInfo { tr.normalSampler, tr.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                VkDescriptorImageInfo heightInfo { tr.heightSampler, tr.height.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                VkDescriptorImageInfo shadowInfo { shadowMapper.getShadowMapSampler(), shadowMapper.getShadowMapView(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };

                VkWriteDescriptorSet uboWrite{}; uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; uboWrite.dstSet = ds; uboWrite.dstBinding = 0; uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; uboWrite.descriptorCount = 1; uboWrite.pBufferInfo = &bufferInfo;
                VkWriteDescriptorSet samplerWrite{}; samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; samplerWrite.dstSet = ds; samplerWrite.dstBinding = 1; samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; samplerWrite.descriptorCount = 1; samplerWrite.pImageInfo = &imageInfo;
                VkWriteDescriptorSet normalWrite{}; normalWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; normalWrite.dstSet = ds; normalWrite.dstBinding = 2; normalWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; normalWrite.descriptorCount = 1; normalWrite.pImageInfo = &normalInfo;
                VkWriteDescriptorSet heightWrite{}; heightWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; heightWrite.dstSet = ds; heightWrite.dstBinding = 3; heightWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; heightWrite.descriptorCount = 1; heightWrite.pImageInfo = &heightInfo;
                VkWriteDescriptorSet shadowWrite{}; shadowWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; shadowWrite.dstSet = ds; shadowWrite.dstBinding = 4; shadowWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; shadowWrite.descriptorCount = 1; shadowWrite.pImageInfo = &shadowInfo;
                updateDescriptorSet(ds, { uboWrite, samplerWrite, normalWrite, heightWrite, shadowWrite });

                // Create shadow descriptor set for spheres
                VkDescriptorBufferInfo shadowBufferInfo { shadowSphereUniforms[i].buffer, 0 , sizeof(UniformObject) };
                VkWriteDescriptorSet shadowUboWrite = uboWrite;
                shadowUboWrite.dstSet = sds;
                shadowUboWrite.pBufferInfo = &shadowBufferInfo;
                // Make sure the image sampler writes also target the shadow descriptor set
                samplerWrite.dstSet = sds;
                normalWrite.dstSet = sds;
                heightWrite.dstSet = sds;
                shadowWrite.dstSet = sds;
                updateDescriptorSet(sds, { shadowUboWrite, samplerWrite, normalWrite, heightWrite, shadowWrite });
            }

            // build cube mesh and geometry (per-face tex indices all zero by default)
            auto cube = std::make_unique<CubeMesh>();
            cube->build({});
            VertexBufferObject cubeVbo = VertexBufferObjectBuilder::create(this, *cube);
            
            // build ground plane mesh (20x20 units)
            auto plane = std::make_unique<PlaneMesh>();
            plane->build(20.0f, 20.0f, 0.0f); // Will use editable texture index
            VertexBufferObject planeVbo = VertexBufferObjectBuilder::create(this, *plane);
            
            // build sphere mesh
            auto sphere = std::make_unique<SphereMesh>();
            sphere->build(0.5f, 32, 16, 0.0f);
            VertexBufferObject sphereVbo = VertexBufferObjectBuilder::create(this, *sphere);
            // Store meshes and their GPU buffers for later use
            meshes.push_back(std::move(cube));
            meshVBOs.push_back(cubeVbo);
            meshes.push_back(std::move(plane));
            meshVBOs.push_back(planeVbo);
            meshes.push_back(std::move(sphere));
            meshVBOs.push_back(sphereVbo);
            
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
            widgetManager.addWidget(textureViewer);
            
            // Add editable textures widget (already created above)
            editableTextures->setOnTextureGenerated([this, editableIndex]() {
                // Material properties persist, just textures get updated
                printf("Editable textures regenerated (index %zu)\n", editableIndex);
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
            if (skyWidget) {
                uboStatic.skyHorizon = glm::vec4(skyWidget->getHorizonColor(), 1.0f);
                uboStatic.skyZenith = glm::vec4(skyWidget->getZenithColor(), 1.0f);
                uboStatic.skyParams = glm::vec4(skyWidget->getWarmth(), skyWidget->getExponent(), skyWidget->getSunFlare(), 0.0f);
                uboStatic.nightHorizon = glm::vec4(skyWidget->getNightHorizon(), 1.0f);
                uboStatic.nightZenith = glm::vec4(skyWidget->getNightZenith(), 1.0f);
                uboStatic.nightParams = glm::vec4(skyWidget->getNightIntensity(), skyWidget->getStarIntensity(), 0.0f, 0.0f);
            }
            
            // Create vegetation atlas editor widget
            auto vegAtlasEditor = std::make_shared<VegetationAtlasEditor>(&vegetationTextureManager, &vegetationAtlasManager);
            widgetManager.addWidget(vegAtlasEditor);
            
            // Create billboard creator widget
            billboardCreator = std::make_shared<BillboardCreator>(&billboardManager, &vegetationAtlasManager, &vegetationTextureManager);
            billboardCreator->setVulkanApp(this);
            widgetManager.addWidget(billboardCreator);
            
            createCommandBuffers();
        };

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
                
                ImGui::SameLine(ImGui::GetIO().DisplaySize.x - 120);
                ImGui::Text("FPS: %.1f", imguiFps);
                ImGui::EndMainMenuBar();
            }

            if (imguiShowDemo) ImGui::ShowDemoWindow(&imguiShowDemo);

            // Render all widgets
            widgetManager.renderAll();
        }

        void update(float deltaTime) override {
            // Process queued events (dispatch to handlers)
            eventManager.processQueued();
            // compute MVP = proj * view * model
            glm::mat4 proj = glm::mat4(1.0f);
            glm::mat4 view = glm::mat4(1.0f);
            glm::mat4 model = glm::mat4(1.0f);
            glm::mat4 tmp = glm::mat4(1.0f);
            glm::mat4 mvp = glm::mat4(1.0f);

            float aspect = (float)getWidth() / (float) getHeight();
            proj = glm::perspective(45.0f * 3.1415926f / 180.0f, aspect, 0.1f, 100.0f);
            // flip Y for Vulkan clip space
            proj[1][1] *= -1.0f;

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
            view = camera.getViewMatrix();

            // keep model identity (stop cube rotation)
            model = glm::mat4(1.0f);

            tmp = view * model;
            mvp = proj * view * model;

            // store projection and view for per-cube MVP computation in draw()
            projMat = proj;
            viewMat = view;

            // prepare static parts of the UBO (viewPos, light, material flags) - model and mvp will be set per-cube in draw()
            uboStatic.viewPos = glm::vec4(camera.getPosition(), 1.0f);
            // The UI `lightDirection` represents a vector TO the light. Shaders expect a light vector that points FROM the
            // light TOWARD the surface when performing lighting/shadow calculations. Send the negated direction to the GPU
            // so both lighting and shadow projection use the same convention.
            glm::vec3 lightDir = glm::normalize(lightDirection);
            uboStatic.lightDir = glm::vec4(lightDir, 0.0f);
            // Debug: print UI lightDirection and sent UBO lightDir
            std::cout << "[UBO] UI lightDirection=(" << lightDirection.x << ", " << lightDirection.y << ", " << lightDirection.z << ")\n";
            std::cout << "[UBO] sent ubo.lightDir=(" << uboStatic.lightDir.x << ", " << uboStatic.lightDir.y << ", " << uboStatic.lightDir.z << ")\n";
            uboStatic.lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
            // Note: material flags and specularParams are set per-instance from material properties
            // Update sky UBO values from widget (if present)
            if (skyWidget) {
                uboStatic.skyHorizon = glm::vec4(skyWidget->getHorizonColor(), 1.0f);
                uboStatic.skyZenith = glm::vec4(skyWidget->getZenithColor(), 1.0f);
                uboStatic.skyParams = glm::vec4(skyWidget->getWarmth(), skyWidget->getExponent(), skyWidget->getSunFlare(), 0.0f);
                uboStatic.nightHorizon = glm::vec4(skyWidget->getNightHorizon(), 1.0f);
                uboStatic.nightZenith = glm::vec4(skyWidget->getNightZenith(), 1.0f);
                uboStatic.nightParams = glm::vec4(skyWidget->getNightIntensity(), skyWidget->getStarIntensity(), 0.0f, 0.0f);
            }
            
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
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);
            
            // Clear previous frame's instances
            modelManager.clear();
            
            // Get mesh pointers (cube at index 0, plane at index 1)
            Model3D* cube = meshes[0].get();
            Model3D* plane = meshes[1].get();
            
            // Calculate grid layout for cubes
            size_t n = descriptorSets.size();
            float spacing = 2.5f;
            if (n == 0) n = 1;
            int cols = static_cast<int>(std::ceil(std::sqrt((float)n)));
            int rows = static_cast<int>(std::ceil((float)n / cols));
            float halfW = (cols - 1) * 0.5f * spacing;
            float halfH = (rows - 1) * 0.5f * spacing;
            
            // Add plane instance (uses editable texture)
            glm::mat4 planeModel = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.5f, 0.0f));
            const MaterialProperties* planeMat = &textureManager.getMaterial(editableTextureIndex);
            modelManager.addInstance(plane, meshVBOs[1], planeModel, descriptorSets[editableTextureIndex], &uniforms[editableTextureIndex], shadowDescriptorSets[editableTextureIndex], &shadowUniforms[editableTextureIndex], planeMat);
            
            // Add cube instances
            for (size_t i = 0; i < descriptorSets.size(); ++i) {
                // Skip the editable texture index (it's used for the plane)
                if (i == editableTextureIndex) continue;
                
                int col = static_cast<int>(i % cols);
                int row = static_cast<int>(i / cols);
                
                float x = col * spacing - halfW;
                float y = -1.0f; // Above the plane
                float z = row * spacing - halfH;
                
                glm::mat4 model_i = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
                // Pass material properties for this texture
                const MaterialProperties* mat = &textureManager.getMaterial(i);
                modelManager.addInstance(cube, meshVBOs[0], model_i, descriptorSets[i], &uniforms[i], shadowDescriptorSets[i], &shadowUniforms[i], mat);

                // Create sphere above this cube if sphere mesh available
                if (meshes.size() > 2) {
                    Model3D* sphere = meshes[2].get();
                    float sphereRadius = 0.5f; // match mesh radius
                    // Place the sphere just above the cube top with a small gap to avoid intersection
                    float cubeHalfHeight = 0.5f;
                    float gap = 0.5f; // slightly larger gap to avoid numerical intersection
                    float sphereCenterY = y + cubeHalfHeight + sphereRadius + gap; // place on top of cube
                    glm::mat4 sphereModel = glm::translate(glm::mat4(1.0f), glm::vec3(x, sphereCenterY, z));
                    modelManager.addInstance(sphere, meshVBOs[2], sphereModel, sphereDescriptorSets[i], &sphereUniforms[i], shadowSphereDescriptorSets[i], &shadowSphereUniforms[i], mat);
                }
            }
            
            // Add spheres above each cube instance (if sphere mesh is present)

            // Helper: compute tessellation level for an instance based on camera distance and settings
            auto computeTess = [&](const glm::mat4 &modelMat, const MaterialProperties* mat) -> float {
                float baseTess = mat ? mat->tessLevel : 1.0f;
                glm::vec3 camPosForTess = camera.getPosition();
                glm::vec3 objPos = glm::vec3(modelMat[3]);
                float dist = glm::distance(camPosForTess, objPos);
                float tessForThis = baseTess;
                if (settingsWidget && settingsWidget->getAdaptiveTessellation()) {
                    float maxDist = settingsWidget->getTessMaxDistance();
                    float t = glm::clamp(dist / maxDist, 0.0f, 1.0f);
                    float maxLevel = settingsWidget->getTessMaxLevel() + baseTess;
                    float minLevel = settingsWidget->getTessMinLevel();
                    tessForThis = glm::mix(maxLevel, minLevel, t);
                }
                return glm::clamp(tessForThis, 1.0f, 64.0f);
            };

            // First pass: Render shadow map (skip if shadows globally disabled)
            if (!settingsWidget || settingsWidget->getShadowsEnabled()) {
                shadowMapper.beginShadowPass(commandBuffer, uboStatic.lightSpaceMatrix);
                
                // Render all instances to shadow map
                for (const auto& instance : modelManager.getInstances()) {
                // Build material flags for shadow pass
                glm::vec4 materialFlags(0.0f, 0.0f, 0.0f, 0.0f);
                if (instance.material) {
                    materialFlags = glm::vec4(
                        0.0f,
                        0.0f,
                        0.0f, // ambient not used in shadow pass
                        0.0f
                    );
                }

                // Update uniform buffer for shadow pass
                glm::mat4 shadowMvp = uboStatic.lightSpaceMatrix * instance.transform;
                UniformObject shadowUbo = uboStatic;
                shadowUbo.model = instance.transform;
                shadowUbo.mvp = shadowMvp;
                shadowUbo.materialFlags = materialFlags;
                // Set mapping mode for tessellation displacement in shadow pass
                float tessForThis = computeTess(instance.transform, instance.material);
                shadowUbo.mappingParams = glm::vec4(
                    instance.material ? (instance.material->mappingMode ? 1.0f : 0.0f) : 0.0f,
                    tessForThis,
                    instance.material ? (instance.material->invertHeight ? 1.0f : 0.0f) : 0.0f,
                    instance.material ? instance.material->tessHeightScale : 0.1f
                );
                shadowUbo.passParams = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f); // isShadowPass = 1.0
                updateUniformBuffer(*instance.shadowUniformBuffer, &shadowUbo, sizeof(UniformObject));
                
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
            if (skyPipeline != VK_NULL_HANDLE && meshes.size() > 2 && meshVBOs.size() > 2 && !sphereDescriptorSets.empty() && !sphereUniforms.empty()) {
                // update sky uniform (centered at camera)
                UniformObject skyUbo = uboStatic;
                glm::vec3 camPos = glm::vec3(uboStatic.viewPos);
                glm::mat4 model = glm::translate(glm::mat4(1.0f), camPos) * glm::scale(glm::mat4(1.0f), glm::vec3(50.0f));
                skyUbo.model = model;
                skyUbo.mvp = projMat * viewMat * model;
                skyUbo.passParams = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f); // isShadowPass = 0.0
                // use first sphere uniform buffer / descriptor set to bind UBO
                updateUniformBuffer(sphereUniforms[0], &skyUbo, sizeof(UniformObject));

                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipelineLayout(), 0, 1, &sphereDescriptorSets[0], 0, nullptr);

                // bind sphere vertex/index buffers
                const auto &vbo = meshVBOs[2];
                VkBuffer vertexBuffers[] = { vbo.vertexBuffer.buffer };
                VkDeviceSize offsets[] = { 0 };
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
                vkCmdBindIndexBuffer(commandBuffer, vbo.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
                vkCmdDrawIndexed(commandBuffer, vbo.indexCount, 1, 0, 0, 0);
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
                
                // Update uniform buffer for this instance
                glm::mat4 mvp = projMat * viewMat * instance.transform;
                UniformObject ubo = uboStatic;
                ubo.model = instance.transform;
                ubo.mvp = mvp;
                
                // Apply per-texture material properties if available
                if (instance.material) {
                    ubo.setMaterial(*instance.material);
                }
                // Override tessellation level in mappingParams based on camera distance
                ubo.mappingParams.y = computeTess(instance.transform, instance.material);
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
                // Choose pipeline based on per-material mapping enabled flag (false=none, true=tessellation+bump)
                bool mappingEnabled = (instance.material) ? instance.material->mappingMode : false;
                bool wire = settingsWidget ? settingsWidget->getWireframeEnabled() : false;
                if (mappingEnabled && graphicsPipelineTess != VK_NULL_HANDLE) {
                    if (wire && graphicsPipelineTessWire != VK_NULL_HANDLE) vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineTessWire);
                    else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineTess);
                } else {
                    if (wire && graphicsPipelineWire != VK_NULL_HANDLE) vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineWire);
                    else vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
                }

                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                       getPipelineLayout(), 0, 1, &instance.descriptorSet, 0, nullptr);
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
            if (graphicsPipelineTess != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), graphicsPipelineTess, nullptr);
            if (graphicsPipelineTessWire != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), graphicsPipelineTessWire, nullptr);
            if (skyPipeline != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), skyPipeline, nullptr);
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

