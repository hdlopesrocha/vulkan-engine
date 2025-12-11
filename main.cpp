#include "vulkan/vulkan.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include "vulkan/Camera.hpp"
#include "events/EventManager.hpp"
#include "events/WindowEvents.hpp"
#include "events/KeyboardPublisher.hpp"
#include "events/GamepadPublisher.hpp"
#include "vulkan/TextureManager.hpp"
#include "events/CameraEvents.hpp"
#include "vulkan/AtlasManager.hpp"
#include "vulkan/BillboardManager.hpp"
#include "vulkan/CubeMesh.hpp"
#include "vulkan/PlaneMesh.hpp"
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
#include "widgets/VegetationAtlasEditor.hpp"
#include "widgets/BillboardCreator.hpp"
#include <string>
#include <memory>
#include <iostream>

struct UniformObject {
    glm::mat4 mvp;
    glm::mat4 model;
    glm::vec4 viewPos; // world-space camera position
    glm::vec4 lightDir; // xyz = direction, w = unused (padding)
    glm::vec4 lightColor; // rgb = color, w = intensity
    glm::vec4 pomParams; // x=heightScale, y=minLayers, z=maxLayers, w=enabled
    glm::vec4 pomFlags;  // x=flipNormalY, y=flipTangentHandedness, z=ambient, w=flipParallaxDirection
    glm::vec4 specularParams; // x=specularStrength, y=shininess, z=unused, w=unused
    glm::mat4 lightSpaceMatrix; // for shadow mapping
    glm::vec4 shadowEffects; // x=enableSelfShadow, y=enableShadowDisplacement, z=selfShadowQuality, w=unused
    
    // Set material properties from MaterialProperties struct
    void setMaterial(const MaterialProperties& mat) {
        pomParams = glm::vec4(mat.pomHeightScale, mat.pomMinLayers, mat.pomMaxLayers, mat.pomEnabled);
        pomFlags = glm::vec4(mat.flipNormalY, mat.flipTangentHandedness, mat.ambientFactor, mat.flipParallaxDirection);
        specularParams = glm::vec4(mat.specularStrength, mat.shininess, 0.0f, 0.0f);
    }
};

class MyApp : public VulkanApp, public IEventHandler {
    public:
        MyApp() : shadowMapper(this, 8192) {}
        
        VkPipeline graphicsPipeline = VK_NULL_HANDLE;
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
    glm::vec3 lightDirection = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));
    
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

            // Initialize shadow mapper
            shadowMapper.init();
            
            // Graphics Pipeline
            {
                ShaderStage vertexShader = ShaderStage(
                    createShaderModule(FileReader::readFile("shaders/triangle.vert.spv")), 
                    VK_SHADER_STAGE_VERTEX_BIT
                );

                ShaderStage fragmentShader = ShaderStage(
                    createShaderModule(FileReader::readFile("shaders/triangle.frag.spv")), 
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
                            VkVertexInputAttributeDescription { 4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent) },
                            VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SFLOAT, offsetof(Vertex, texIndex) }
                    }
                );

                vkDestroyShaderModule(getDevice(), fragmentShader.info.module, nullptr);
                vkDestroyShaderModule(getDevice(), vertexShader.info.module, nullptr);
            }
            // initialize texture manager and explicitly load known albedo/normal/height triples by filename
            textureManager.init(this);
            std::vector<size_t> loadedIndices;

            // Explicit per-name loads (one-by-one) with realistic material properties
            // Bricks - rough masonry surface
            size_t bricksIdx = textureManager.loadTriple("textures/bricks_color.jpg", "textures/bricks_normal.jpg", "textures/bricks_bump.jpg");
            textureManager.getMaterial(bricksIdx).pomHeightScale = 0.08f;
            textureManager.getMaterial(bricksIdx).pomMinLayers = 10.0f;
            textureManager.getMaterial(bricksIdx).pomMaxLayers = 40.0f;
            textureManager.getMaterial(bricksIdx).specularStrength = 0.2f;
            textureManager.getMaterial(bricksIdx).shininess = 8.0f;
            textureManager.getMaterial(bricksIdx).ambientFactor = 0.12f;
            loadedIndices.push_back(bricksIdx);
            
            // Dirt - rough, non-reflective organic material
            size_t dirtIdx = textureManager.loadTriple("textures/dirt_color.jpg", "textures/dirt_normal.jpg", "textures/dirt_bump.jpg");
            textureManager.getMaterial(dirtIdx).pomHeightScale = 0.05f;
            textureManager.getMaterial(dirtIdx).pomMinLayers = 8.0f;
            textureManager.getMaterial(dirtIdx).pomMaxLayers = 32.0f;
            textureManager.getMaterial(dirtIdx).specularStrength = 0.05f;
            textureManager.getMaterial(dirtIdx).shininess = 4.0f;
            textureManager.getMaterial(dirtIdx).ambientFactor = 0.15f;
            loadedIndices.push_back(dirtIdx);
            
            // Forest floor - organic, moderately rough
            size_t forestIdx = textureManager.loadTriple("textures/forest_color.jpg", "textures/forest_normal.jpg", "textures/forest_bump.jpg");
            textureManager.getMaterial(forestIdx).pomHeightScale = 0.06f;
            textureManager.getMaterial(forestIdx).pomMinLayers = 8.0f;
            textureManager.getMaterial(forestIdx).pomMaxLayers = 32.0f;
            textureManager.getMaterial(forestIdx).specularStrength = 0.1f;
            textureManager.getMaterial(forestIdx).shininess = 6.0f;
            textureManager.getMaterial(forestIdx).ambientFactor = 0.18f;
            loadedIndices.push_back(forestIdx);
            
            // Grass - soft, slightly reflective when wet
            size_t grassIdx = textureManager.loadTriple("textures/grass_color.jpg", "textures/grass_normal.jpg", "textures/grass_bump.jpg");
            textureManager.getMaterial(grassIdx).pomHeightScale = 0.04f;
            textureManager.getMaterial(grassIdx).pomMinLayers = 8.0f;
            textureManager.getMaterial(grassIdx).pomMaxLayers = 28.0f;
            textureManager.getMaterial(grassIdx).specularStrength = 0.15f;
            textureManager.getMaterial(grassIdx).shininess = 10.0f;
            textureManager.getMaterial(grassIdx).ambientFactor = 0.2f;
            loadedIndices.push_back(grassIdx);
            
            // Lava - bright, highly emissive, some specular from molten surface
            size_t lavaIdx = textureManager.loadTriple("textures/lava_color.jpg", "textures/lava_normal.jpg", "textures/lava_bump.jpg");
            textureManager.getMaterial(lavaIdx).pomHeightScale = 0.03f;
            textureManager.getMaterial(lavaIdx).pomMinLayers = 6.0f;
            textureManager.getMaterial(lavaIdx).pomMaxLayers = 24.0f;
            textureManager.getMaterial(lavaIdx).specularStrength = 0.8f;
            textureManager.getMaterial(lavaIdx).shininess = 64.0f;
            textureManager.getMaterial(lavaIdx).ambientFactor = 0.4f; // Self-illuminated
            loadedIndices.push_back(lavaIdx);
            
            // Metal - highly reflective, smooth surface
            size_t metalIdx = textureManager.loadTriple("textures/metal_color.jpg", "textures/metal_normal.jpg", "textures/metal_bump.jpg");
            textureManager.getMaterial(metalIdx).pomHeightScale = 0.02f;
            textureManager.getMaterial(metalIdx).pomMinLayers = 8.0f;
            textureManager.getMaterial(metalIdx).pomMaxLayers = 32.0f;
            textureManager.getMaterial(metalIdx).specularStrength = 0.9f;
            textureManager.getMaterial(metalIdx).shininess = 128.0f;
            textureManager.getMaterial(metalIdx).ambientFactor = 0.1f;
            loadedIndices.push_back(metalIdx);
            
            // Pixel art - stylized, no parallax
            size_t pixelIdx = textureManager.loadTriple("textures/pixel_color.jpg", "textures/pixel_normal.jpg", "textures/pixel_bump.jpg");
            textureManager.getMaterial(pixelIdx).pomHeightScale = 0.01f;
            textureManager.getMaterial(pixelIdx).pomMinLayers = 4.0f;
            textureManager.getMaterial(pixelIdx).pomMaxLayers = 16.0f;
            textureManager.getMaterial(pixelIdx).specularStrength = 0.3f;
            textureManager.getMaterial(pixelIdx).shininess = 16.0f;
            textureManager.getMaterial(pixelIdx).ambientFactor = 0.15f;
            loadedIndices.push_back(pixelIdx);
            
            // Rock - rough, hard surface with deep crevices
            size_t rockIdx = textureManager.loadTriple("textures/rock_color.jpg", "textures/rock_normal.jpg", "textures/rock_bump.jpg");
            textureManager.getMaterial(rockIdx).pomHeightScale = 0.1f;
            textureManager.getMaterial(rockIdx).pomMinLayers = 12.0f;
            textureManager.getMaterial(rockIdx).pomMaxLayers = 48.0f;
            textureManager.getMaterial(rockIdx).specularStrength = 0.15f;
            textureManager.getMaterial(rockIdx).shininess = 8.0f;
            textureManager.getMaterial(rockIdx).ambientFactor = 0.1f;
            loadedIndices.push_back(rockIdx);
            
            // Sand - fine grain, subtle parallax, slightly reflective
            size_t sandIdx = textureManager.loadTriple("textures/sand_color.jpg", "textures/sand_normal.jpg", "textures/sand_bump.jpg");
            textureManager.getMaterial(sandIdx).pomHeightScale = 0.03f;
            textureManager.getMaterial(sandIdx).pomMinLayers = 6.0f;
            textureManager.getMaterial(sandIdx).pomMaxLayers = 24.0f;
            textureManager.getMaterial(sandIdx).specularStrength = 0.25f;
            textureManager.getMaterial(sandIdx).shininess = 12.0f;
            textureManager.getMaterial(sandIdx).ambientFactor = 0.2f;
            loadedIndices.push_back(sandIdx);
            
            // Snow - soft, bright, diffuse reflection
            size_t snowIdx = textureManager.loadTriple("textures/snow_color.jpg", "textures/snow_normal.jpg", "textures/snow_bump.jpg");
            textureManager.getMaterial(snowIdx).pomHeightScale = 0.04f;
            textureManager.getMaterial(snowIdx).pomMinLayers = 6.0f;
            textureManager.getMaterial(snowIdx).pomMaxLayers = 28.0f;
            textureManager.getMaterial(snowIdx).specularStrength = 0.6f;
            textureManager.getMaterial(snowIdx).shininess = 32.0f;
            textureManager.getMaterial(snowIdx).ambientFactor = 0.3f;
            loadedIndices.push_back(snowIdx);
            
            // Soft sand - very fine, smooth, gentle highlights
            size_t softSandIdx = textureManager.loadTriple("textures/soft_sand_color.jpg", "textures/soft_sand_normal.jpg", "textures/soft_sand_bump.jpg");
            textureManager.getMaterial(softSandIdx).pomHeightScale = 0.025f;
            textureManager.getMaterial(softSandIdx).pomMinLayers = 6.0f;
            textureManager.getMaterial(softSandIdx).pomMaxLayers = 20.0f;
            textureManager.getMaterial(softSandIdx).specularStrength = 0.3f;
            textureManager.getMaterial(softSandIdx).shininess = 16.0f;
            textureManager.getMaterial(softSandIdx).ambientFactor = 0.22f;
            loadedIndices.push_back(softSandIdx);

            // Initialize vegetation texture manager for billboard vegetation
            vegetationTextureManager.init(this);
            
            // Load vegetation textures (albedo/normal/opacity triples)
            // Note: We use the height slot for opacity masks
            size_t foliageIdx = vegetationTextureManager.loadTriple(
                "textures/vegetation/foliage_color.jpg", 
                "textures/vegetation/foliage_normal.jpg", 
                "textures/vegetation/foliage_opacity.jpg"
            );
            vegetationTextureManager.getMaterial(foliageIdx).pomEnabled = false; // No parallax for billboards
            vegetationTextureManager.getMaterial(foliageIdx).specularStrength = 0.1f;
            vegetationTextureManager.getMaterial(foliageIdx).shininess = 4.0f;
            vegetationTextureManager.getMaterial(foliageIdx).ambientFactor = 0.3f;
            
            size_t grassVegIdx = vegetationTextureManager.loadTriple(
                "textures/vegetation/grass_color.jpg", 
                "textures/vegetation/grass_normal.jpg", 
                "textures/vegetation/grass_opacity.jpg"
            );
            vegetationTextureManager.getMaterial(grassVegIdx).pomEnabled = false;
            vegetationTextureManager.getMaterial(grassVegIdx).specularStrength = 0.15f;
            vegetationTextureManager.getMaterial(grassVegIdx).shininess = 6.0f;
            vegetationTextureManager.getMaterial(grassVegIdx).ambientFactor = 0.35f;
            
            size_t wildIdx = vegetationTextureManager.loadTriple(
                "textures/vegetation/wild_color.jpg", 
                "textures/vegetation/wild_normal.jpg", 
                "textures/vegetation/wild_opacity.jpg"
            );
            vegetationTextureManager.getMaterial(wildIdx).pomEnabled = false;
            vegetationTextureManager.getMaterial(wildIdx).specularStrength = 0.12f;
            vegetationTextureManager.getMaterial(wildIdx).shininess = 5.0f;
            vegetationTextureManager.getMaterial(wildIdx).ambientFactor = 0.32f;

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
            editableTextureIndex = editableIndex;  // Store for plane rendering

            // remove any zeros that might come from failed loads (TextureManager may throw or return an index; assume valid indices)
            if (!loadedIndices.empty()) currentTextureIndex = loadedIndices[0];
            
            // create uniform buffers - one per texture (including editable texture)
            uniforms.resize(textureManager.count());
            for (size_t i = 0; i < uniforms.size(); ++i) {
                uniforms[i] = createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            }
            
            // create descriptor sets - one per texture triple
            size_t tripleCount = textureManager.count();
            if (tripleCount == 0) tripleCount = 1; // ensure at least one
            createDescriptorPool(static_cast<uint32_t>(tripleCount));

            descriptorSets.resize(tripleCount, VK_NULL_HANDLE);
            for (size_t i = 0; i < tripleCount; ++i) {
                VkDescriptorSet ds = createDescriptorSet();
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
            }

            // build cube mesh and GPU buffers (per-face tex indices all zero by default)
            auto cube = std::make_unique<CubeMesh>();
            cube->build(this, {});
            
            // build ground plane mesh (20x20 units)
            auto plane = std::make_unique<PlaneMesh>();
            plane->build(this, 20.0f, 20.0f, 0.0f); // Will use editable texture index
            
            // Store meshes for later use
            meshes.push_back(std::move(cube));
            meshes.push_back(std::move(plane));
            
            // Initialize light space matrix BEFORE creating command buffers
            // Light direction points FROM surface TO light (for lighting calculations)
            glm::vec3 lightDirToLight = glm::normalize(lightDirection);
            // Adjust scene center to be between cubes (around y=0) and plane (y=-1.5)
            glm::vec3 sceneCenter = glm::vec3(0.0f, -0.75f, 0.0f);
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
            widgetManager.addWidget(settingsWidget);
            
            // Create light control widget
            auto lightWidget = std::make_shared<LightWidget>(&lightDirection);
            widgetManager.addWidget(lightWidget);
            
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

            // prepare static parts of the UBO (viewPos, light, POM params) - model and mvp will be set per-cube in draw()
            uboStatic.viewPos = glm::vec4(camera.getPosition(), 1.0f);
            // Light direction: pointing upward from surface to light
            glm::vec3 lightDir = glm::normalize(lightDirection);
            uboStatic.lightDir = glm::vec4(lightDir, 0.0f);
            uboStatic.lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
            // Note: pomParams, pomFlags, and specularParams are set per-instance from material properties
            
            // Compute light space matrix for shadow mapping
            // Adjust scene center to be between cubes and plane
            glm::vec3 sceneCenter = glm::vec3(0.0f, -0.75f, 0.0f);
            
            // Diagonal light direction matching setup()
            glm::vec3 shadowLightDir = glm::normalize(lightDirection);
            glm::vec3 lightPos = sceneCenter + shadowLightDir * 20.0f;
            
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
            modelManager.addInstance(plane, planeModel, descriptorSets[editableTextureIndex], &uniforms[editableTextureIndex], planeMat);
            
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
                modelManager.addInstance(cube, model_i, descriptorSets[i], &uniforms[i], mat);
            }
            
            // First pass: Render shadow map
            shadowMapper.beginShadowPass(commandBuffer, uboStatic.lightSpaceMatrix);
            
            // Render all instances to shadow map
            for (const auto& instance : modelManager.getInstances()) {
                // Build POM parameters from instance material
                glm::vec4 pomParams(0.1f, 8.0f, 32.0f, 0.0f); // default: disabled
                glm::vec4 pomFlags(0.0f, 0.0f, 0.0f, 0.0f);
                
                if (instance.material && settingsWidget->getParallaxInShadowPassEnabled()) {
                    pomParams = glm::vec4(
                        instance.material->pomHeightScale,
                        instance.material->pomMinLayers,
                        instance.material->pomMaxLayers,
                        instance.material->pomEnabled
                    );
                    pomFlags = glm::vec4(
                        instance.material->flipNormalY,
                        instance.material->flipTangentHandedness,
                        0.0f, // unused in shadow pass
                        instance.material->flipParallaxDirection
                    );
                }
                
                shadowMapper.renderObject(commandBuffer, instance.transform, instance.model->getVBO(),
                                        instance.descriptorSet, pomParams, pomFlags, camera.getPosition());
            }
            
            shadowMapper.endShadowPass(commandBuffer);
            
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

            // bind pipeline
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            // Render all instances with main pass
            for (const auto& instance : modelManager.getInstances()) {
                const auto& vbo = instance.model->getVBO();
                
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
                
                // Apply shadow effect settings from settings widget
                ubo.shadowEffects = glm::vec4(
                    settingsWidget->getSelfShadowingEnabled() ? 1.0f : 0.0f,
                    settingsWidget->getShadowDisplacementEnabled() ? 1.0f : 0.0f,
                    settingsWidget->getSelfShadowQuality(),
                    0.0f  // unused
                );
                
                updateUniformBuffer(*instance.uniformBuffer, &ubo, sizeof(UniformObject));
                
                // Bind descriptor set and draw
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
            // texture cleanup via manager
            textureManager.destroyAll();
            // vertex/index buffers cleanup - destroy all meshes
            for (auto& mesh : meshes) {
                mesh->destroy(getDevice());
            }
            meshes.clear();

            // uniform buffers cleanup
            for (auto& uniformBuffer : uniforms) {
                if (uniformBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(getDevice(), uniformBuffer.buffer, nullptr);
            }
            for (auto& uniformBuffer : uniforms) {
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

