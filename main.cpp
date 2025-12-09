#include "vulkan/vulkan.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include "vulkan/Camera.hpp"
#include "vulkan/TextureManager.hpp"
#include "vulkan/CubeMesh.hpp"
#include "vulkan/PlaneMesh.hpp"
#include "vulkan/TextureViewer.hpp"
#include <string>
// (removed unused includes: filesystem, iostream, map, algorithm, cctype)

struct UniformObject {
    glm::mat4 mvp;
    glm::mat4 model;
    glm::vec4 viewPos; // world-space camera position
    glm::vec4 lightDir; // xyz = direction, w = unused (padding)
    glm::vec4 lightColor; // rgb = color, w = intensity
    glm::vec4 pomParams; // x=heightScale, y=minLayers, z=maxLayers, w=enabled
    glm::vec4 pomFlags;  // x=flipNormalY, y=flipTangentHandedness, z=ambient, w=unused
    glm::mat4 lightSpaceMatrix; // for shadow mapping
};

class MyApp : public VulkanApp {
    public:
        VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    // texture manager handles albedo/normal/height triples
    TextureManager textureManager;
    // mesh helpers
    CubeMesh cube;
    PlaneMesh plane;
    TextureViewer textureViewer;
    // UI: currently selected texture triple for preview
    size_t currentTextureIndex = 0;
    // Camera
    // start the camera further back so multiple cubes are visible
    Camera camera = Camera(glm::vec3(0.0f, 0.0f, 8.0f));
        // POM / tuning controls
        float pomHeightScale = 0.06f;
        float pomMinLayers = 8.0f;
        float pomMaxLayers = 32.0f;
        bool pomEnabled = true;
        bool flipNormalY = false;
        bool flipTangentHandedness = false;
    bool flipParallaxDirection = false; // false = dark goes inward (standard), true = dark goes outward
        float ambientFactor = 0.4f; // Increased from 0.25 for brighter base lighting
    // (managed by TextureManager)
        std::vector<VkDescriptorSet> descriptorSets;
        std::vector<Buffer> uniforms; // One uniform buffer per cube
        VkDescriptorSet planeDescriptorSet = VK_NULL_HANDLE;
        Buffer planeUniform;
        
        // Shadow mapping resources
        const uint32_t SHADOW_MAP_SIZE = 2048;
        VkImage shadowMapImage = VK_NULL_HANDLE;
        VkDeviceMemory shadowMapMemory = VK_NULL_HANDLE;
        VkImageView shadowMapView = VK_NULL_HANDLE;
        VkSampler shadowMapSampler = VK_NULL_HANDLE;
        VkFramebuffer shadowFramebuffer = VK_NULL_HANDLE;
        VkRenderPass shadowRenderPass = VK_NULL_HANDLE;
        VkPipeline shadowPipeline = VK_NULL_HANDLE;
        VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout shadowDescSetLayout = VK_NULL_HANDLE;
        Buffer lightSpaceUniform; // Light-space matrix for shadow pass
        VkDescriptorSet shadowMapImGuiDescSet = VK_NULL_HANDLE; // For ImGui display
        // store projection and view for per-cube MVP computation in draw()
        glm::mat4 projMat = glm::mat4(1.0f);
        glm::mat4 viewMat = glm::mat4(1.0f);
        // static parts of the UBO that don't vary per-cube (we'll set model/mvp per draw)
        UniformObject uboStatic{};
    // textureImage is now owned by TextureManager
        //VertexBufferObject vertexBufferObject; // now owned by CubeMesh

        void setup() override {
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

            // Explicit per-name loads (one-by-one). Adjust list below if you add/remove texture files.
            loadedIndices.push_back(textureManager.loadTriple("textures/bricks_color.jpg", "textures/bricks_normal.jpg", "textures/bricks_bump.jpg"));
            loadedIndices.push_back(textureManager.loadTriple("textures/dirt_color.jpg", "textures/dirt_normal.jpg", "textures/dirt_bump.jpg"));
            loadedIndices.push_back(textureManager.loadTriple("textures/forest_color.jpg", "textures/forest_normal.jpg", "textures/forest_bump.jpg"));
            loadedIndices.push_back(textureManager.loadTriple("textures/grass_color.jpg", "textures/grass_normal.jpg", "textures/grass_bump.jpg"));
            loadedIndices.push_back(textureManager.loadTriple("textures/lava_color.jpg", "textures/lava_normal.jpg", "textures/lava_bump.jpg"));
            loadedIndices.push_back(textureManager.loadTriple("textures/metal_color.jpg", "textures/metal_normal.jpg", "textures/metal_bump.jpg"));
            loadedIndices.push_back(textureManager.loadTriple("textures/pixel_color.jpg", "textures/pixel_normal.jpg", "textures/pixel_bump.jpg"));
            loadedIndices.push_back(textureManager.loadTriple("textures/rock_color.jpg", "textures/rock_normal.jpg", "textures/rock_bump.jpg"));
            loadedIndices.push_back(textureManager.loadTriple("textures/sand_color.jpg", "textures/sand_normal.jpg", "textures/sand_bump.jpg"));
            loadedIndices.push_back(textureManager.loadTriple("textures/snow_color.jpg", "textures/snow_normal.jpg", "textures/snow_bump.jpg"));
            loadedIndices.push_back(textureManager.loadTriple("textures/soft_sand_color.jpg", "textures/soft_sand_normal.jpg", "textures/soft_sand_bump.jpg"));

            // remove any zeros that might come from failed loads (TextureManager may throw or return an index; assume valid indices)
            if (!loadedIndices.empty()) currentTextureIndex = loadedIndices[0];
            
            // create uniform buffers - one per cube
            uniforms.resize(loadedIndices.size());
            for (size_t i = 0; i < uniforms.size(); ++i) {
                uniforms[i] = createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            }
            
            // Create shadow map resources
            createShadowMap();
            
            // Create shadow pipeline
            {
                ShaderStage shadowVertexShader = ShaderStage(
                    createShaderModule(FileReader::readFile("shaders/shadow.vert.spv")),
                    VK_SHADER_STAGE_VERTEX_BIT
                );

                ShaderStage shadowFragmentShader = ShaderStage(
                    createShaderModule(FileReader::readFile("shaders/shadow.frag.spv")),
                    VK_SHADER_STAGE_FRAGMENT_BIT
                );

                // Use the same descriptor set layout and pipeline layout as the main pipeline
                // but create a pipeline compatible with the shadow render pass
                VkGraphicsPipelineCreateInfo pipelineInfo{};
                pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                
                VkPipelineShaderStageCreateInfo shaderStages[] = { shadowVertexShader.info, shadowFragmentShader.info };
                pipelineInfo.stageCount = 2;
                pipelineInfo.pStages = shaderStages;
                
                // Vertex input
                VkVertexInputBindingDescription bindingDescription{};
                bindingDescription.binding = 0;
                bindingDescription.stride = sizeof(Vertex);
                bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
                
                std::vector<VkVertexInputAttributeDescription> attributeDescriptions = {
                    VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
                    VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                    VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) },
                    VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                    VkVertexInputAttributeDescription { 4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent) },
                    VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SFLOAT, offsetof(Vertex, texIndex) }
                };
                
                VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
                vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
                vertexInputInfo.vertexBindingDescriptionCount = 1;
                vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
                vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
                vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
                pipelineInfo.pVertexInputState = &vertexInputInfo;
                
                // Input assembly
                VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
                inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
                inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
                inputAssembly.primitiveRestartEnable = VK_FALSE;
                pipelineInfo.pInputAssemblyState = &inputAssembly;
                
                // Viewport state
                VkPipelineViewportStateCreateInfo viewportState{};
                viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
                viewportState.viewportCount = 1;
                viewportState.scissorCount = 1;
                pipelineInfo.pViewportState = &viewportState;
                
                // Rasterization
                VkPipelineRasterizationStateCreateInfo rasterizer{};
                rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                rasterizer.depthClampEnable = VK_TRUE; // Enable depth clamp to prevent far plane clipping
                rasterizer.rasterizerDiscardEnable = VK_FALSE;
                rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
                rasterizer.lineWidth = 1.0f;
                rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT; // Cull front faces for shadow mapping (Peter Panning prevention)
                rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                rasterizer.depthBiasEnable = VK_TRUE; // Enable depth bias for shadow mapping
                pipelineInfo.pRasterizationState = &rasterizer;
                
                // Multisample
                VkPipelineMultisampleStateCreateInfo multisampling{};
                multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
                multisampling.sampleShadingEnable = VK_FALSE;
                multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
                pipelineInfo.pMultisampleState = &multisampling;
                
                // Depth stencil
                VkPipelineDepthStencilStateCreateInfo depthStencil{};
                depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
                depthStencil.depthTestEnable = VK_TRUE;
                depthStencil.depthWriteEnable = VK_TRUE;
                depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
                depthStencil.depthBoundsTestEnable = VK_FALSE;
                depthStencil.stencilTestEnable = VK_FALSE;
                pipelineInfo.pDepthStencilState = &depthStencil;
                
                // Color blend - no color attachments for shadow pass
                VkPipelineColorBlendStateCreateInfo colorBlending{};
                colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
                colorBlending.logicOpEnable = VK_FALSE;
                colorBlending.attachmentCount = 0; // No color attachments
                pipelineInfo.pColorBlendState = &colorBlending;
                
                // Dynamic state
                VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
                VkPipelineDynamicStateCreateInfo dynamicState{};
                dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
                dynamicState.dynamicStateCount = 3;
                dynamicState.pDynamicStates = dynamicStates;
                pipelineInfo.pDynamicState = &dynamicState;
                
                // Create separate pipeline layout for shadow pass with push constants
                VkPushConstantRange pushConstantRange{};
                pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
                pushConstantRange.offset = 0;
                pushConstantRange.size = sizeof(glm::mat4); // Size of MVP matrix
                
                VkPipelineLayoutCreateInfo shadowLayoutInfo{};
                shadowLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                shadowLayoutInfo.setLayoutCount = 0; // No descriptor sets needed for shadow pass
                shadowLayoutInfo.pSetLayouts = nullptr;
                shadowLayoutInfo.pushConstantRangeCount = 1;
                shadowLayoutInfo.pPushConstantRanges = &pushConstantRange;
                
                if (vkCreatePipelineLayout(getDevice(), &shadowLayoutInfo, nullptr, &shadowPipelineLayout) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create shadow pipeline layout!");
                }
                
                pipelineInfo.layout = shadowPipelineLayout;
                pipelineInfo.renderPass = shadowRenderPass;
                pipelineInfo.subpass = 0;
                
                if (vkCreateGraphicsPipelines(getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowPipeline) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create shadow pipeline!");
                }
                
                vkDestroyShaderModule(getDevice(), shadowFragmentShader.info.module, nullptr);
                vkDestroyShaderModule(getDevice(), shadowVertexShader.info.module, nullptr);
            }
            
            // create descriptor sets - one per cube plus one for the plane
            size_t tripleCount = textureManager.count();
            if (tripleCount == 0) tripleCount = 1; // ensure at least one
            createDescriptorPool(static_cast<uint32_t>(tripleCount + 1)); // +1 for the plane

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
                VkDescriptorImageInfo shadowInfo { shadowMapSampler, shadowMapView, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
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
            cube.build(this, {});
            
            // build ground plane mesh (20x20 units, textured with fourth texture)
            plane.build(this, 20.0f, 20.0f, 3.0f); // Using texture index 3
            
            // create uniform buffer for the plane
            planeUniform = createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            
            // create descriptor set for the plane (use first texture triple)
            planeDescriptorSet = createDescriptorSet();
            VkDescriptorBufferInfo planeBufferInfo { planeUniform.buffer, 0, sizeof(UniformObject) };
            
            VkWriteDescriptorSet planeUboWrite{};
            planeUboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            planeUboWrite.dstSet = planeDescriptorSet;
            planeUboWrite.dstBinding = 0;
            planeUboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            planeUboWrite.descriptorCount = 1;
            planeUboWrite.pBufferInfo = &planeBufferInfo;
            
            // Use texture index 3 for the plane (fourth texture)
            const auto& planeTriple = textureManager.getTriple(3);
            
            VkDescriptorImageInfo planeAlbedoInfo{ planeTriple.albedoSampler, planeTriple.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo planeNormalInfo{ planeTriple.normalSampler, planeTriple.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo planeHeightInfo{ planeTriple.heightSampler, planeTriple.height.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            
            VkWriteDescriptorSet planeAlbedoWrite{};
            planeAlbedoWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            planeAlbedoWrite.dstSet = planeDescriptorSet;
            planeAlbedoWrite.dstBinding = 1;
            planeAlbedoWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            planeAlbedoWrite.descriptorCount = 1;
            planeAlbedoWrite.pImageInfo = &planeAlbedoInfo;
            
            VkWriteDescriptorSet planeNormalWrite{};
            planeNormalWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            planeNormalWrite.dstSet = planeDescriptorSet;
            planeNormalWrite.dstBinding = 2;
            planeNormalWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            planeNormalWrite.descriptorCount = 1;
            planeNormalWrite.pImageInfo = &planeNormalInfo;
            
            VkWriteDescriptorSet planeHeightWrite{};
            planeHeightWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            planeHeightWrite.dstSet = planeDescriptorSet;
            planeHeightWrite.dstBinding = 3;
            planeHeightWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            planeHeightWrite.descriptorCount = 1;
            planeHeightWrite.pImageInfo = &planeHeightInfo;
            
            // Shadow map descriptor for plane (binding 4)
            VkDescriptorImageInfo planeShadowInfo { shadowMapSampler, shadowMapView, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet planeShadowWrite{};
            planeShadowWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            planeShadowWrite.dstSet = planeDescriptorSet;
            planeShadowWrite.dstBinding = 4;
            planeShadowWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            planeShadowWrite.descriptorCount = 1;
            planeShadowWrite.pImageInfo = &planeShadowInfo;
            
            updateDescriptorSet(planeDescriptorSet, { planeUboWrite, planeAlbedoWrite, planeNormalWrite, planeHeightWrite, planeShadowWrite });
            
            // Initialize light space matrix BEFORE creating command buffers
            // Light direction points FROM surface TO light (for lighting calculations)
            glm::vec3 lightDirToLight = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));
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
            
            // initialize texture viewer (after textures loaded)
            textureViewer.init(&textureManager);
            createCommandBuffers();
        };

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
                ImGui::SameLine(ImGui::GetIO().DisplaySize.x - 120);
                ImGui::Text("FPS: %.1f", imguiFps);
                ImGui::EndMainMenuBar();
            }

            if (imguiShowDemo) ImGui::ShowDemoWindow(&imguiShowDemo);

            // POM controls
            ImGui::Begin("POM Controls");
            ImGui::Checkbox("Enable POM", &pomEnabled);
            ImGui::SliderFloat("Height Scale", &pomHeightScale, 0.0f, 0.2f, "%.3f");
            ImGui::SliderFloat("Min Layers", &pomMinLayers, 1.0f, 64.0f, "%.0f");
            ImGui::SliderFloat("Max Layers", &pomMaxLayers, 1.0f, 128.0f, "%.0f");
            ImGui::Checkbox("Flip normal Y", &flipNormalY);
            ImGui::Checkbox("Flip parallax direction", &flipParallaxDirection);
            ImGui::Checkbox("Flip tangent handedness", &flipTangentHandedness);
            ImGui::SliderFloat("Ambient", &ambientFactor, 0.0f, 1.0f, "%.2f");
            ImGui::End();

            // Camera controls
            ImGui::Begin("Camera");
            ImGui::DragFloat("Move Speed", &camera.speed, 0.1f, 0.0f, 50.0f);
            float angDeg = glm::degrees(camera.angularSpeedRad);
            if (ImGui::SliderFloat("Angular Speed (deg/s)", &angDeg, 1.0f, 360.0f)) {
                camera.angularSpeedRad = glm::radians(angDeg);
            }
            glm::vec3 pos = camera.getPosition();
            ImGui::Text("Position: %.2f, %.2f, %.2f", pos.x, pos.y, pos.z);
            glm::quat orient = camera.getOrientation();
            glm::vec3 euler = glm::degrees(glm::eulerAngles(orient));
            ImGui::Text("Euler (deg): X=%.1f Y=%.1f Z=%.1f", euler.x, euler.y, euler.z);
            if (ImGui::Button("Reset Orientation")) {
                // small helper: recreate camera to reset orientation (preserve position)
                camera = Camera(camera.getPosition());
            }
            ImGui::End();

            // Textures visualization (single preview with navigation)
            textureViewer.render();

            // Debug panel: show scene info
            ImGui::Begin("Debug");
            ImGui::Text("Loaded texture triples: %zu", textureManager.count());
            ImGui::Text("Rendered cubes: %zu", descriptorSets.size());
            glm::vec3 camPos = camera.getPosition();
            ImGui::Text("Camera pos: %.2f %.2f %.2f", camPos.x, camPos.y, camPos.z);
            ImGui::Text("Cube grid spacing: %.1f", 2.5f);
            ImGui::Text("Grid layout: 4x3 cubes");
            ImGui::End();
            
            // Shadow Map Viewer
            ImGui::Begin("Shadow Map");
            ImGui::Text("Shadow Map Size: %dx%d", SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
            
            // Debug: show light direction and position
            glm::vec3 lightDirVec = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));
            glm::vec3 sceneCenter = glm::vec3(0.0f, 0.0f, 0.0f);
            glm::vec3 lightPosVec = sceneCenter + lightDirVec * 20.0f;
            ImGui::Text("Light Dir: %.2f, %.2f, %.2f", lightDirVec.x, lightDirVec.y, lightDirVec.z);
            ImGui::Text("Light Pos: %.2f, %.2f, %.2f", lightPosVec.x, lightPosVec.y, lightPosVec.z);
            ImGui::Text("Scene Center: %.2f, %.2f, %.2f", sceneCenter.x, sceneCenter.y, sceneCenter.z);
            
            // Display shadow map as a texture
            if (shadowMapImGuiDescSet != VK_NULL_HANDLE) {
                ImGui::Image((ImTextureID)shadowMapImGuiDescSet, ImVec2(256, 256));
            } else {
                ImGui::Text("Shadow map not available");
            }
            ImGui::End();
        }

        void update(float deltaTime) override {
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
            if (win) camera.processInput(win, deltaTime);

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
            glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));
            uboStatic.lightDir = glm::vec4(lightDir, 0.0f);
            uboStatic.lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
            uboStatic.pomParams = glm::vec4(pomHeightScale, pomMinLayers, pomMaxLayers, pomEnabled ? 1.0f : 0.0f);
            uboStatic.pomFlags = glm::vec4(flipNormalY ? 1.0f : 0.0f, flipTangentHandedness ? 1.0f : 0.0f, ambientFactor, flipParallaxDirection ? 1.0f : 0.0f);
            
            // Compute light space matrix for shadow mapping
            // Adjust scene center to be between cubes and plane
            glm::vec3 sceneCenter = glm::vec3(0.0f, -0.75f, 0.0f);
            
            // Diagonal light direction matching setup()
            glm::vec3 shadowLightDir = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));
            glm::vec3 lightPos = sceneCenter + shadowLightDir * 20.0f;
            
            glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
            if (glm::abs(glm::dot(shadowLightDir, worldUp)) > 0.9f) {
                worldUp = glm::vec3(1.0f, 0.0f, 0.0f);
            }
            
            glm::mat4 lightView = glm::lookAt(lightPos, sceneCenter, worldUp);
            
            float orthoSize = 15.0f;
            glm::mat4 lightProjection = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 5.0f, 40.0f);
            
            uboStatic.lightSpaceMatrix = lightProjection * lightView;
        };

        void draw(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo) override {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);
            
            // Calculate grid layout for cubes
            size_t n = descriptorSets.size();
            float spacing = 2.5f;
            if (n == 0) n = 1;
            int cols = static_cast<int>(std::ceil(std::sqrt((float)n)));
            int rows = static_cast<int>(std::ceil((float)n / cols));
            float halfW = (cols - 1) * 0.5f * spacing;
            float halfH = (rows - 1) * 0.5f * spacing;
            
            // First pass: Render shadow map
            renderShadowPass(commandBuffer);
            
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

            // bind pipeline and draw the indexed square
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);


            VkBuffer vertexBuffers[] = { cube.getVBO().vertexBuffer.buffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, cube.getVBO().indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
            // Draw one cube per loaded texture triple laid out in a centered grid
            // (grid calculation already done at start of draw())
            // Render ground plane first (underneath cubes)
            glm::mat4 planeModel = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.5f, 0.0f)); // Position below cubes
            glm::mat4 planeMVP = projMat * viewMat * planeModel;
            
            UniformObject planeUBO = uboStatic;
            planeUBO.model = planeModel;
            planeUBO.mvp = planeMVP;
            // Ensure plane UBO has the light space matrix from uboStatic
            planeUBO.lightSpaceMatrix = uboStatic.lightSpaceMatrix;
            updateUniformBuffer(planeUniform, &planeUBO, sizeof(UniformObject));
            
            // Bind plane vertex/index buffers
            VkBuffer planeVertexBuffers[] = { plane.getVBO().vertexBuffer.buffer };
            VkDeviceSize planeOffsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, planeVertexBuffers, planeOffsets);
            vkCmdBindIndexBuffer(commandBuffer, plane.getVBO().indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
            
            // Bind plane descriptor set and draw
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipelineLayout(), 0, 1, &planeDescriptorSet, 0, nullptr);
            vkCmdDrawIndexed(commandBuffer, plane.getVBO().indexCount, 1, 0, 0, 0);
            
            // Rebind cube vertex/index buffers
            VkBuffer cubeVertexBuffers[] = { cube.getVBO().vertexBuffer.buffer };
            VkDeviceSize cubeOffsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, cubeVertexBuffers, cubeOffsets);
            vkCmdBindIndexBuffer(commandBuffer, cube.getVBO().indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
            
            // Render all cubes with separate uniform buffers
            for (size_t i = 0; i < descriptorSets.size(); ++i) {
                int col = static_cast<int>(i % cols);
                int row = static_cast<int>(i / cols);
                
                // Position cubes horizontally above the plane (plane at Y=-1.5)
                float x = col * spacing - halfW;
                float y = -1.0f; // Above the plane (plane is at -1.5)
                float z = row * spacing - halfH; // Spread in Z direction
                glm::mat4 model_i = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
                glm::mat4 mvp_i = projMat * viewMat * model_i;

                UniformObject ubo = uboStatic;
                ubo.model = model_i;
                ubo.mvp = mvp_i;
                updateUniformBuffer(uniforms[i], &ubo, sizeof(UniformObject));

                // bind descriptor set for this cube's textures
                VkDescriptorSet ds = descriptorSets[i];
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipelineLayout(), 0, 1, &ds, 0, nullptr);

                vkCmdDrawIndexed(commandBuffer, cube.getVBO().indexCount, 1, 0, 0, 0);
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
            if (graphicsPipeline != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), graphicsPipeline, nullptr);
            // texture cleanup via manager
            textureManager.destroyAll();
            // vertex/index buffers cleanup
            cube.destroy(getDevice());
            plane.cleanup(this);

            // uniform buffers cleanup
            for (auto& uniformBuffer : uniforms) {
                if (uniformBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(getDevice(), uniformBuffer.buffer, nullptr);
            }
            for (auto& uniformBuffer : uniforms) {
                if (uniformBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), uniformBuffer.memory, nullptr);
            }
            
            // plane uniform buffer cleanup
            if (planeUniform.buffer != VK_NULL_HANDLE) vkDestroyBuffer(getDevice(), planeUniform.buffer, nullptr);
            if (planeUniform.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), planeUniform.memory, nullptr);
            
            // shadow map cleanup
            cleanupShadowMap();
        }
        
    private:
        void createShadowMap() {
            // Create depth image for shadow map
            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.extent.width = SHADOW_MAP_SIZE;
            imageInfo.extent.height = SHADOW_MAP_SIZE;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = VK_FORMAT_D32_SFLOAT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            
            if (vkCreateImage(getDevice(), &imageInfo, nullptr, &shadowMapImage) != VK_SUCCESS) {
                throw std::runtime_error("failed to create shadow map image!");
            }
            
            VkMemoryRequirements memRequirements;
            vkGetImageMemoryRequirements(getDevice(), shadowMapImage, &memRequirements);
            
            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            
            if (vkAllocateMemory(getDevice(), &allocInfo, nullptr, &shadowMapMemory) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate shadow map memory!");
            }
            
            vkBindImageMemory(getDevice(), shadowMapImage, shadowMapMemory, 0);
            
            // Create image view
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = shadowMapImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_D32_SFLOAT;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;
            
            if (vkCreateImageView(getDevice(), &viewInfo, nullptr, &shadowMapView) != VK_SUCCESS) {
                throw std::runtime_error("failed to create shadow map image view!");
            }
            
            // Create sampler for shadow map with comparison
            VkSamplerCreateInfo samplerInfo{};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_NEAREST;
            samplerInfo.minFilter = VK_FILTER_NEAREST;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.anisotropyEnable = VK_FALSE;
            samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            samplerInfo.unnormalizedCoordinates = VK_FALSE;
            samplerInfo.compareEnable = VK_FALSE; // Can enable for hardware PCF
            samplerInfo.compareOp = VK_COMPARE_OP_LESS;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            
            if (vkCreateSampler(getDevice(), &samplerInfo, nullptr, &shadowMapSampler) != VK_SUCCESS) {
                throw std::runtime_error("failed to create shadow map sampler!");
            }
            
            // Create ImGui descriptor set for shadow map visualization
            shadowMapImGuiDescSet = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
                shadowMapSampler, 
                shadowMapView, 
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
            );
            
            // Create shadow render pass
            createShadowRenderPass();
            
            // Create shadow framebuffer
            createShadowFramebuffer();
        }
        
        void createShadowRenderPass() {
            VkAttachmentDescription depthAttachment{};
            depthAttachment.format = VK_FORMAT_D32_SFLOAT;
            depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            
            VkAttachmentReference depthAttachmentRef{};
            depthAttachmentRef.attachment = 0;
            depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 0;
            subpass.pDepthStencilAttachment = &depthAttachmentRef;
            
            VkSubpassDependency dependency{};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;
            dependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            
            VkRenderPassCreateInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = 1;
            renderPassInfo.pAttachments = &depthAttachment;
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
            renderPassInfo.dependencyCount = 1;
            renderPassInfo.pDependencies = &dependency;
            
            if (vkCreateRenderPass(getDevice(), &renderPassInfo, nullptr, &shadowRenderPass) != VK_SUCCESS) {
                throw std::runtime_error("failed to create shadow render pass!");
            }
        }
        
        void createShadowFramebuffer() {
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = shadowRenderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &shadowMapView;
            framebufferInfo.width = SHADOW_MAP_SIZE;
            framebufferInfo.height = SHADOW_MAP_SIZE;
            framebufferInfo.layers = 1;
            
            if (vkCreateFramebuffer(getDevice(), &framebufferInfo, nullptr, &shadowFramebuffer) != VK_SUCCESS) {
                throw std::runtime_error("failed to create shadow framebuffer!");
            }
        }
        
        void cleanupShadowMap() {
            if (shadowMapImGuiDescSet != VK_NULL_HANDLE) {
                ImGui_ImplVulkan_RemoveTexture(shadowMapImGuiDescSet);
                shadowMapImGuiDescSet = VK_NULL_HANDLE;
            }
            if (shadowPipeline != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), shadowPipeline, nullptr);
            if (shadowFramebuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(getDevice(), shadowFramebuffer, nullptr);
            if (shadowRenderPass != VK_NULL_HANDLE) vkDestroyRenderPass(getDevice(), shadowRenderPass, nullptr);
            if (shadowMapSampler != VK_NULL_HANDLE) vkDestroySampler(getDevice(), shadowMapSampler, nullptr);
            if (shadowMapView != VK_NULL_HANDLE) vkDestroyImageView(getDevice(), shadowMapView, nullptr);
            if (shadowMapImage != VK_NULL_HANDLE) vkDestroyImage(getDevice(), shadowMapImage, nullptr);
            if (shadowMapMemory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), shadowMapMemory, nullptr);
        }
        
        void renderShadowPass(VkCommandBuffer commandBuffer) {
            // Debug: print what lightSpaceMatrix we're using in shadow pass
            static int shadowDebugCount = 0;
            if (shadowDebugCount == 0) {
                printf("\n=== SHADOW PASS DEBUG ===\n");
                glm::mat4 ls = uboStatic.lightSpaceMatrix;
                printf("lightSpaceMatrix being used:\n");
                for(int row=0; row<4; row++) {
                    printf("  [%.3f, %.3f, %.3f, %.3f]\n", ls[0][row], ls[1][row], ls[2][row], ls[3][row]);
                }
                shadowDebugCount++;
            }
            
            // Begin shadow render pass
            VkRenderPassBeginInfo shadowRenderPassInfo{};
            shadowRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            shadowRenderPassInfo.renderPass = shadowRenderPass;
            shadowRenderPassInfo.framebuffer = shadowFramebuffer;
            shadowRenderPassInfo.renderArea.offset = {0, 0};
            shadowRenderPassInfo.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
            
            VkClearValue clearDepth;
            clearDepth.depthStencil = {1.0f, 0};
            shadowRenderPassInfo.clearValueCount = 1;
            shadowRenderPassInfo.pClearValues = &clearDepth;
            
            vkCmdBeginRenderPass(commandBuffer, &shadowRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
            
            // Set viewport and scissor for shadow map
            VkViewport shadowViewport{};
            shadowViewport.x = 0.0f;
            shadowViewport.y = 0.0f;
            shadowViewport.width = (float)SHADOW_MAP_SIZE;
            shadowViewport.height = (float)SHADOW_MAP_SIZE;
            shadowViewport.minDepth = 0.0f;
            shadowViewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &shadowViewport);
            
            VkRect2D shadowScissor{};
            shadowScissor.offset = {0, 0};
            shadowScissor.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
            vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);
            
            // Set depth bias to reduce shadow acne (reduced values for better shadow visibility)
            vkCmdSetDepthBias(commandBuffer, 0.5f, 0.0f, 1.0f);
            
            // Bind the shadow pipeline
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
            
            // Don't render the plane to shadow map - we only want cube shadows on the plane
            // Skip plane rendering in shadow pass
            
            // Render all cubes to shadow map
            size_t n = descriptorSets.size();
            float spacing = 2.5f;
            if (n == 0) n = 1;
            int cols = static_cast<int>(std::ceil(std::sqrt((float)n)));
            int rows = static_cast<int>(std::ceil((float)n / cols));
            float halfW = (cols - 1) * 0.5f * spacing;
            float halfH = (rows - 1) * 0.5f * spacing;
            
            // Debug: Print once
            static bool printed = false;
            if (!printed) {
                printf("Shadow pass: rendering %zu cubes in %dx%d grid\n", descriptorSets.size(), cols, rows);
                printf("Grid dimensions: halfW=%.2f, halfH=%.2f, spacing=%.2f\n", halfW, halfH, spacing);
                printed = true;
            }
            
            VkBuffer cubeVertexBuffers[] = { cube.getVBO().vertexBuffer.buffer };
            VkDeviceSize cubeOffsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, cubeVertexBuffers, cubeOffsets);
            vkCmdBindIndexBuffer(commandBuffer, cube.getVBO().indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
            
            for (size_t i = 0; i < descriptorSets.size(); ++i) {
                int col = static_cast<int>(i % cols);
                int row = static_cast<int>(i / cols);
                
                // Position cubes horizontally above the plane (plane at Y=-1.5)
                float x = col * spacing - halfW;
                float y = -1.0f; // Above the plane (plane is at -1.5)
                float z = row * spacing - halfH; // Spread in Z direction
                
                // Skip shadow rendering for cubes below the plane
                if (y < -1.5f) {
                    continue;
                }
                
                glm::mat4 model_i = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
                glm::mat4 mvp_i = uboStatic.lightSpaceMatrix * model_i;
                
                // Debug: Print ALL cube positions to see the full grid
                static bool printedPos = false;
                if (!printedPos) {
                    printf("  Cube %zu: pos(%.2f, %.2f, %.2f)\n", i, x, y, z);
                    if (i == descriptorSets.size() - 1) printedPos = true;
                }

                // Push MVP matrix directly to the shader
                vkCmdPushConstants(commandBuffer, shadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &mvp_i);
                
                vkCmdDrawIndexed(commandBuffer, cube.getVBO().indexCount, 1, 0, 0, 0);
            }
            
            vkCmdEndRenderPass(commandBuffer);
            
            // Transition shadow map from depth attachment to shader read
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = shadowMapImage;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            
            vkCmdPipelineBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
            );
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

