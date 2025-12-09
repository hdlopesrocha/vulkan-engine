#include "vulkan/vulkan.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

struct UniformObject {
    glm::mat4 mvp;
    glm::mat4 model;
    glm::vec4 viewPos; // world-space camera position
    glm::vec4 lightDir; // xyz = direction, w = unused (padding)
    glm::vec4 lightColor; // rgb = color, w = intensity
    glm::vec4 pomParams; // x=heightScale, y=minLayers, z=maxLayers, w=enabled
    glm::vec4 pomFlags;  // x=flipNormalY, y=flipTangentHandedness, z=ambient, w=unused
};

class MyApp : public VulkanApp {
    public:
        VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkSampler textureSampler = VK_NULL_HANDLE;
    // Camera state for WASD/QE movement
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 2.5f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    float cameraSpeed = 2.5f; // units per second
    // camera orientation (quaternion) for rotation via RTYFGH
    glm::quat cameraQuat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    float cameraAngularSpeed = glm::radians(45.0f); // degrees per second converted to radians
        // POM / tuning controls
        float pomHeightScale = 0.06f;
        float pomMinLayers = 8.0f;
        float pomMaxLayers = 32.0f;
        bool pomEnabled = true;
        bool flipNormalY = false;
        bool flipTangentHandedness = false;
        float ambientFactor = 0.25f;
    TextureImage normalImage;
    VkSampler normalSampler = VK_NULL_HANDLE;
    TextureImage heightImage;
    VkSampler heightSampler = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        Buffer uniform;
        TextureImage textureImage;
        VertexBufferObject vertexBufferObject;

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
                        VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, uv) },
                        VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                        VkVertexInputAttributeDescription { 4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent) }
                    }
                );

                vkDestroyShaderModule(getDevice(), fragmentShader.info.module, nullptr);
                vkDestroyShaderModule(getDevice(), vertexShader.info.module, nullptr);
            }
            textureImage = createTextureImage("textures/grass_color.jpg");
            textureSampler = createTextureSampler(textureImage.mipLevels);
            normalImage = createTextureImage("textures/grass_normal.jpg");
            normalSampler = createTextureSampler(normalImage.mipLevels);
            heightImage = createTextureImage("textures/grass_bump.jpg");
            heightSampler = createTextureSampler(heightImage.mipLevels);
            uniform = createBuffer(sizeof(UniformObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            createDescriptorPool();
            // Descriptor Set
            {
                descriptorSet = createDescriptorSet();
            
                // uniform buffer descriptor (binding 0)
                VkDescriptorBufferInfo bufferInfo { uniform.buffer, 0 , sizeof(UniformObject) };

                VkWriteDescriptorSet uboWrite{};
                uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                uboWrite.dstSet = descriptorSet;
                uboWrite.dstBinding = 0;
                uboWrite.dstArrayElement = 0;
                uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                uboWrite.descriptorCount = 1;
                uboWrite.pBufferInfo = &bufferInfo;

                // image sampler descriptor (binding 1)
                VkDescriptorImageInfo imageInfo { textureSampler, textureImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

                VkWriteDescriptorSet samplerWrite{};
                samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                samplerWrite.dstSet = descriptorSet;
                samplerWrite.dstBinding = 1;
                samplerWrite.dstArrayElement = 0;
                samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                samplerWrite.descriptorCount = 1;
                samplerWrite.pImageInfo = &imageInfo;

                // normal map descriptor (binding 2)
                VkDescriptorImageInfo normalInfo { normalSampler, normalImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                VkWriteDescriptorSet normalWrite{};
                normalWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                normalWrite.dstSet = descriptorSet;
                normalWrite.dstBinding = 2;
                normalWrite.dstArrayElement = 0;
                normalWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                normalWrite.descriptorCount = 1;
                normalWrite.pImageInfo = &normalInfo;

                // height/bump map descriptor (binding 3)
                VkDescriptorImageInfo heightInfo { heightSampler, heightImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                VkWriteDescriptorSet heightWrite{};
                heightWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                heightWrite.dstSet = descriptorSet;
                heightWrite.dstBinding = 3;
                heightWrite.dstArrayElement = 0;
                heightWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                heightWrite.descriptorCount = 1;
                heightWrite.pImageInfo = &heightInfo;

                updateDescriptorSet(
                    descriptorSet,
                    { uboWrite, samplerWrite, normalWrite, heightWrite }
                );
            }

            // 24 unique vertices (4 per face) so each face can have its own UVs
            std::vector<Vertex> vertices = {
                // +X face (normal +1,0,0) tangent computed later
                {{ 0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f,0.0f}, {1,0,0}, {0,0,0}},
                {{ 0.5f, -0.5f,  0.5f }, {1,1,1}, {1.0f,0.0f}, {1,0,0}, {0,0,0}},
                {{ 0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f,1.0f}, {1,0,0}, {0,0,0}},
                {{ 0.5f,  0.5f, -0.5f }, {1,1,1}, {0.0f,1.0f}, {1,0,0}, {0,0,0}},
                // -X face (normal -1,0,0) tangent along -Z
                {{-0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f,0.0f}, {-1,0,0}, {0,0,0}},
                {{-0.5f, -0.5f, -0.5f }, {1,1,1}, {1.0f,0.0f}, {-1,0,0}, {0,0,0}},
                {{-0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f,1.0f}, {-1,0,0}, {0,0,0}},
                {{-0.5f,  0.5f,  0.5f }, {1,1,1}, {0.0f,1.0f}, {-1,0,0}, {0,0,0}},
                // +Y face (normal 0,1,0) tangent along +X
                {{-0.5f,  0.5f, -0.5f }, {1,1,1}, {0.0f,0.0f}, {0,1,0}, {0,0,0}},
                {{ 0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f,0.0f}, {0,1,0}, {0,0,0}},
                {{ 0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f,1.0f}, {0,1,0}, {0,0,0}},
                {{-0.5f,  0.5f,  0.5f }, {1,1,1}, {0.0f,1.0f}, {0,1,0}, {0,0,0}},
                // -Y face (normal 0,-1,0) tangent along +X
                {{-0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f,0.0f}, {0,-1,0}, {0,0,0}},
                {{ 0.5f, -0.5f,  0.5f }, {1,1,1}, {1.0f,0.0f}, {0,-1,0}, {0,0,0}},
                {{ 0.5f, -0.5f, -0.5f }, {1,1,1}, {1.0f,1.0f}, {0,-1,0}, {0,0,0}},
                {{-0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f,1.0f}, {0,-1,0}, {0,0,0}},
                // +Z face (normal 0,0,1) tangent along +X
                {{-0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f,0.0f}, {0,0,1}, {0,0,0}},
                {{-0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f,0.0f}, {0,0,1}, {0,0,0}},
                {{ 0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f,1.0f}, {0,0,1}, {0,0,0}},
                {{ 0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f,1.0f}, {0,0,1}, {0,0,0}},
                // -Z face (normal 0,0,-1) tangent along -X
                {{ 0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f,0.0f}, {0,0,-1}, {0,0,0}},
                {{ 0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f,0.0f}, {0,0,-1}, {0,0,0}},
                {{-0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f,1.0f}, {0,0,-1}, {0,0,0}},
                {{-0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f,1.0f}, {0,0,-1}, {0,0,0}},
            };
            // 12 triangles (2 per face) * 3 indices = 36
            std::vector<uint16_t> indices = {
                0,1,2, 2,3,0,         // +X
                4,5,6, 6,7,4,         // -X
                8,9,10, 10,11,8,      // +Y
                12,13,14, 14,15,12,   // -Y
                16,17,18, 18,19,16,   // +Z
                20,21,22, 22,23,20    // -Z
            };

            // compute smooth per-vertex normals by averaging face normals
            std::vector<glm::vec3> normAccum(vertices.size(), glm::vec3(0.0f));
            for (size_t i = 0; i < indices.size(); i += 3) {
                uint32_t i0 = indices[i];
                uint32_t i1 = indices[i+1];
                uint32_t i2 = indices[i+2];

                glm::vec3 p0(vertices[i0].pos[0], vertices[i0].pos[1], vertices[i0].pos[2]);
                glm::vec3 p1(vertices[i1].pos[0], vertices[i1].pos[1], vertices[i1].pos[2]);
                glm::vec3 p2(vertices[i2].pos[0], vertices[i2].pos[1], vertices[i2].pos[2]);

                glm::vec3 edge1 = p1 - p0;
                glm::vec3 edge2 = p2 - p0;
                glm::vec3 faceNormal = glm::cross(edge1, edge2);
                if (glm::length2(faceNormal) > 0.0f) faceNormal = glm::normalize(faceNormal);

                normAccum[i0] += faceNormal;
                normAccum[i1] += faceNormal;
                normAccum[i2] += faceNormal;
            }

            // store averaged normals back into vertices
            for (size_t i = 0; i < vertices.size(); ++i) {
                glm::vec3 n = normAccum[i];
                if (glm::length2(n) > 0.0f) n = glm::normalize(n);
                else n = glm::vec3(0.0f, 0.0f, 1.0f);
                vertices[i].normal[0] = n.x;
                vertices[i].normal[1] = n.y;
                vertices[i].normal[2] = n.z;
            }

            // compute per-vertex tangents from positions and UVs
            std::vector<glm::vec3> tanAccum(vertices.size(), glm::vec3(0.0f));
            for (size_t i = 0; i < indices.size(); i += 3) {
                uint32_t i0 = indices[i];
                uint32_t i1 = indices[i+1];
                uint32_t i2 = indices[i+2];

                glm::vec3 p0(vertices[i0].pos[0], vertices[i0].pos[1], vertices[i0].pos[2]);
                glm::vec3 p1(vertices[i1].pos[0], vertices[i1].pos[1], vertices[i1].pos[2]);
                glm::vec3 p2(vertices[i2].pos[0], vertices[i2].pos[1], vertices[i2].pos[2]);

                glm::vec2 uv0(vertices[i0].uv[0], vertices[i0].uv[1]);
                glm::vec2 uv1(vertices[i1].uv[0], vertices[i1].uv[1]);
                glm::vec2 uv2(vertices[i2].uv[0], vertices[i2].uv[1]);

                glm::vec3 edge1 = p1 - p0;
                glm::vec3 edge2 = p2 - p0;

                glm::vec2 deltaUV1 = uv1 - uv0;
                glm::vec2 deltaUV2 = uv2 - uv0;

                float denom = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
                if (fabs(denom) < 1e-6f) continue;
                float f = 1.0f / denom;

                glm::vec3 tangent = f * (edge1 * deltaUV2.y - edge2 * deltaUV1.y);

                tanAccum[i0] += tangent;
                tanAccum[i1] += tangent;
                tanAccum[i2] += tangent;
            }

            // orthonormalize and store tangent per-vertex
            for (size_t i = 0; i < vertices.size(); ++i) {
                glm::vec3 n(vertices[i].normal[0], vertices[i].normal[1], vertices[i].normal[2]);
                glm::vec3 t = tanAccum[i];
                // Gram-Schmidt orthogonalize
                t = glm::normalize(t - n * glm::dot(n, t));
                if (!glm::isnan(t.x) && glm::length2(t) > 0.0f) {
                    vertices[i].tangent[0] = t.x;
                    vertices[i].tangent[1] = t.y;
                    vertices[i].tangent[2] = t.z;
                } else {
                    // fallback tangent
                    vertices[i].tangent[0] = 1.0f;
                    vertices[i].tangent[1] = 0.0f;
                    vertices[i].tangent[2] = 0.0f;
                }
            }

            vertexBufferObject = VertexBufferObject { createVertexBuffer(vertices), createIndexBuffer(indices), (uint) indices.size() };

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
            ImGui::Checkbox("Flip tangent handedness", &flipTangentHandedness);
            ImGui::SliderFloat("Ambient", &ambientFactor, 0.0f, 1.0f, "%.2f");
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
            proj = glm::perspective(45.0f * 3.1415926f / 180.0f, aspect, 0.1f, 10.0f);
            // flip Y for Vulkan clip space
            proj[1][1] *= -1.0f;

            GLFWwindow* win = getWindow();
            // compute camera axes from quaternion
            cameraFront = glm::normalize(glm::rotate(cameraQuat, glm::vec3(0.0f, 0.0f, -1.0f)));
            cameraUp = glm::normalize(glm::rotate(cameraQuat, glm::vec3(0.0f, 1.0f, 0.0f)));

            if (win) {
                // handle camera rotation keys (R/T = pitch up/down, Y/F = yaw left/right, G/H = roll left/right)
                float ang = cameraAngularSpeed * deltaTime;
                glm::vec3 forward = cameraFront;
                glm::vec3 up = cameraUp;
                glm::vec3 right = glm::normalize(glm::cross(forward, up));

                // New mapping: FR = yaw (- / +), G/T = pitch (- / +), H/Y = roll (- / +)
                // FR: yaw around world up (negative, positive)
                if (glfwGetKey(win, GLFW_KEY_H) == GLFW_PRESS) {
                    cameraQuat = glm::normalize(glm::angleAxis(-ang, glm::vec3(0.0f,1.0f,0.0f)) * cameraQuat);
                }
                if (glfwGetKey(win, GLFW_KEY_F) == GLFW_PRESS) {
                    cameraQuat = glm::normalize(glm::angleAxis(ang, glm::vec3(0.0f,1.0f,0.0f)) * cameraQuat);
                }
                // G/T: pitch around camera right (negative, positive)
                if (glfwGetKey(win, GLFW_KEY_G) == GLFW_PRESS) {
                    cameraQuat = glm::normalize(glm::angleAxis(-ang, right) * cameraQuat);
                }
                if (glfwGetKey(win, GLFW_KEY_T) == GLFW_PRESS) {
                    cameraQuat = glm::normalize(glm::angleAxis(ang, right) * cameraQuat);
                }
                // H/Y: roll around camera forward (negative, positive)
                if (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS) {
                    cameraQuat = glm::normalize(glm::angleAxis(-ang, forward) * cameraQuat);
                }
                if (glfwGetKey(win, GLFW_KEY_Y) == GLFW_PRESS) {
                    cameraQuat = glm::normalize(glm::angleAxis(ang, forward) * cameraQuat);
                }

                // recompute axes after possible rotation
                cameraFront = glm::normalize(glm::rotate(cameraQuat, glm::vec3(0.0f, 0.0f, -1.0f)));
                cameraUp = glm::normalize(glm::rotate(cameraQuat, glm::vec3(0.0f, 1.0f, 0.0f)));

                // handle keyboard input for camera movement (W/A/S/D forward/left/back/right, Q/E up/down)
                float velocity = cameraSpeed * deltaTime;
                if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) cameraPos += cameraFront * velocity;
                if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= cameraFront * velocity;
                right = glm::normalize(glm::cross(cameraFront, cameraUp));
                if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= right * velocity;
                if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) cameraPos += right * velocity;
                if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) cameraPos += cameraUp * velocity;
                if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) cameraPos -= cameraUp * velocity;
            }

            // build view matrix from camera state (no cube rotation)
            view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

            // keep model identity (stop cube rotation)
            model = glm::mat4(1.0f);

            tmp = view * model;
            mvp = proj * view * model;

            // update per-frame uniform buffer (MVP)
            // fill uniform object (MVP, model, viewPos + directional light + POM params)
            UniformObject ubo{};
            ubo.mvp = mvp;
            ubo.model = model;
            // camera world position
            ubo.viewPos = glm::vec4(cameraPos, 1.0f);
            // directional light pointing slightly down and towards -Z
            glm::vec3 lightDir = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.3f));
            ubo.lightDir = glm::vec4(lightDir, 0.0f);
            // white light with intensity in w
            ubo.lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
            // POM params
            ubo.pomParams = glm::vec4(pomHeightScale, pomMinLayers, pomMaxLayers, pomEnabled ? 1.0f : 0.0f);
            ubo.pomFlags = glm::vec4(flipNormalY ? 1.0f : 0.0f, flipTangentHandedness ? 1.0f : 0.0f, ambientFactor, 0.0f);

            updateUniformBuffer(uniform, &ubo, sizeof(UniformObject));
        };

        void draw(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo) override {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);
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


            VkBuffer vertexBuffers[] = { vertexBufferObject.vertexBuffer.buffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, vertexBufferObject.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
            // bind descriptor set for texture
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipelineLayout(), 0, 1, &descriptorSet, 0, nullptr);

            vkCmdDrawIndexed(commandBuffer, vertexBufferObject.indexCount, 1, 0, 0, 0);

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
            // texture and descriptor cleanup (albedo)
            if (textureSampler != VK_NULL_HANDLE) vkDestroySampler(getDevice(), textureSampler, nullptr);
            if (textureImage.view != VK_NULL_HANDLE) vkDestroyImageView(getDevice(), textureImage.view, nullptr);
            if (textureImage.image != VK_NULL_HANDLE) vkDestroyImage(getDevice(), textureImage.image, nullptr);
            if (textureImage.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), textureImage.memory, nullptr);
            // normal map cleanup
            if (normalSampler != VK_NULL_HANDLE) vkDestroySampler(getDevice(), normalSampler, nullptr);
            if (normalImage.view != VK_NULL_HANDLE) vkDestroyImageView(getDevice(), normalImage.view, nullptr);
            if (normalImage.image != VK_NULL_HANDLE) vkDestroyImage(getDevice(), normalImage.image, nullptr);
            if (normalImage.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), normalImage.memory, nullptr);
            // height map cleanup (parallax)
            if (heightSampler != VK_NULL_HANDLE) vkDestroySampler(getDevice(), heightSampler, nullptr);
            if (heightImage.view != VK_NULL_HANDLE) vkDestroyImageView(getDevice(), heightImage.view, nullptr);
            if (heightImage.image != VK_NULL_HANDLE) vkDestroyImage(getDevice(), heightImage.image, nullptr);
            if (heightImage.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), heightImage.memory, nullptr);
            // vertex/index buffers
            vertexBufferObject.destroy(getDevice());

            // uniform buffer
            if (uniform.buffer != VK_NULL_HANDLE) vkDestroyBuffer(getDevice(), uniform.buffer, nullptr);
            if (uniform.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), uniform.memory, nullptr);
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

