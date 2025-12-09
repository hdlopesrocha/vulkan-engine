#include "vulkan/vulkan.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

struct UniformObject {
    glm::mat4 mvp;
    glm::vec4 lightDir; // xyz = direction, w = unused (padding)
    glm::vec4 lightColor; // rgb = color, w = intensity
};

class MyApp : public VulkanApp {
    public:
        VkPipeline graphicsPipeline = VK_NULL_HANDLE;
        VkSampler textureSampler = VK_NULL_HANDLE;
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
                        VkVertexInputAttributeDescription {
                            0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)
                        },
                        VkVertexInputAttributeDescription {
                            1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)
                        },
                        VkVertexInputAttributeDescription {
                            2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)
                        }
                    }
                );

                vkDestroyShaderModule(getDevice(), fragmentShader.info.module, nullptr);
                vkDestroyShaderModule(getDevice(), vertexShader.info.module, nullptr);
            }
            textureImage = createTextureImage("textures/grass_color.jpg");
            textureSampler = createTextureSampler(textureImage.mipLevels);
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

                updateDescriptorSet(
                    descriptorSet, 
                    { uboWrite, samplerWrite }
                );
            }

            // 24 unique vertices (4 per face) so each face can have its own UVs
            std::vector<Vertex> vertices = {
                // +X face
                {{ 0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f, 0.0f}},
                {{ 0.5f, -0.5f,  0.5f }, {1,1,1}, {1.0f, 0.0f}},
                {{ 0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f, 1.0f}},
                {{ 0.5f,  0.5f, -0.5f }, {1,1,1}, {0.0f, 1.0f}},
                // -X face
                {{-0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, -0.5f }, {1,1,1}, {1.0f, 0.0f}},
                {{-0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f, 1.0f}},
                {{-0.5f,  0.5f,  0.5f }, {1,1,1}, {0.0f, 1.0f}},
                // +Y face
                {{-0.5f,  0.5f, -0.5f }, {1,1,1}, {0.0f, 0.0f}},
                {{ 0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f, 0.0f}},
                {{ 0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f, 1.0f}},
                {{-0.5f,  0.5f,  0.5f }, {1,1,1}, {0.0f, 1.0f}},
                // -Y face
                {{-0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f, 0.0f}},
                {{ 0.5f, -0.5f,  0.5f }, {1,1,1}, {1.0f, 0.0f}},
                {{ 0.5f, -0.5f, -0.5f }, {1,1,1}, {1.0f, 1.0f}},
                {{-0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f, 1.0f}},
                // +Z face
                {{-0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f, 0.0f}},
                {{-0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f, 0.0f}},
                {{ 0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f, 1.0f}},
                {{ 0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f, 1.0f}},
                // -Z face
                {{ 0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f, 0.0f}},
                {{ 0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f, 0.0f}},
                {{-0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f, 1.0f}},
                {{-0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f, 1.0f}},
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

            view = glm::translate(view, glm::vec3(0.0f, 0.0f, -2.5f));

            float time = (float)glfwGetTime();
            model = glm::rotate(model, time * 0.8f, glm::vec3(0.0f, 1.0f, 0.0f));

            tmp = view * model;
            mvp = proj * view * model;

            // update per-frame uniform buffer (MVP)
            // fill uniform object (MVP + directional light)
            UniformObject ubo{};
            ubo.mvp = mvp;
            // directional light pointing slightly down and towards -Z
            glm::vec3 lightDir = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.3f));
            ubo.lightDir = glm::vec4(lightDir, 0.0f);
            // white light with intensity in w
            ubo.lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

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
            // texture and descriptor cleanup
            if (textureSampler != VK_NULL_HANDLE) vkDestroySampler(getDevice(), textureSampler, nullptr);
            // texture and descriptor cleanup
            if (textureImage.view != VK_NULL_HANDLE) vkDestroyImageView(getDevice(), textureImage.view, nullptr);
            if (textureImage.image != VK_NULL_HANDLE) vkDestroyImage(getDevice(), textureImage.image, nullptr);
            if (textureImage.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), textureImage.memory, nullptr);
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

