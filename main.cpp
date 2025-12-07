#include "vulkan/vulkan.hpp"

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
            uniform = createBuffer(sizeof(float) * 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            createDescriptorPool();
            // Descriptor Set
            {
                descriptorSet = createDescriptorSet();
            
                // uniform buffer descriptor (binding 0)
                VkDescriptorBufferInfo bufferInfo { uniform.buffer, 0 , sizeof(float) * 16 };

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

        void update(float deltaTime) override {
            // update per-frame uniform buffer (MVP)
            updateUniformBuffer(uniform);
        };

        void draw(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo) override {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);



            vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
            // bind pipeline and draw the indexed square
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);


            VkBuffer vertexBuffers[] = { vertexBufferObject.vertexBuffer.buffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, vertexBufferObject.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
            // bind descriptor set for texture
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipelineLayout(), 0, 1, &descriptorSet, 0, nullptr);

            vkCmdDrawIndexed(commandBuffer, vertexBufferObject.indexCount, 1, 0, 0, 0);

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

