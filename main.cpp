#include "vulkan/vulkan.hpp"

class MyApp : public VulkanApp {
    public:
        VkPipeline graphicsPipeline = VK_NULL_HANDLE;
        VkSampler textureSampler = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        Buffer uniform;
        TextureImage textureImage;

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
            createVertexBuffer();
            createIndexBuffer();

            createCommandBuffers(graphicsPipeline, descriptorSet);
        };

        void update(float deltaTime) override {
            // update per-frame uniform buffer (MVP)
            updateUniformBuffer(uniform);
        };

        void draw() override {

        };

        void clean() override {
            if (graphicsPipeline != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), graphicsPipeline, nullptr);
            // texture and descriptor cleanup
            if (textureSampler != VK_NULL_HANDLE) vkDestroySampler(getDevice(), textureSampler, nullptr);
            // texture and descriptor cleanup
            if (textureImage.view != VK_NULL_HANDLE) vkDestroyImageView(getDevice(), textureImage.view, nullptr);
            if (textureImage.image != VK_NULL_HANDLE) vkDestroyImage(getDevice(), textureImage.image, nullptr);
            if (textureImage.memory != VK_NULL_HANDLE) vkFreeMemory(getDevice(), textureImage.memory, nullptr);
            
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

