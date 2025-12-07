#include "vulkan/vulkan.hpp"

class MyApp : public VulkanApp {
    public:

        VkPipeline graphicsPipeline = VK_NULL_HANDLE;

        void setup() override {
            ShaderStage vertexShader = ShaderStage(createShaderModule(
                FileReader::readFile("shaders/triangle.vert.spv")), 
                VK_SHADER_STAGE_VERTEX_BIT
            );

            ShaderStage fragmentShader = ShaderStage(createShaderModule(
                FileReader::readFile("shaders/triangle.frag.spv")), 
                VK_SHADER_STAGE_FRAGMENT_BIT
            );

            graphicsPipeline = createGraphicsPipeline({
                vertexShader.info, 
                fragmentShader.info
            });

            vkDestroyShaderModule(getDevice(), fragmentShader.info.module, nullptr);
            vkDestroyShaderModule(getDevice(), vertexShader.info.module, nullptr);

            createTextureImage();
            createTextureImageView();
            createTextureSampler();
            createUniformBuffer();
            createDescriptorPool();
            createDescriptorSet();
            createVertexBuffer();
            createIndexBuffer();

            createCommandBuffers(graphicsPipeline);
        };

        void update(float deltaTime) override {

        };

        void draw() override {

        };

        void clean() override {
            if (graphicsPipeline != VK_NULL_HANDLE) vkDestroyPipeline(getDevice(), graphicsPipeline, nullptr);
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

