#pragma once

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <optional>
#include <set>
#include <fstream>
#include <array>
#include <algorithm>
#include <cmath>
// stb_image for texture loading
#include <stb/stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/integer.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp> 
#include <glm/matrix.hpp>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct ShaderStage {
    VkPipelineShaderStageCreateInfo info{};

    ShaderStage(VkShaderModule m, VkShaderStageFlagBits s) {
        info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage  = s;
        info.module = m;
        info.pName  = "main";
    }
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

class FileReader {
    public:
    static std::vector<char> readFile(const std::string& filename);
};

struct TextureImage {
    uint32_t mipLevels = 1;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

struct VertexBufferObject {
    Buffer vertexBuffer;
    Buffer indexBuffer;
    uint32_t indexCount = 0;

    void destroy(const VkDevice &device) {
        if (indexBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, indexBuffer.buffer, nullptr);
        if (indexBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(device, indexBuffer.memory, nullptr);
        if (vertexBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, vertexBuffer.buffer, nullptr);
        if (vertexBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(device, vertexBuffer.memory, nullptr);
    }

};


struct Vertex { float pos[3]; float color[3]; float uv[2]; float normal[3]; float tangent[3]; float texIndex; };


class VulkanApp {
    GLFWwindow* window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainExtent;
    std::vector<VkImageView> swapchainImageViews;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> swapchainFramebuffers;
    VkCommandPool commandPool = VK_NULL_HANDLE;

    VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
    VkFence inFlightFence = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    // texture and descriptor

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    // depth resources
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;
    
protected:
    // set when the framebuffer (GLFW window) is resized so we can recreate swapchain
    bool framebufferResized = false;
    // fullscreen handling
    bool isFullscreen = false;
    int windowedPosX = 100;
    int windowedPosY = 100;
    int windowedWidth = WIDTH;
    int windowedHeight = HEIGHT;
    // ImGui integration state
    VkDescriptorPool imguiDescriptorPool = VK_NULL_HANDLE;
    bool imguiShowDemo = false;
    double imguiLastTime = 0.0;
    float imguiFps = 0.0f;
    // frame timing for update delta calculation
    double lastFrameTime = 0.0;

    // Allow derived classes to build ImGui UI per-frame
    virtual void renderImGui() {}

    // expose the GLFW window to derived classes for input polling
    GLFWwindow* getWindow();

    private:
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    protected:
        void toggleFullscreen();

        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
        void createSwapchain();
        void createImageViews();
        void createRenderPass();
        void createFramebuffers();
        void createCommandPool();
        void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, uint32_t mipLevelCount, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, uint32_t layerCount = 1);
        void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
        void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
        void createDepthResources();
        void createDescriptorSetLayout();
    void cleanupSwapchain();
    void recreateSwapchain();
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);


        void drawFrame();
        void createInstance();
        bool checkValidationLayerSupport();
        void setupDebugMessenger();
        void createSurface() ;
        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);        
        void pickPhysicalDevice();
        bool isDeviceSuitable(VkPhysicalDevice device);
        void createLogicalDevice() ;
        void createSyncObjects();
        void createTextureImageView(TextureImage &textureImage);

    private:
        void initWindow();
        void initVulkan();
    void initImGui();
    void cleanupImGui();
        void mainLoop();
        void cleanup();




    public:
        Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
        TextureImage createTextureImage(const char * filename);
    TextureImage createTextureImageArray(const std::vector<std::string>& filenames);
        VkSampler createTextureSampler(uint32_t mipLevels);
        void updateUniformBuffer(Buffer &uniform, void * data, size_t dataSize);
    void createDescriptorPool(uint32_t maxSets = 1);
        VkDescriptorSet createDescriptorSet();
        void updateDescriptorSet(VkDescriptorSet &descriptorSet, std::initializer_list<VkWriteDescriptorSet> descriptors);

        Buffer createVertexBuffer(std::vector<Vertex> &vertices);
        Buffer createIndexBuffer(std::vector<uint16_t> &indices);
        VkShaderModule createShaderModule(const std::vector<char>& code);
        VkPipeline createGraphicsPipeline(std::initializer_list<VkPipelineShaderStageCreateInfo> stages, VkVertexInputBindingDescription bindingDescription, std::initializer_list<VkVertexInputAttributeDescription> attributeDescriptions);
        std::vector<VkCommandBuffer> createCommandBuffers();

        VkDevice getDevice() const;
        VkPipelineLayout getPipelineLayout() const;

        int getWidth();
        int getHeight();
        
        // Public utility methods for texture manipulation
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
        VkCommandBuffer beginSingleTimeCommands();
        void endSingleTimeCommands(VkCommandBuffer commandBuffer);

        void run();
    // request the app to close the main window
    void requestClose();
        virtual void setup() = 0;
        virtual void update(float deltaTime) = 0;
        virtual void draw(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo) = 0;
        virtual void clean() = 0;

};