
// Standard library includes first

#pragma once

// Standard library includes first
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <cstring>

#include <vulkan/vulkan.h>

#include "vulkan.hpp"

class VulkanApp {
    public:
        // Main UBO/sampler descriptor set (allocated from descriptorSetLayout)
        VkDescriptorSet mainDescriptorSet = VK_NULL_HANDLE;
        VkDescriptorSet getMainDescriptorSet() const { return mainDescriptorSet; }
        void createRenderPasses();
        // Utility: Generate vegetation instances using a compute shader
        // vertexBuffer: input triangle mesh (positions, 3 floats per vertex)
        // vertexCount: number of vertices in the buffer
        // indexBuffer: index buffer (uint32_t indices)
        // indexCount: number of indices in the buffer
        // instancesPerTriangle: how many instances to generate per triangle
        // outputBuffer: will be filled with instance positions (vec3)
        // Returns: number of generated instances
        uint32_t generateVegetationInstancesCompute(
            VkBuffer vertexBuffer, uint32_t vertexCount,
            VkBuffer indexBuffer, uint32_t indexCount,
            uint32_t instancesPerTriangle,
            VkBuffer outputBuffer, uint32_t outputBufferSize, uint32_t seed = 1337);
        void initWindow(); // Only one declaration, public
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
    VkRenderPass continuationRenderPass = VK_NULL_HANDLE;  // Render pass that loads existing color/depth
    std::vector<VkFramebuffer> swapchainFramebuffers;
    VkCommandPool commandPool = VK_NULL_HANDLE;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    // track which fence is using each swapchain image (to avoid writing to an image in use)
    std::vector<VkFence> imagesInFlight;
    // frame index for round-robin CPU frames-in-flight
    uint32_t currentFrame = 0;

public:
    uint32_t getCurrentFrame() const { return currentFrame; }

private:
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    public:
        // Public accessor for command pool (needed for buffer transfers)
        VkCommandPool getCommandPool() const { return commandPool; }
    // Application main graphics pipeline (owner: app / main.cpp)
    VkPipeline appGraphicsPipeline = VK_NULL_HANDLE;
    // texture and descriptor

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    // Dedicated descriptor set layout for global materials (binding 5)
    VkDescriptorSetLayout materialDescriptorSetLayout = VK_NULL_HANDLE;
    // Global material descriptor set (bound once and updated when materials change)
    VkDescriptorSet materialDescriptorSet = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    // Registered descriptor sets for runtime inspection (widgets can read these)
    std::vector<VkDescriptorSet> registeredDescriptorSets;
    // Registered graphics pipelines for runtime inspection
    std::vector<VkPipeline> registeredPipelines;
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
    // V-Sync preference (affects present mode selection)
    bool vsyncEnabled = true;
    bool vsyncChanged = false; // track if user toggled vsync to trigger swapchain recreation

    // Allow derived classes to build ImGui UI per-frame
    virtual void renderImGui();

    // expose the GLFW window to derived classes for input polling
    GLFWwindow* getWindow();

    private:
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    protected:
        void toggleFullscreen();
        // V-Sync control (call to change present mode at runtime)
        void setVSyncEnabled(bool enabled);

        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
        void createSwapchain();
        void createImageViews();
        void createRenderPass();
        void createFramebuffers();
        void createCommandPool();
        void createDepthResources();

    public:
        // Generate mipmaps for an image. Works on array textures by specifying
        // layerCount and baseArrayLayer to affect a subset of layers.
        void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, uint32_t layerCount = 1, uint32_t baseArrayLayer = 0);

        // Record mipmap generation commands into an existing command buffer (no begin/end or wait)
        void recordGenerateMipmaps(VkCommandBuffer commandBuffer, VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, uint32_t layerCount = 1, uint32_t baseArrayLayer = 0);

        // Submit a pre-recorded command buffer asynchronously and return a fence that will be signaled on completion.
        // If outSemaphore is non-null, the submission will signal that semaphore when finished (useful to make frame submit wait on generation).
        VkFence submitCommandBufferAsync(VkCommandBuffer commandBuffer, VkSemaphore* outSemaphore = nullptr);
        void processPendingCommandBuffers();
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

    public:
        void initVulkan();
    void initImGui();
    void cleanupImGui();
        void mainLoop();
        void cleanup();




    public:
        Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
        TextureImage createTextureImage(const char * filename);
    TextureImage createTextureImageArray(const std::vector<std::string>& filenames, bool srgb = false);
        VkSampler createTextureSampler(uint32_t mipLevels);
        void updateUniformBuffer(Buffer &uniform, void * data, size_t dataSize);
    void createDescriptorPool(uint32_t uboCount, uint32_t samplerCount);
        VkDescriptorSet createDescriptorSet(VkDescriptorSetLayout layout);
        VkDescriptorSet createMaterialDescriptorSet();
        void updateDescriptorSet(VkDescriptorSet &descriptorSet, std::initializer_list<VkWriteDescriptorSet> descriptors);
        void updateDescriptorSet(VkDescriptorSet &descriptorSet, const std::vector<VkWriteDescriptorSet> &descriptors);
        void registerDescriptorSet(VkDescriptorSet ds) { if (ds != VK_NULL_HANDLE) registeredDescriptorSets.push_back(ds); }
        const std::vector<VkDescriptorSet>& getRegisteredDescriptorSets() const { return registeredDescriptorSets; }
        VkDescriptorSetLayout getMaterialDescriptorSetLayout() const { return materialDescriptorSetLayout; }
        void setMaterialDescriptorSet(VkDescriptorSet ds) { materialDescriptorSet = ds; }
        VkDescriptorSet getMaterialDescriptorSet() const { return materialDescriptorSet; }
        VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout; }

        // App-owned graphics pipeline accessor
        void setAppGraphicsPipeline(VkPipeline p) { 
            //printf("[VulkanApp] setAppGraphicsPipeline: pipeline=%p\n", (void*)p);
            appGraphicsPipeline = p; 
        }
        VkPipeline getAppGraphicsPipeline() const { 
            //printf("[VulkanApp] getAppGraphicsPipeline: pipeline=%p\n", (void*)appGraphicsPipeline);
            return appGraphicsPipeline; 
        }
        const std::vector<VkPipeline>& getRegisteredPipelines() const { return registeredPipelines; }

        Buffer createVertexBuffer(const std::vector<Vertex> &vertices);
        Buffer createIndexBuffer(const std::vector<uint> &indices);
        // Create a device-local storage buffer and upload data via staging transfer
        Buffer createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage);
        VkShaderModule createShaderModule(const std::vector<char>& code);
    // Refactored: Accepts set layouts and optional push constant range, returns pipeline and layout
    std::pair<VkPipeline, VkPipelineLayout> createGraphicsPipeline(
        std::initializer_list<VkPipelineShaderStageCreateInfo> stages,
        const std::vector<VkVertexInputBindingDescription>& bindingDescriptions,
        std::initializer_list<VkVertexInputAttributeDescription> attributeDescriptions,
        const std::vector<VkDescriptorSetLayout>& setLayouts = {},
        const VkPushConstantRange* pushConstantRange = nullptr,
        VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL,
        VkCullModeFlagBits cullMode = VK_CULL_MODE_BACK_BIT,
        bool depthWrite = true,
        bool colorWrite = true,
        VkCompareOp depthCompare = VK_COMPARE_OP_LESS,
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        VkRenderPass renderPassOverride = VK_NULL_HANDLE);
        std::vector<VkCommandBuffer> createCommandBuffers();

        VkDevice getDevice() const;
        VkPipelineLayout getPipelineLayout() const;

        // Public getters for runtime inspection (used by widgets)
        VkInstance getInstance() const { return instance; }
        VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
        VkQueue getGraphicsQueue() const { return graphicsQueue; }
        VkQueue getPresentQueue() const { return presentQueue; }
        VkSwapchainKHR getSwapchain() const { return swapchain; }
        VkFormat getSwapchainImageFormat() const { return swapchainImageFormat; }
        VkExtent2D getSwapchainExtent() const { return swapchainExtent; }
        const std::vector<VkImage>& getSwapchainImages() const { return swapchainImages; }
        const std::vector<VkImageView>& getSwapchainImageViews() const { return swapchainImageViews; }
        const std::vector<VkFramebuffer>& getSwapchainFramebuffers() const { return swapchainFramebuffers; }
        VkDescriptorPool getDescriptorPool() const { return descriptorPool; }
        VkDescriptorPool getImGuiDescriptorPool() const { return imguiDescriptorPool; }
        VkRenderPass getSwapchainRenderPass() const { return renderPass; }
        VkRenderPass getContinuationRenderPass() const { return continuationRenderPass; }

        VkImage getDepthImage() const { return depthImage; }
        VkImageView getDepthImageView() const { return depthImageView; }

        int getWidth();
        int getHeight();
        
        // Public utility methods for texture manipulation
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
        VkCommandBuffer beginSingleTimeCommands();
        void endSingleTimeCommands(VkCommandBuffer commandBuffer);

        // Image helpers (moved from protected so external helpers can use them)
        void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, uint32_t mipLevelCount, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
        void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels = 1, uint32_t arrayLayers = 1);
        void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

        void run();
    // request the app to close the main window
    void requestClose();
        virtual void setup() = 0;
        virtual void update(float deltaTime) = 0;
        virtual void preRenderPass(VkCommandBuffer &commandBuffer) {} // Called after vkBeginCommandBuffer but before vkCmdBeginRenderPass
        virtual void draw(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo) = 0;
        virtual void clean() = 0;
        // Called after swapchain recreation so derived apps can resize their offscreen resources
        virtual void onSwapchainResized(uint32_t /*width*/, uint32_t /*height*/) {}
        // Called after a frame has been submitted/presented. Derived apps may override.
        virtual void postSubmit();

};
