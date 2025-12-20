#pragma once

#include "vulkan.hpp"

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

    // Allow derived classes to build ImGui UI per-frame
    virtual void renderImGui();

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
        void createDepthResources();

    public:
        // Generate mipmaps for an image. Works on array textures by specifying
        // layerCount and baseArrayLayer to affect a subset of layers.
        void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, uint32_t layerCount = 1, uint32_t baseArrayLayer = 0);
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
    TextureImage createTextureImageArray(const std::vector<std::string>& filenames, bool srgb = false);
        VkSampler createTextureSampler(uint32_t mipLevels);
        void updateUniformBuffer(Buffer &uniform, void * data, size_t dataSize);
    void createDescriptorPool(uint32_t uboCount, uint32_t samplerCount);
        VkDescriptorSet createDescriptorSet(VkDescriptorSetLayout layout);
        VkDescriptorSet createMaterialDescriptorSet();
        void updateDescriptorSet(VkDescriptorSet &descriptorSet, std::initializer_list<VkWriteDescriptorSet> descriptors);
        void registerDescriptorSet(VkDescriptorSet ds) { if (ds != VK_NULL_HANDLE) registeredDescriptorSets.push_back(ds); }
        const std::vector<VkDescriptorSet>& getRegisteredDescriptorSets() const { return registeredDescriptorSets; }
        VkDescriptorSetLayout getMaterialDescriptorSetLayout() const { return materialDescriptorSetLayout; }
        void setMaterialDescriptorSet(VkDescriptorSet ds) { materialDescriptorSet = ds; }
        VkDescriptorSet getMaterialDescriptorSet() const { return materialDescriptorSet; }
        VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout; }

        // App-owned graphics pipeline accessor
        void setAppGraphicsPipeline(VkPipeline p) { appGraphicsPipeline = p; }
        VkPipeline getAppGraphicsPipeline() const { return appGraphicsPipeline; }
        const std::vector<VkPipeline>& getRegisteredPipelines() const { return registeredPipelines; }

        Buffer createVertexBuffer(const std::vector<Vertex> &vertices);
        Buffer createIndexBuffer(const std::vector<uint> &indices);
        VkShaderModule createShaderModule(const std::vector<char>& code);
    VkPipeline createGraphicsPipeline(std::initializer_list<VkPipelineShaderStageCreateInfo> stages, VkVertexInputBindingDescription bindingDescription, std::initializer_list<VkVertexInputAttributeDescription> attributeDescriptions, VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL, VkCullModeFlagBits cullMode = VK_CULL_MODE_BACK_BIT, bool depthWrite = true, bool colorWrite = true, VkCompareOp depthCompare = VK_COMPARE_OP_LESS);
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
        VkDescriptorPool getDescriptorPool() const { return descriptorPool; }
        VkDescriptorPool getImGuiDescriptorPool() const { return imguiDescriptorPool; }

        int getWidth();
        int getHeight();
        
        // Public utility methods for texture manipulation
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
        VkCommandBuffer beginSingleTimeCommands();
        void endSingleTimeCommands(VkCommandBuffer commandBuffer);

        // Image helpers (moved from protected so external helpers can use them)
        void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, uint32_t mipLevelCount, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
        void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
        void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

        void run();
    // request the app to close the main window
    void requestClose();
        virtual void setup() = 0;
        virtual void update(float deltaTime) = 0;
        virtual void draw(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo) = 0;
        virtual void clean() = 0;
        // Called after a frame has been submitted/presented. Derived apps may override.
        virtual void postSubmit();

};
