#include <cstdint>
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <mutex>
#include <vector>

// Async submission bookkeeping
std::mutex pendingCmdMutex;
std::vector<std::pair<VkCommandBuffer,VkFence>> pendingCommandBuffers;

std::mutex extraSemaphoreMutex;
std::vector<VkSemaphore> extraWaitSemaphores;
// semaphores scheduled for destruction paired with the frame fence they were associated with
std::vector<std::pair<VkSemaphore,VkFence>> semaphoresPendingDestroy;

#include "VulkanApp.hpp"
#include "TextureArrayManager.hpp"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include <cmath>


void VulkanApp::initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createRenderPasses();
    createDescriptorSetLayout();
    createCommandPool();
    createDepthResources();
    createFramebuffers();
    std::cerr << "[DEBUG][initVulkan] swapchainFramebuffers.size() after createFramebuffers: " << swapchainFramebuffers.size() << std::endl;
    std::cerr << "[DEBUG][initVulkan] swapchainImages.size() after createSwapchain: " << swapchainImages.size() << std::endl;
    std::cerr << "[DEBUG][initVulkan] swapchainFramebuffers before createCommandBuffers: ";
    for (size_t i = 0; i < swapchainFramebuffers.size(); ++i) {
        std::cerr << (void*)swapchainFramebuffers[i] << " ";
    }
    std::cerr << std::endl;
    createCommandPool();
    std::cerr << "[DEBUG][initVulkan] commandBuffers.size() before createCommandBuffers: " << commandBuffers.size() << std::endl;
    commandBuffers = createCommandBuffers();
    std::cerr << "[DEBUG][initVulkan] commandBuffers.size() after createCommandBuffers: " << commandBuffers.size() << std::endl;
    createSyncObjects();
    initImGui();
}


void VulkanApp::requestClose() {
    if (window) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

// Utility: Generate vegetation instances using a compute shader
// vertexBuffer: input triangle mesh (positions, 3 floats per vertex)
// vertexCount: number of vertices in the buffer
// instancesPerTriangle: how many instances to generate per triangle
// outputBuffer: will be filled with instance positions (vec3)
// Returns: number of generated instances
// Now supports indexed geometry: pass indexBuffer and indexCount
uint32_t VulkanApp::generateVegetationInstancesCompute(
    VkBuffer vertexBuffer, uint32_t vertexCount,
    VkBuffer indexBuffer, uint32_t indexCount,
    uint32_t instancesPerTriangle,
    VkBuffer outputBuffer, uint32_t outputBufferSize, uint32_t seed) {
    // ...existing implementation (restore as needed)...
    return 0; // TODO: restore correct return value and implementation
}

void VulkanApp::createImageViews() {
    swapchainImageViews.resize(swapchainImages.size());
    std::cerr << "[DEBUG] createImageViews: swapchainImageViews.size()=" << swapchainImageViews.size() << std::endl;
    for (size_t i = 0; i < swapchainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapchainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapchainImageFormat;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &createInfo, nullptr, &swapchainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }
}

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "validation: " << pCallbackData->pMessage << "\n";
    }
    return VK_FALSE;
}

// --- New helper methods for basic rendering ---
std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME,  // For GPU-driven indirect draw count
    VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME
};



void VulkanApp::initWindow() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // explicit allow window resizing
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Engine", nullptr, nullptr);
    // register resize callback so we can recreate the swapchain when user resizes window
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    // keyboard input is handled by the event system (KeyboardPublisher)
    // do not register a direct key callback here to avoid duplicate handling
}

void VulkanApp::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Intentionally left empty: keyboard input (including F11/ESC)
    // is handled by the event system (KeyboardPublisher -> EventManager).
    (void)window; (void)key; (void)scancode; (void)action; (void)mods;
}

void VulkanApp::toggleFullscreen() {
    if (!window) return;

    if (!isFullscreen) {
        // store windowed position and size
        glfwGetWindowPos(window, &windowedPosX, &windowedPosY);
        glfwGetWindowSize(window, &windowedWidth, &windowedHeight);

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        if (!monitor) return;
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        if (!mode) return;

        glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
        isFullscreen = true;
        return;
    }
    createDescriptorSetLayout();
    createCommandPool();
    createDepthResources();
    createFramebuffers();
    createSyncObjects();
    initImGui();
}

void VulkanApp::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }
}

void VulkanApp::cleanup() {
    printf("[VulkanApp] cleanup start - device=%p\n", (void*)device);
    if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);

    // destroy sync objects
    for (auto f : inFlightFences) {
        if (f != VK_NULL_HANDLE) vkDestroyFence(device, f, nullptr);
    }
    for (auto s : renderFinishedSemaphores) {
        if (s != VK_NULL_HANDLE) vkDestroySemaphore(device, s, nullptr);
    }
    for (auto s : imageAvailableSemaphores) {
        if (s != VK_NULL_HANDLE) vkDestroySemaphore(device, s, nullptr);
    }

    // command pool
    if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);

    // framebuffers
    for (auto fb : swapchainFramebuffers) {
        if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(device, fb, nullptr);
    }

    clean();

    // ImGui cleanup (must happen before destroying descriptor pools and device)
    printf("[VulkanApp] calling cleanupImGui() - imguiDescriptorPool=%p\n", (void*)imguiDescriptorPool);
    cleanupImGui();
    printf("[VulkanApp] cleanupImGui() returned\n");
    // depth resources
    if (depthImageView != VK_NULL_HANDLE) vkDestroyImageView(device, depthImageView, nullptr);
    if (depthImage != VK_NULL_HANDLE) vkDestroyImage(device, depthImage, nullptr);
    if (depthImageMemory != VK_NULL_HANDLE) vkFreeMemory(device, depthImageMemory, nullptr);
    if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    if (materialDescriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, materialDescriptorSetLayout, nullptr);

    // destroy pipeline
    if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

    // render pass
    if (renderPass != VK_NULL_HANDLE) vkDestroyRenderPass(device, renderPass, nullptr);
    if (continuationRenderPass != VK_NULL_HANDLE) vkDestroyRenderPass(device, continuationRenderPass, nullptr);

    // image views
    for (auto iv : swapchainImageViews) {
        if (iv != VK_NULL_HANDLE) vkDestroyImageView(device, iv, nullptr);
    }

    // swapchain
    if (swapchain != VK_NULL_HANDLE) {
        auto fp = (PFN_vkDestroySwapchainKHR)vkGetInstanceProcAddr(instance, "vkDestroySwapchainKHR");
        if (fp) fp(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }

    // destroy surface
    if (surface != VK_NULL_HANDLE) {
        auto fp = (PFN_vkDestroySurfaceKHR)vkGetInstanceProcAddr(instance, "vkDestroySurfaceKHR");
        if (fp) fp(instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }

    // Destroy window and terminate GLFW BEFORE destroying Vulkan instance
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();

    printf("[VulkanApp] about to vkDestroyDevice(device=%p)\n", (void*)device);
    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    printf("[VulkanApp] vkDestroyDevice returned\n");

    if (debugMessenger != VK_NULL_HANDLE) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func) func(instance, debugMessenger, nullptr);
        debugMessenger = VK_NULL_HANDLE;
    }

    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
    printf("[VulkanApp] vkDestroyInstance returned\n");
}

void VulkanApp::initImGui() {
    // Create descriptor pool for ImGui
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * (uint32_t)std::size(pool_sizes);
    pool_info.poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes));
    pool_info.pPoolSizes = pool_sizes;

    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &imguiDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create ImGui descriptor pool!");
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = instance;
    init_info.PhysicalDevice = physicalDevice;
    init_info.Device = device;
    init_info.QueueFamily = findQueueFamilies(physicalDevice).graphicsFamily.value();
    init_info.Queue = graphicsQueue;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = imguiDescriptorPool;
    init_info.MinImageCount = 2;
    init_info.ImageCount = static_cast<uint32_t>(swapchainImages.size());
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = nullptr;
    init_info.RenderPass = renderPass;

    bool imguiInitOk = ImGui_ImplVulkan_Init(&init_info);
    printf("[ImGui] ImGui_ImplVulkan_Init returned %s\n", imguiInitOk ? "true" : "false");

    // upload fonts (API differs between ImGui versions)
    ImGui_ImplVulkan_CreateFontsTexture();
    ImGui_ImplVulkan_DestroyFontsTexture();
    if (!imguiInitOk) {
        printf("[ImGui] ERROR: ImGui_ImplVulkan_Init failed!\n");
    }
}

void VulkanApp::cleanupImGui() {
    if (imguiDescriptorPool != VK_NULL_HANDLE) {
        printf("[VulkanApp] cleanupImGui start - imguiDescriptorPool=%p device=%p\n", (void*)imguiDescriptorPool, (void*)device);
        vkDeviceWaitIdle(device);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        vkDestroyDescriptorPool(device, imguiDescriptorPool, nullptr);
        imguiDescriptorPool = VK_NULL_HANDLE;
        printf("[VulkanApp] cleanupImGui done\n");
    }
}

// Default empty implementations for overridable hooks declared in the header.
void VulkanApp::renderImGui() {
    // No-op default
}

void VulkanApp::postSubmit() {
    // No-op default
}




VkSurfaceFormatKHR VulkanApp::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

VkPresentModeKHR VulkanApp::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    // If V-Sync is disabled, prefer IMMEDIATE mode for uncapped FPS (lowest latency, may tear)
    if (!vsyncEnabled) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                return availablePresentMode;
            }
        }
    }
    // With V-Sync enabled, prefer MAILBOX (triple-buffering, low latency, no tearing)
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) return availablePresentMode;
    }
    // Fallback to FIFO (guaranteed available, V-Sync, highest latency)
    return VK_PRESENT_MODE_FIFO_KHR;
}

void VulkanApp::setVSyncEnabled(bool enabled) {
    if (vsyncEnabled != enabled) {
        vsyncEnabled = enabled;
        vsyncChanged = true; // will trigger swapchain recreation on next frame
    }
}

VkExtent2D VulkanApp::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        VkExtent2D actualExtent = {WIDTH, HEIGHT};
        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
        return actualExtent;
    }
}

void VulkanApp::createSwapchain() {
    // query support
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    if (formatCount != 0) vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    if (presentModeCount != 0) vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(presentModes);
    VkExtent2D extent = chooseSwapExtent(capabilities);

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    auto fpCreateSwapchain = (PFN_vkCreateSwapchainKHR)vkGetInstanceProcAddr(instance, "vkCreateSwapchainKHR");
    if (fpCreateSwapchain == nullptr || fpCreateSwapchain(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
        // try using device-level create
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }
    }

    uint32_t actualImageCount = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &actualImageCount, nullptr);
    swapchainImages.resize(actualImageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &actualImageCount, swapchainImages.data());
    std::cerr << "[DEBUG] createSwapchain: swapchainImages.size()=" << swapchainImages.size() << std::endl;
    swapchainImageFormat = surfaceFormat.format;
    swapchainExtent = extent;
}

void VulkanApp::createRenderPasses() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapchainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency externalDependency{};
    externalDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    externalDependency.dstSubpass = 0;
    externalDependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    externalDependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    externalDependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    externalDependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // Self-dependency for barriers inside the subpass
    VkSubpassDependency selfDependency{};
    selfDependency.srcSubpass = 0;
    selfDependency.dstSubpass = 0;
    selfDependency.srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    selfDependency.dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    selfDependency.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    selfDependency.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    selfDependency.dependencyFlags = 0;

    std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };

    std::array<VkSubpassDependency, 2> dependencies = { externalDependency, selfDependency };

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
    // Create continuation render pass (loads existing color and depth instead of clearing)
    VkAttachmentDescription contColorAttachment{};
    contColorAttachment.format = swapchainImageFormat;
    contColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    contColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // Load existing color
    contColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    contColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    contColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    contColorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    contColorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription contDepthAttachment{};
    contDepthAttachment.format = VK_FORMAT_D32_SFLOAT;
    contDepthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    contDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // Load existing depth
    contDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    contDepthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    contDepthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    contDepthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    contDepthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::array<VkAttachmentDescription, 2> contAttachments = { contColorAttachment, contDepthAttachment };

    VkSubpassDependency contDependency{};
    contDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    contDependency.dstSubpass = 0;
    contDependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    contDependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    contDependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    contDependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo contRenderPassInfo{};
    contRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    contRenderPassInfo.attachmentCount = static_cast<uint32_t>(contAttachments.size());
    contRenderPassInfo.pAttachments = contAttachments.data();
    contRenderPassInfo.subpassCount = 1;
    contRenderPassInfo.pSubpasses = &subpass;  // Reuse the same subpass description
    contRenderPassInfo.dependencyCount = 1;
    contRenderPassInfo.pDependencies = &contDependency;

    if (vkCreateRenderPass(device, &contRenderPassInfo, nullptr, &continuationRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create continuation render pass!");
    }
}

void VulkanApp::createFramebuffers() {
    swapchainFramebuffers.resize(swapchainImageViews.size());
    std::cerr << "[DEBUG] createFramebuffers: swapchainFramebuffers.size()=" << swapchainFramebuffers.size() << std::endl;
    for (size_t i = 0; i < swapchainImageViews.size(); i++) {
        VkImageView attachments[] = { swapchainImageViews[i], depthImageView };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 2;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapchainExtent.width;
        framebufferInfo.height = swapchainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapchainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }

    // Called here to cleanup any completed async submissions from previous runs
    processPendingCommandBuffers();
}


void VulkanApp::createCommandPool() {
    // destroy command pool (if any)
    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    // initialize async bookkeeping containers
    // pendingCommandBuffers and extraWaitSemaphores are guarded by mutexes
    pendingCommandBuffers.clear();
    extraWaitSemaphores.clear();
    semaphoresPendingDestroy.clear();
}

VkCommandBuffer VulkanApp::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    return commandBuffer;
}

void VulkanApp::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void VulkanApp::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, uint32_t mipLevelCount, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevelCount;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    // Debug: log the created image handle and requested usage flags
    printf("[VulkanApp::createImage] created image=%p usage=0x%08x size=%ux%u format=%d mipLevels=%u arrayLayers=%u\n",
           (void*)image, (unsigned int)usage, width, height, (int)format, mipLevelCount, imageInfo.arrayLayers);

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
}

void VulkanApp::generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, uint32_t layerCount, uint32_t baseArrayLayer) {
    // Existing blocking helper kept for convenience
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    recordGenerateMipmaps(commandBuffer, image, imageFormat, texWidth, texHeight, mipLevels, layerCount, baseArrayLayer);
    endSingleTimeCommands(commandBuffer);
}

// Record mipmap generation commands into an existing command buffer (no begin/end or wait)
void VulkanApp::recordGenerateMipmaps(VkCommandBuffer commandBuffer, VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, uint32_t layerCount, uint32_t baseArrayLayer) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    for (uint32_t layer = 0; layer < layerCount; ++layer) {
        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;
        uint32_t targetLayer = baseArrayLayer + layer;

        for (uint32_t i = 1; i < mipLevels; i++) {
            // transition current mip level i to TRANSFER_DST_OPTIMAL from UNDEFINED
            barrier.subresourceRange.baseMipLevel = i;
            barrier.subresourceRange.baseArrayLayer = targetLayer;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            vkCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                0,
                                0, nullptr,
                                0, nullptr,
                                1, &barrier);

            // now transition previous level (i-1) to TRANSFER_SRC_OPTIMAL
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.subresourceRange.baseArrayLayer = targetLayer;

            vkCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                0,
                                0, nullptr,
                                0, nullptr,
                                1, &barrier);

            VkImageBlit blit{};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = targetLayer;
            blit.srcSubresource.layerCount = 1;

            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = { std::max(1, mipWidth / 2), std::max(1, mipHeight / 2), 1 };
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = targetLayer;
            blit.dstSubresource.layerCount = 1;

            vkCmdBlitImage(commandBuffer,
                        image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit,
                        VK_FILTER_LINEAR);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.subresourceRange.baseArrayLayer = targetLayer;

            vkCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                0,
                                0, nullptr,
                                0, nullptr,
                                1, &barrier);

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        // Transition last mip level for this layer to SHADER_READ_ONLY_OPTIMAL
        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.subresourceRange.baseArrayLayer = targetLayer;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            0,
                            0, nullptr,
                            0, nullptr,
                            1, &barrier);
    }
}
// Process any pending command buffers (free command buffers and fences when fence signaled)
void VulkanApp::processPendingCommandBuffers() {
    std::lock_guard<std::mutex> lk(pendingCmdMutex);
    for (auto it = pendingCommandBuffers.begin(); it != pendingCommandBuffers.end(); ) {
        VkCommandBuffer cmd = it->first;
        VkFence fence = it->second;
        VkResult st = vkGetFenceStatus(device, fence);
        if (st == VK_SUCCESS) {
            // free command buffer and destroy fence
            vkFreeCommandBuffers(device, commandPool, 1, &cmd);
            vkDestroyFence(device, fence, nullptr);
            it = pendingCommandBuffers.erase(it);
        } else if (st == VK_ERROR_DEVICE_LOST) {
            // Something bad happened; free resources conservatively
            vkFreeCommandBuffers(device, commandPool, 1, &cmd);
            vkDestroyFence(device, fence, nullptr);
            it = pendingCommandBuffers.erase(it);
        } else {
            ++it;
        }
    }
}

// Submit a pre-recorded command buffer asynchronously and return a fence that will be signaled on completion.
VkFence VulkanApp::submitCommandBufferAsync(VkCommandBuffer commandBuffer, VkSemaphore* outSemaphore) {
    // End command buffer here (caller recorded commands assumed)
    vkEndCommandBuffer(commandBuffer);

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("failed to create fence for async submit");
    }

    VkSemaphore semaphore = VK_NULL_HANDLE;
    if (outSemaphore) {
        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(device, &semInfo, nullptr, &semaphore) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            throw std::runtime_error("failed to create semaphore for async submit");
        }
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    if (semaphore != VK_NULL_HANDLE) {
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &semaphore;
    }

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
        if (semaphore != VK_NULL_HANDLE) vkDestroySemaphore(device, semaphore, nullptr);
        vkDestroyFence(device, fence, nullptr);
        throw std::runtime_error("failed to submit async command buffer");
    }

    // Track command buffer+fence ownership so we can free command buffers later when fence signals
    {
        std::lock_guard<std::mutex> lk(pendingCmdMutex);
        pendingCommandBuffers.emplace_back(commandBuffer, fence);
    }

    if (semaphore != VK_NULL_HANDLE && outSemaphore) {
        *outSemaphore = semaphore;
        // register the semaphore so drawFrame will wait on it and later clean it up
        std::lock_guard<std::mutex> lk(extraSemaphoreMutex);
        extraWaitSemaphores.push_back(semaphore);
    }

    return fence;
}

void VulkanApp::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, uint32_t arrayLayers) {
    if (image == VK_NULL_HANDLE) {
        throw std::runtime_error("transitionImageLayout called with VK_NULL_HANDLE image!");
    }
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = arrayLayers;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    endSingleTimeCommands(commandBuffer);
}

void VulkanApp::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = { width, height, 1 };

    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(commandBuffer);
}

std::vector<VkCommandBuffer> VulkanApp::createCommandBuffers() {
    commandBuffers.clear();
    commandBuffers.resize(swapchainFramebuffers.size());
    std::cerr << "[DEBUG] createCommandBuffers: requested " << commandBuffers.size() << " command buffers, swapchainFramebuffers=" << swapchainFramebuffers.size() << "\n";
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    VkResult allocResult = vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
    std::cerr << "[DEBUG] vkAllocateCommandBuffers result=" << allocResult << "\n";
    if (allocResult != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
    std::cerr << "[DEBUG] createCommandBuffers: post-alloc commandBuffers.size()=" << commandBuffers.size() << "\n";

    return commandBuffers;
}

void VulkanApp::createSyncObjects() {
    // Create semaphores per swapchain image to avoid reuse before presentation completes.
    // Fences are per frame-in-flight for CPU-GPU synchronization.
    const uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    const uint32_t numImages = static_cast<uint32_t>(swapchainImages.size());

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    // Semaphores per swapchain image (not per frame-in-flight)
    imageAvailableSemaphores.resize(numImages);
    renderFinishedSemaphores.resize(numImages);
    
    for (uint32_t i = 0; i < numImages; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create semaphores for image " + std::to_string(i));
        }
    }
    
    // Fences per frame-in-flight
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create fence for frame " + std::to_string(i));
        }
    }

    // imagesInFlight tracks which fence is using each swapchain image (initialized null)
    imagesInFlight.clear();
    imagesInFlight.resize(numImages, VK_NULL_HANDLE);

    // async submission bookkeeping
    pendingCommandBuffers.clear();
    semaphoresPendingDestroy.clear();
    extraWaitSemaphores.clear();
}

TextureImage VulkanApp::createTextureImageArray(const std::vector<std::string>& filenames, bool srgb) {
    TextureImage textureImage;
    if (filenames.empty()) throw std::runtime_error("createTextureImageArray: empty filename list");

    int texWidth = 0, texHeight = 0, texChannels = 0;
    std::vector<unsigned char*> layersData;
    layersData.reserve(filenames.size());

    for (size_t i = 0; i < filenames.size(); ++i) {
        unsigned char* pixels = stbi_load(filenames[i].c_str(), &texWidth, &texHeight, &texChannels, 4);
        if (!pixels) {
            // free any previously loaded
            for (auto p : layersData) if (p) stbi_image_free(p);
            throw std::runtime_error(std::string("failed to load texture image: ") + filenames[i]);
        }
        layersData.push_back(pixels);
    }

    const uint32_t layerCount = static_cast<uint32_t>(layersData.size());
    VkDeviceSize layerSize = texWidth * texHeight * 4;
    VkDeviceSize imageSize = layerSize * layerCount;

    // If the caller requested sRGB handling, convert loaded sRGB data to linear before storing as UNORM
    if (srgb) {
        for (uint32_t i = 0; i < layerCount; ++i) {
            convertSRGB8ToLinearInPlace(layersData[i], static_cast<size_t>(texWidth) * static_cast<size_t>(texHeight));
        }
    }

    // create staging buffer containing all layers consecutively
    Buffer stagingBuffer = createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* data;
    vkMapMemory(device, stagingBuffer.memory, 0, imageSize, 0, &data);
    for (uint32_t i = 0; i < layerCount; ++i) {
        memcpy((unsigned char*)data + layerSize * i, layersData[i], layerSize);
    }
    vkUnmapMemory(device, stagingBuffer.memory);

    for (auto p : layersData) stbi_image_free(p);

    textureImage.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    // choose format: use UNORM for array textures
    VkFormat chosenFormat = VK_FORMAT_R8G8B8A8_UNORM;

    // create image with arrayLayers = layerCount
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = 0;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = static_cast<uint32_t>(texWidth);
    imageInfo.extent.height = static_cast<uint32_t>(texHeight);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = textureImage.mipLevels;
    imageInfo.arrayLayers = layerCount;
    imageInfo.format = chosenFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // need transfer src/dst for mipmap generation and sampled for shader access
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(device, &imageInfo, nullptr, &textureImage.image) != VK_SUCCESS) {
        vkDestroyBuffer(device, stagingBuffer.buffer, nullptr);
        vkFreeMemory(device, stagingBuffer.memory, nullptr);
        throw std::runtime_error("failed to create texture array image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, textureImage.image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &textureImage.memory) != VK_SUCCESS) {
        vkDestroyImage(device, textureImage.image, nullptr);
        vkDestroyBuffer(device, stagingBuffer.buffer, nullptr);
        vkFreeMemory(device, stagingBuffer.memory, nullptr);
        throw std::runtime_error("failed to allocate texture image memory!");
    }

    vkBindImageMemory(device, textureImage.image, textureImage.memory, 0);

    // copy buffer to image per-layer
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    // transition entire image to TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = textureImage.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = layerCount;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    std::vector<VkBufferImageCopy> regions(layerCount);
    for (uint32_t i = 0; i < layerCount; ++i) {
        VkBufferImageCopy region{};
        region.bufferOffset = layerSize * i;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = i;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0,0,0};
        region.imageExtent = { static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1 };
        regions[i] = region;
    }

    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer.buffer, textureImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(regions.size()), regions.data());

    // transition to SHADER_READ_ONLY_OPTIMAL
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(commandBuffer);

    // generate mipmaps for the array texture (per-layer)
    generateMipmaps(textureImage.image, chosenFormat, texWidth, texHeight, textureImage.mipLevels, layerCount);

    vkDestroyBuffer(device, stagingBuffer.buffer, nullptr);
    vkFreeMemory(device, stagingBuffer.memory, nullptr);

    // create view for array texture
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = textureImage.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = chosenFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = textureImage.mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = layerCount;

    if (vkCreateImageView(device, &viewInfo, nullptr, &textureImage.view) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view (array)!");
    }

    return textureImage;
}

void VulkanApp::createTextureImageView(TextureImage &textureImage) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = textureImage.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    // match the image format (use UNORM view)
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = textureImage.mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &textureImage.view) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }
}

VkSampler VulkanApp::createTextureSampler(uint32_t mipLevels) {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    // enable anisotropic filtering when supported by the device
    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    if (deviceProperties.limits.maxSamplerAnisotropy > 1.0f) {
        samplerInfo.anisotropyEnable = VK_TRUE;
        // clamp requested anisotropy to device maximum
        float desiredAniso = std::min<float>(16.0f, deviceProperties.limits.maxSamplerAnisotropy);
        samplerInfo.maxAnisotropy = desiredAniso;
    } else {
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;
    }
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = (float)mipLevels;
    samplerInfo.mipLodBias = 0.0f;
    VkSampler textureSampler = VK_NULL_HANDLE;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
    return textureSampler;
}

void VulkanApp::createDescriptorSetLayout() {
    // binding 0 : uniform buffer (vertex shader)
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    // UBO is referenced by vertex, fragment and tessellation stages
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    // bindings 1..3: arrays of combined image samplers (albedo / normal / height)
    // bindings 1..3: one combined image sampler each (we use a texture2D array as the image view)
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    VkDescriptorSetLayoutBinding normalSamplerBinding{};
    normalSamplerBinding.binding = 2;
    normalSamplerBinding.descriptorCount = 1;
    normalSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    normalSamplerBinding.pImmutableSamplers = nullptr;
    normalSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding heightSamplerBinding{};
    heightSamplerBinding.binding = 3;
    heightSamplerBinding.descriptorCount = 1;
    heightSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    heightSamplerBinding.pImmutableSamplers = nullptr;
    // Height sampler is used by fragment shader and tessellation evaluation shader (for displacement)
    heightSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    // binding 4: shadow map sampler
    VkDescriptorSetLayoutBinding shadowSamplerBinding{};
    shadowSamplerBinding.binding = 4;
    shadowSamplerBinding.descriptorCount = 1;
    shadowSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowSamplerBinding.pImmutableSamplers = nullptr;
    shadowSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 6: Sky UBO
    VkDescriptorSetLayoutBinding skyBinding{};
    skyBinding.binding = 6;
    skyBinding.descriptorCount = 1;
    skyBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    skyBinding.pImmutableSamplers = nullptr;
    skyBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    
    // binding 7: Water params UBO (for water shader)
    VkDescriptorSetLayoutBinding waterParamsBinding{};
    waterParamsBinding.binding = 7;
    waterParamsBinding.descriptorCount = 1;
    waterParamsBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    waterParamsBinding.pImmutableSamplers = nullptr;
    // Make the water params visible to both fragment and tessellation evaluation shaders
    waterParamsBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    // Per-instance / per-draw descriptor set uses bindings: 0 (UBO), 1..3 (samplers), 4 (shadow), 5 (Materials SSBO), 6 (Sky UBO), 7 (water params), 8 (Models SSBO)
    // Note: Materials (binding 5) is declared in shaders as set=0 binding=5, so include it in the main layout.
    std::array<VkDescriptorSetLayoutBinding, 9> bindings = {uboLayoutBinding, samplerLayoutBinding, normalSamplerBinding, heightSamplerBinding, shadowSamplerBinding, /* material */ VkDescriptorSetLayoutBinding{}, skyBinding, waterParamsBinding, /* models */ VkDescriptorSetLayoutBinding{}};
    // Fill the material binding at position 5
    bindings[5].binding = 5;
    bindings[5].descriptorCount = 1;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].pImmutableSamplers = nullptr;
    bindings[5].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    
    // Fill the models SSBO binding at position 8
    bindings[8].binding = 8;
    bindings[8].descriptorCount = 1;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[8].pImmutableSamplers = nullptr;
    bindings[8].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    // Allocate the main UBO/sampler/materials descriptor set and store it
    mainDescriptorSet = createDescriptorSet(descriptorSetLayout);

    // Create a separate material descriptor layout used for models (different binding to avoid colliding with sky UBO)
    VkDescriptorSetLayoutBinding modelsBinding{};
    modelsBinding.binding = 8; // avoid binding 6 which is used by Sky UBO
    modelsBinding.descriptorCount = 1;
    modelsBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    modelsBinding.pImmutableSamplers = nullptr;
    modelsBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> materialBindings = { bindings[5], modelsBinding };
    VkDescriptorSetLayoutCreateInfo materialLayoutInfo{};
    materialLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    materialLayoutInfo.bindingCount = static_cast<uint32_t>(materialBindings.size());
    materialLayoutInfo.pBindings = materialBindings.data();

    if (vkCreateDescriptorSetLayout(device, &materialLayoutInfo, nullptr, &materialDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create material descriptor set layout!");
    }

    // If we later add a normal map sampler (binding 2), extend bindings dynamically when required by the app.
}

void VulkanApp::createDepthResources() {
    // simple depth resources using a 32-bit float depth format
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = swapchainExtent.width;
    imageInfo.extent.height = swapchainExtent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = depthFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(device, &imageInfo, nullptr, &depthImage) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depth image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, depthImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &depthImageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate depth image memory!");
    }

    vkBindImageMemory(device, depthImage, depthImageMemory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = depthImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = depthFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &depthImageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depth image view!");
    }

    fprintf(stderr, "[VulkanApp] depthImage (swapchain) = %p\n", (void*)depthImage);

    // transition depth image to DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = depthImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    endSingleTimeCommands(commandBuffer);
}

void VulkanApp::createDescriptorPool(uint32_t uboCount, uint32_t samplerCount) {
    // Reserve descriptors: uniform buffers, combined image samplers, and storage buffers for materials
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // Each descriptor set will reference the per-set scene UBO (binding 0)
    // and the shared Sky UBO (binding 6). Reserve two uniform descriptors per set.
    poolSizes[0].descriptorCount = uboCount * 2;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = samplerCount;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    // Increase storage buffer descriptors for compute workloads (was: uboCount)
    poolSizes[2].descriptorCount = uboCount * 8;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    // Allow freeing individual descriptor sets (vegetation, etc.)
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    // Increase maxSets to support many compute/graphics allocations
    poolInfo.maxSets = uboCount * 16;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

VkDescriptorSet VulkanApp::createDescriptorSet(VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    if (layout == VK_NULL_HANDLE) {
        fprintf(stderr, "[VulkanApp::createDescriptorSet] ERROR: requested layout is VK_NULL_HANDLE\n");
        throw std::runtime_error("createDescriptorSet called with VK_NULL_HANDLE layout");
    }

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        fprintf(stderr, "[VulkanApp::createDescriptorSet] vkAllocateDescriptorSets failed for layout=%p\n", (void*)layout);
        throw std::runtime_error("failed to allocate descriptor set!");
    }
    return descriptorSet;
}

VkDescriptorSet VulkanApp::createMaterialDescriptorSet() {
    if (materialDescriptorSetLayout == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    return createDescriptorSet(materialDescriptorSetLayout);
}

void VulkanApp::updateDescriptorSet(VkDescriptorSet &descriptorSet, std::initializer_list<VkWriteDescriptorSet> descriptors) {
    std::vector<VkWriteDescriptorSet> descriptorWrites(descriptors);
    // Debugging: log imageView values before calling into Vulkan
    for (size_t i = 0; i < descriptorWrites.size(); ++i) {
        const VkWriteDescriptorSet &w = descriptorWrites[i];
        if (w.pImageInfo) {
            fprintf(stderr, "[VulkanApp::updateDescriptorSet] write[%zu] binding=%u type=%d imageView=%p sampler=%p\n", i, w.dstBinding, w.descriptorType, (void*)w.pImageInfo[0].imageView, (void*)w.pImageInfo[0].sampler);
        } else if (w.pBufferInfo) {
            fprintf(stderr, "[VulkanApp::updateDescriptorSet] write[%zu] binding=%u type=%d buffer=%p\n", i, w.dstBinding, w.descriptorType, (void*)w.pBufferInfo[0].buffer);
        }
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void VulkanApp::updateDescriptorSet(VkDescriptorSet &descriptorSet, const std::vector<VkWriteDescriptorSet> &descriptors) {
    // Debugging: log imageView values before calling into Vulkan
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const VkWriteDescriptorSet &w = descriptors[i];
        if (w.pImageInfo) {
            fprintf(stderr, "[VulkanApp::updateDescriptorSet] write[%zu] binding=%u type=%d imageView=%p sampler=%p\n", i, w.dstBinding, w.descriptorType, (void*)w.pImageInfo[0].imageView, (void*)w.pImageInfo[0].sampler);
        } else if (w.pBufferInfo) {
            fprintf(stderr, "[VulkanApp::updateDescriptorSet] write[%zu] binding=%u type=%d buffer=%p\n", i, w.dstBinding, w.descriptorType, (void*)w.pBufferInfo[0].buffer);
        }
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptors.size()), descriptors.data(), 0, nullptr);
}

VkShaderModule VulkanApp::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}

VkDevice VulkanApp::getDevice() const {
    return device;
}

VkPipelineLayout VulkanApp::getPipelineLayout() const {
    return pipelineLayout;
}

std::pair<VkPipeline, VkPipelineLayout> VulkanApp::createGraphicsPipeline(
    std::initializer_list<VkPipelineShaderStageCreateInfo> stages,
    const std::vector<VkVertexInputBindingDescription>& bindingDescriptions,
    std::initializer_list<VkVertexInputAttributeDescription> descriptions,
    const std::vector<VkDescriptorSetLayout>& setLayouts,
    const VkPushConstantRange* pushConstantRange,
    VkPolygonMode polygonMode,
    VkCullModeFlagBits cullMode,
    bool depthWrite,
    bool colorWrite,
    VkCompareOp depthCompare,
    VkPrimitiveTopology topology,
    VkRenderPass renderPassOverride) {

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages(stages);
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions(descriptions);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = bindingDescriptions.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = topology;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swapchainExtent.width;
    viewport.height = (float) swapchainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0,0};
    scissor.extent = swapchainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = nullptr;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = polygonMode;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = cullMode;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    if (colorWrite) {
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    } else {
        colorBlendAttachment.colorWriteMask = 0;
    }
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pipelineLayoutInfo.pSetLayouts = setLayouts.empty() ? nullptr : setLayouts.data();
    // Diagnostic: print provided set layouts for easier debugging of descriptor mismatches
    if (!setLayouts.empty()) {
        fprintf(stderr, "[createGraphicsPipeline] setLayouts count=%u\n", static_cast<uint32_t>(setLayouts.size()));
        for (size_t i = 0; i < setLayouts.size(); ++i) {
            fprintf(stderr, "  set[%zu] = %p\n", i, (void*)setLayouts[i]);
        }
    } else {
        fprintf(stderr, "[createGraphicsPipeline] setLayouts count=0\n");
    }
    if (pushConstantRange) {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange;
    } else {
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;
    }

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = depthCompare;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    // If tessellation shaders are present, set input assembly topology to patch list and add tessellation state
    bool hasTessellation = false;
    for (const auto &s : shaderStages) {
        if (s.stage & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)) {
            hasTessellation = true;
            break;
        }
    }
    VkPipelineTessellationStateCreateInfo tessState{};
    if (hasTessellation) {
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        tessState.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
        tessState.patchControlPoints = 3; // triangles
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPassOverride == VK_NULL_HANDLE ? renderPass : renderPassOverride;
    pipelineInfo.subpass = 0;
    if (hasTessellation) pipelineInfo.pTessellationState = &tessState;
    VkPipeline graphicsPipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
    std::cerr << "graphics pipeline created\n";
    registeredPipelines.push_back(graphicsPipeline);
    return {graphicsPipeline, pipelineLayout};
}

uint32_t VulkanApp::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

Buffer VulkanApp::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    Buffer buffer;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer.buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer.buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &buffer.memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0);
    return buffer;
}



void VulkanApp::updateUniformBuffer(Buffer &uniform, void *data, size_t dataSize) {
    // Defensive: ensure memory is valid before mapping
    if (uniform.memory == VK_NULL_HANDLE) {
        std::cerr << "[updateUniformBuffer] Error: attempt to map VK_NULL_HANDLE memory (buffer=" << uniform.buffer << ")" << std::endl;
        return;
    }
    // map the exact size requested by the caller so different uniform sizes are supported
    void* bufferData;
    vkMapMemory(device, uniform.memory, 0, dataSize, 0, &bufferData);
    memcpy(bufferData, data, dataSize);
    vkUnmapMemory(device, uniform.memory);
}

Buffer VulkanApp::createVertexBuffer(const std::vector<Vertex> &vertices) {
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    
    // Create staging buffer (host-visible) to transfer data to GPU
    Buffer stagingBuffer = createBuffer(bufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    
    void* data;
    vkMapMemory(device, stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBuffer.memory);
    
    // Create device-local vertex buffer for fast GPU access
    Buffer vertexBuffer = createBuffer(bufferSize, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Copy from staging to device-local buffer
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy copyRegion{};
    copyRegion.size = bufferSize;
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, vertexBuffer.buffer, 1, &copyRegion);
    endSingleTimeCommands(cmd);
    
    // Clean up staging buffer
    vkDestroyBuffer(device, stagingBuffer.buffer, nullptr);
    vkFreeMemory(device, stagingBuffer.memory, nullptr);
    
    return vertexBuffer;
}

Buffer VulkanApp::createIndexBuffer(const std::vector<uint> &indices) {
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
    
    // Create staging buffer (host-visible) to transfer data to GPU
    Buffer stagingBuffer = createBuffer(bufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    void* data;
    vkMapMemory(device, stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBuffer.memory);
    
    // Create device-local index buffer for fast GPU access
    Buffer indexBuffer = createBuffer(bufferSize, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Copy from staging to device-local buffer
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy copyRegion{};
    copyRegion.size = bufferSize;
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, indexBuffer.buffer, 1, &copyRegion);
    endSingleTimeCommands(cmd);
    
    // Clean up staging buffer
    vkDestroyBuffer(device, stagingBuffer.buffer, nullptr);
    vkFreeMemory(device, stagingBuffer.memory, nullptr);
    
    return indexBuffer;
}

Buffer VulkanApp::createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
    // Create staging buffer (host-visible) to transfer data to GPU
    Buffer stagingBuffer = createBuffer(size, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    void* mapped;
    vkMapMemory(device, stagingBuffer.memory, 0, size, 0, &mapped);
    memcpy(mapped, data, (size_t)size);
    vkUnmapMemory(device, stagingBuffer.memory);
    
    // Create device-local buffer for fast GPU access
    Buffer gpuBuffer = createBuffer(size, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Copy from staging to device-local buffer
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, gpuBuffer.buffer, 1, &copyRegion);
    endSingleTimeCommands(cmd);
    
    // Clean up staging buffer
    vkDestroyBuffer(device, stagingBuffer.buffer, nullptr);
    vkFreeMemory(device, stagingBuffer.memory, nullptr);
    
    return gpuBuffer;
}

void VulkanApp::drawFrame() {
    const uint32_t MAX_FRAMES_IN_FLIGHT = static_cast<uint32_t>(inFlightFences.size());
    const uint32_t numImages = static_cast<uint32_t>(swapchainImages.size());
    uint32_t imageIndex;

    // Use a rotating semaphore index based on an acquire counter to avoid reusing
    // a semaphore that may still be pending from a previous acquire
    static uint32_t acquireIndex = 0;
    uint32_t semaphoreIndex = acquireIndex % numImages;
    acquireIndex++;

    // Acquire next image using per-image semaphore
    VkResult r = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphores[semaphoreIndex], VK_NULL_HANDLE, &imageIndex);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    } else if (r == VK_ERROR_DEVICE_LOST) {
        return;
    } else if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        std::cerr << "vkAcquireNextImageKHR failed: " << r << std::endl;
        return;
    }

    // Wait for the CPU frame fence for the current frame (ensures GPU finished with this frame's resources)
    VkResult waitForFrameFence = vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    if (waitForFrameFence == VK_ERROR_DEVICE_LOST) return;
    if (waitForFrameFence != VK_SUCCESS) {
        std::cerr << "vkWaitForFences failed: " << waitForFrameFence << std::endl;
        return;
    }

    // If a previous frame is using this image and it's a DIFFERENT frame than current,
    // we need to wait for it. With MAX_FRAMES_IN_FLIGHT >= swapchain images, this is rare.
    // Optimization: only wait if the image's fence differs from the one we just waited on.
    if (imagesInFlight.size() > imageIndex && 
        imagesInFlight[imageIndex] != VK_NULL_HANDLE &&
        imagesInFlight[imageIndex] != inFlightFences[currentFrame]) {
        VkResult res = vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        if (res == VK_ERROR_DEVICE_LOST) return;
        if (res != VK_SUCCESS) {
            std::cerr << "vkWaitForFences (image) failed: " << res << std::endl;
            return;
        }
    }

    // Mark this image as now being used by the current frame's fence
    if (imagesInFlight.size() > imageIndex) imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    // reset current frame fence so vkQueueSubmit can signal it
    VkResult resetFenceResult = vkResetFences(device, 1, &inFlightFences[currentFrame]);
    if (resetFenceResult == VK_ERROR_DEVICE_LOST) return;
    if (resetFenceResult != VK_SUCCESS) {
        std::cerr << "vkResetFences failed: " << resetFenceResult << std::endl;
        return;
    }
    // compute deltaTime for this frame
    double frameNow = glfwGetTime();
    float deltaTime = 0.0f;
    if (lastFrameTime > 0.0) deltaTime = static_cast<float>(frameNow - lastFrameTime);
    lastFrameTime = frameNow;
    update(deltaTime);

    // ImGui new frame (backend)
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // call into the derived app to build the UI
    renderImGui();

    // update FPS (simple moving average could be added)
    double now = glfwGetTime();
    if (imguiLastTime > 0.0) {
        double dt = now - imguiLastTime;
        if (dt > 0.0) imguiFps = static_cast<float>(1.0 / dt);
    }
    imguiLastTime = now;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // Wait on the semaphore used for this acquire (indexed by semaphoreIndex)
    std::vector<VkSemaphore> waitSemaphoresVec = { imageAvailableSemaphores[semaphoreIndex] };
    std::vector<VkPipelineStageFlags> waitStagesVec = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

    // Pull extra semaphores signaled by async generation submissions (if any)
    {
        std::lock_guard<std::mutex> lk(extraSemaphoreMutex);
        for (auto &s : extraWaitSemaphores) {
            waitSemaphoresVec.push_back(s);
            waitStagesVec.push_back(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        // Move them to semaphoresPendingDestroy so they can be destroyed after this frame's fence signals
        if (!extraWaitSemaphores.empty()) {
            for (auto &s : extraWaitSemaphores) semaphoresPendingDestroy.emplace_back(s, inFlightFences[currentFrame]);
            extraWaitSemaphores.clear();
        }
    }

    submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphoresVec.size());
    submitInfo.pWaitSemaphores = waitSemaphoresVec.data();
    submitInfo.pWaitDstStageMask = waitStagesVec.data();

    // Copy the wait arrays into local storage so address stays valid while submitInfo references them
    std::vector<VkSemaphore> localWaitSemaphores = std::move(waitSemaphoresVec);
    std::vector<VkPipelineStageFlags> localWaitStages = std::move(waitStagesVec);
    submitInfo.waitSemaphoreCount = static_cast<uint32_t>(localWaitSemaphores.size());
    submitInfo.pWaitSemaphores = localWaitSemaphores.data();
    submitInfo.pWaitDstStageMask = localWaitStages.data();

    // We'll also fill the signal semaphores from renderFinishedSemaphores as before
    VkSemaphore initialSignalSemaphores[] = { renderFinishedSemaphores[semaphoreIndex] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = initialSignalSemaphores;

    // Debug: check commandBuffers size and imageIndex
    //std::cerr << "[DEBUG] commandBuffers.size()=" << commandBuffers.size() << " imageIndex=" << imageIndex << std::endl;
    if (imageIndex >= commandBuffers.size()) {
        std::cerr << "[FATAL] imageIndex out of bounds for commandBuffers!" << std::endl;
        abort();
    }
    VkCommandBuffer &commandBuffer = commandBuffers[imageIndex];
    //std::cerr << "[DEBUG] commandBuffer=" << (void*)commandBuffer << std::endl;
    if (commandBuffer == VK_NULL_HANDLE) {
        std::cerr << "[FATAL] commandBuffer is VK_NULL_HANDLE!" << std::endl;
        abort();
    }
    // reset the command buffer so we can re-record it for this frame
    VkResult resetCmdResult = vkResetCommandBuffer(commandBuffer, 0);
    if (resetCmdResult == VK_ERROR_DEVICE_LOST) {
        return;
    } else if (resetCmdResult != VK_SUCCESS) {
        std::cerr << "vkResetCommandBuffer failed: " << resetCmdResult << std::endl;
        return;
    }

    // Begin recording commands
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult beginCmdResult = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    if (beginCmdResult != VK_SUCCESS) {
        std::cerr << "vkBeginCommandBuffer failed: " << beginCmdResult << std::endl;
        return;
    }

    // Hook for compute/barrier operations before render pass
    preRenderPass(commandBuffer);

    // Process any completed async generation submissions and free their command buffers/fences/semaphores
    processPendingCommandBuffers();

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapchainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapchainExtent;

    VkClearValue clearValues[2] = {};
    // Clear swapchain to transparent black so composites come only from rendered content
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearValues;

    // Begin render pass
    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    ImGui::Render();
    draw(commandBuffer, renderPassInfo);

    // End render pass
    vkCmdEndRenderPass(commandBuffer);

    // End recording commands
    VkResult endCmdResult = vkEndCommandBuffer(commandBuffer);
    if (endCmdResult != VK_SUCCESS) {
        std::cerr << "vkEndCommandBuffer failed: " << endCmdResult << std::endl;
        return;
    }

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    // Signal semaphore indexed by acquired imageIndex (presentation will wait on this)
    VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[imageIndex] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    r = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
    if (r == VK_ERROR_DEVICE_LOST) {
        return;
    } else if (r != VK_SUCCESS) {
        std::cerr << "vkQueueSubmit failed: " << r << std::endl;
        return;
    }

    // Cleanup any semaphores that were associated with earlier frames and are now safe to destroy
    for (auto it = semaphoresPendingDestroy.begin(); it != semaphoresPendingDestroy.end(); ) {
        VkSemaphore s = it->first;
        VkFence f = it->second;
        VkResult st = vkGetFenceStatus(device, f);
        if (st == VK_SUCCESS) {
            // frame fence signaled -> safe to destroy semaphore
            vkDestroySemaphore(device, s, nullptr);
            it = semaphoresPendingDestroy.erase(it);
        } else {
            ++it;
        }
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = { swapchain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    r = vkQueuePresentKHR(presentQueue, &presentInfo);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || framebufferResized || vsyncChanged) {
        framebufferResized = false;
        vsyncChanged = false;
        recreateSwapchain();
        return;
    } else if (r == VK_ERROR_DEVICE_LOST) {
        return;
    } else if (r != VK_SUCCESS) {
        std::cerr << "vkQueuePresentKHR failed: " << r << std::endl;
        return;
    }

    // Process pending command buffers and semaphores cleanup now that present is done
    processPendingCommandBuffers();
    for (auto it = semaphoresPendingDestroy.begin(); it != semaphoresPendingDestroy.end();) {
        VkSemaphore s = it->first; VkFence f = it->second;
        VkResult st = vkGetFenceStatus(device, f);
        if (st == VK_SUCCESS) {
            vkDestroySemaphore(device, s, nullptr);
            it = semaphoresPendingDestroy.erase(it);
        } else ++it;
    }
    //std::cerr << "presented image " << imageIndex << "\n";

    // Hook for derived apps to run post-submit instrumentation (e.g., readback)
    postSubmit();

    // Advance to next CPU frame
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanApp::cleanupSwapchain() {
    // destroy framebuffers
    for (auto fb : swapchainFramebuffers) {
        if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(device, fb, nullptr);
    }
    swapchainFramebuffers.clear();

    // destroy image views
    for (auto iv : swapchainImageViews) {
        if (iv != VK_NULL_HANDLE) vkDestroyImageView(device, iv, nullptr);
    }
    swapchainImageViews.clear();

    // destroy depth resources
    if (depthImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, depthImageView, nullptr);
        depthImageView = VK_NULL_HANDLE;
    }
    if (depthImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, depthImage, nullptr);
        depthImage = VK_NULL_HANDLE;
    }
    if (depthImageMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, depthImageMemory, nullptr);
        depthImageMemory = VK_NULL_HANDLE;
    }

    // free command buffers (they reference old framebuffers)
    if (!commandBuffers.empty()) {
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
        commandBuffers.clear();
    }

    // destroy swapchain
    if (swapchain != VK_NULL_HANDLE) {
        auto fp = (PFN_vkDestroySwapchainKHR)vkGetInstanceProcAddr(instance, "vkDestroySwapchainKHR");
        if (fp) fp(device, swapchain, nullptr);
        else vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
    swapchainImages.clear();
    imagesInFlight.clear();
}

void VulkanApp::recreateSwapchain() {
    int width = 0, height = 0;
    // wait for non-zero size (window might be minimized)
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    VkResult waitResult = vkDeviceWaitIdle(device);
    if (waitResult == VK_ERROR_DEVICE_LOST) {
        return;
    }
    
    // Wait for all in-flight fences before clearing tracking
    for (size_t i = 0; i < inFlightFences.size(); i++) {
        vkWaitForFences(device, 1, &inFlightFences[i], VK_TRUE, UINT64_MAX);
    }
    
    cleanupSwapchain();


    createSwapchain();
    std::cerr << "[DEBUG] swapchainImages.size() after createSwapchain: " << swapchainImages.size() << std::endl;
    createImageViews();
    createDepthResources();
    createFramebuffers();
    std::cerr << "[DEBUG] swapchainFramebuffers.size() after createFramebuffers: " << swapchainFramebuffers.size() << std::endl;
    std::cerr << "[DEBUG] swapchainImages.size() after createSwapchain: " << swapchainImages.size() << std::endl;
    std::cerr << "[DEBUG] swapchainFramebuffers before createCommandBuffers: ";
    for (size_t i = 0; i < swapchainFramebuffers.size(); ++i) {
        std::cerr << (void*)swapchainFramebuffers[i] << " ";
    }
    std::cerr << std::endl;
    createCommandPool();
    std::cerr << "[DEBUG] commandBuffers.size() before createCommandBuffers: " << commandBuffers.size() << std::endl;
    commandBuffers = createCommandBuffers();
    std::cerr << "[DEBUG] commandBuffers.size() after createCommandBuffers: " << commandBuffers.size() << std::endl;

    // Ensure imagesInFlight matches the new swapchain image count
    imagesInFlight.clear();
    imagesInFlight.resize(swapchainImages.size(), VK_NULL_HANDLE);

    // Re-create ImGui Vulkan backend objects so they use the updated swapchain image count.
    // ImGui_ImplVulkan_Init stores internal arrays sized by ImageCount; when the swapchain
    // changes we must reinit the backend (but keep the ImGui context and descriptor pool).
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = instance;
    init_info.PhysicalDevice = physicalDevice;
    init_info.Device = device;
    init_info.QueueFamily = findQueueFamilies(physicalDevice).graphicsFamily.value();
    init_info.Queue = graphicsQueue;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = imguiDescriptorPool;
    init_info.MinImageCount = 2;
    init_info.ImageCount = static_cast<uint32_t>(swapchainImages.size());
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = nullptr;
    init_info.RenderPass = renderPass;


    bool imguiInitOk = ImGui_ImplVulkan_Init(&init_info);
    printf("[ImGui] ImGui_ImplVulkan_Init (recreate) returned %s\n", imguiInitOk ? "true" : "false");
    // Re-upload fonts (creates/destroys temporary upload objects)
    ImGui_ImplVulkan_CreateFontsTexture();
    ImGui_ImplVulkan_DestroyFontsTexture();
    if (!imguiInitOk) {
        printf("[ImGui] ERROR: ImGui_ImplVulkan_Init (recreate) failed!\n");
    }

    // Notify derived app to recreate size-dependent offscreen resources
    onSwapchainResized(static_cast<uint32_t>(width), static_cast<uint32_t>(height));

    framebufferResized = false;
}

void VulkanApp::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(window));
    if (app) {
        app->framebufferResized = true;
    }
}

void VulkanApp::createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("Validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Starter";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // extensions
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

bool VulkanApp::checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool found = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                found = true;
                break;
            }
        }

        if (!found) return false;
    }

    return true;
}

void VulkanApp::setupDebugMessenger() {
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        if (func(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }
}

void VulkanApp::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

QueueFamilyIndices VulkanApp::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.graphicsFamily = i;
                indices.presentFamily = i;
                return indices;
            }
        }
        ++i;
    }

    // If no family has both, find separate
    i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) break;

        ++i;
    }

    return indices;
}

void VulkanApp::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& dev : devices) {
        if (isDeviceSuitable(dev)) {
            physicalDevice = dev;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

bool VulkanApp::isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    return indices.isComplete();
}

void VulkanApp::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // Query supported features and enable non-solid fill (wireframe) if available
    VkPhysicalDeviceFeatures supportedFeatures{};
    vkGetPhysicalDeviceFeatures(physicalDevice, &supportedFeatures);

    VkPhysicalDeviceFeatures deviceFeatures{};
    if (supportedFeatures.fillModeNonSolid) {
        deviceFeatures.fillModeNonSolid = VK_TRUE;
    }
    // Enable geometry shader if supported
    if (supportedFeatures.geometryShader) {
        deviceFeatures.geometryShader = VK_TRUE;
    } else {
        throw std::runtime_error("Selected GPU does not support geometry shaders, but they are required.");
    }
    deviceFeatures.tessellationShader = VK_TRUE;
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    // Enable multi-draw indirect for GPU-driven rendering
    if (supportedFeatures.multiDrawIndirect) {
        deviceFeatures.multiDrawIndirect = VK_TRUE;
    }

    // Enable Vulkan 1.1 shaderDrawParameters for gl_BaseInstanceARB in shaders
    VkPhysicalDeviceVulkan11Features vulkan11Features{};
    vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan11Features.pNext = nullptr;
    vulkan11Features.shaderDrawParameters = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &vulkan11Features;  // Chain Vulkan 1.1 features
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;


    // No extension dependency logic needed
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }


    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    // Create main descriptor pool immediately after device creation
    // (choose reasonable default counts for UBOs and samplers)
    createDescriptorPool(32, 32);

    // retrieve queue handles
    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    // debug: print queue family indices and whether the two queue handles are equal
    std::cerr << "createLogicalDevice: graphicsFamily=" << indices.graphicsFamily.value() << " presentFamily=" << indices.presentFamily.value() << "\n";
    std::cerr << "graphicsQueue handle: " << graphicsQueue << " presentQueue handle: " << presentQueue << "\n";
}

int VulkanApp::getWidth() {
    return swapchainExtent.width;
}

int VulkanApp::getHeight() {
    return swapchainExtent.height;
}

GLFWwindow* VulkanApp::getWindow() {
    return window;
}


void VulkanApp::run() {
    initWindow();
    initVulkan();
    setup();

    mainLoop();
    cleanup();
}