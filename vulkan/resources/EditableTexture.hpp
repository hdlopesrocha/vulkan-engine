#pragma once

#include "../core/vulkan.hpp"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <vector>
#include <cstring>

class EditableTexture {
public:
    EditableTexture() = default;
    
    void init(VulkanApp* app, uint32_t width, uint32_t height, VkFormat format, const char* name);
    
    void cleanup();
    
    // Edit a pixel (RGBA format)
    void setPixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);
    
    // Fill entire texture with a color
    void fill(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);
    
    // Upload changes to GPU
    void updateGPU(VulkanApp* app);
    
    // Render ImGui widget
    void renderImGui();
    VkImage getImage() const;
    uint32_t getWidth() const;
    uint32_t getHeight() const;
    VkDescriptorSet getImGuiDescriptorSet();
    const uint8_t* getPixelData() const;
    
    // Invalidate the ImGui descriptor (e.g., after ImGui pool is recreated).
    // Must be called before ImGui_ImplVulkan_Shutdown() while the old pool
    // is still alive so the descriptor can be properly freed.
    void invalidateImGuiDescriptor();
    
private:
    uint32_t width = 0;
    uint32_t height = 0;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t bytesPerPixel = 4;
    std::string name = "Editable Texture";
    
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSet imguiDescSet = VK_NULL_HANDLE;
    
    std::vector<uint8_t> cpuData;
    bool isDirty = false;
    
    void createImGuiDescriptor();
};
