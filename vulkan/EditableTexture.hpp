#pragma once

#include "vulkan.hpp"
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
    
    // Edit a pixel (single channel format)
    void setPixelGray(uint32_t x, uint32_t y, uint8_t value);
    
    // Fill entire texture with a color
    void fill(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);
    
    // Upload changes to GPU
    void updateGPU();
    
    // Render ImGui widget
    void renderImGui();
    
    VkImageView getView() const;
    VkSampler getSampler() const;
    VkImage getImage() const;
    bool getDirty() const;
    uint32_t getWidth() const;
    uint32_t getHeight() const;
    uint32_t getBytesPerPixel() const;
    VkDescriptorSet getImGuiDescriptorSet() const;
    const uint8_t* getPixelData() const;
    
    // Get as TextureImage for TextureManager compatibility
    TextureImage getTextureImage() const;
    
private:
    VulkanApp* app = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t bytesPerPixel = 4;
    std::string name = "Editable Texture";
    
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSet imguiDescSet = VK_NULL_HANDLE;
    
    std::vector<uint8_t> cpuData;
    bool isDirty = false;
    
    void createImage();
    
    void createImageView();
    
    void createSampler();
    
    void createImGuiDescriptor();
    
    void transitionImageLayout(VkImageLayout oldLayout, VkImageLayout newLayout);
    
    void copyBufferToImage(VkBuffer buffer);
};
