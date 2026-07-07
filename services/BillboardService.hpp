#pragma once

#include "Service.hpp"
#include "../vulkan/EditableTexture.hpp"
#include <vulkan/vulkan.h>
#include <array>

class VulkanApp;

class BillboardService : public Service {
public:
    BillboardService();
    void init(VulkanApp* app) override;
    void cleanup() override;

    void initializeTextures();
    void invalidateImGuiDescriptors();
    void createBillboardArrayTextures();

    // Bake billboard composed textures into array textures
    void bakeFromTextures(const std::array<EditableTexture, 3>& composedAlbedo,
                          const std::array<EditableTexture, 3>& composedNormal,
                          const std::array<EditableTexture, 3>& composedOpacity);

    VkImageView getAlbedoArrayView() const { return billboardAlbedoArrayView; }
    VkImageView getNormalArrayView()  const { return billboardNormalArrayView; }
    VkImageView getOpacityArrayView() const { return billboardOpacityArrayView; }
    VkSampler   getArraySampler()     const { return billboardArraySampler; }

    bool isInitialized() const { return texturesInitialized; }

private:
    VulkanApp* vulkanApp = nullptr;
    bool texturesInitialized = false;

    // Billboard array images (sampler2DArray, one layer per billboard)
    VkImage        billboardAlbedoArrayImage   = VK_NULL_HANDLE;
    VmaAllocation  billboardAlbedoArrayAllocation  = VK_NULL_HANDLE;
    VkDeviceMemory billboardAlbedoArrayMemory  = VK_NULL_HANDLE;
    VkImageView    billboardAlbedoArrayView    = VK_NULL_HANDLE;
    VkImage        billboardNormalArrayImage   = VK_NULL_HANDLE;
    VmaAllocation  billboardNormalArrayAllocation  = VK_NULL_HANDLE;
    VkDeviceMemory billboardNormalArrayMemory  = VK_NULL_HANDLE;
    VkImageView    billboardNormalArrayView    = VK_NULL_HANDLE;
    VkImage        billboardOpacityArrayImage  = VK_NULL_HANDLE;
    VmaAllocation  billboardOpacityArrayAllocation = VK_NULL_HANDLE;
    VkDeviceMemory billboardOpacityArrayMemory = VK_NULL_HANDLE;
    VkImageView    billboardOpacityArrayView   = VK_NULL_HANDLE;
    VkSampler      billboardArraySampler       = VK_NULL_HANDLE;
};
