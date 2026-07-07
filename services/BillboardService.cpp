#include "BillboardService.hpp"
#include "../vulkan/VulkanApp.hpp"
#include <cstdio>
#include <iostream>
#include <stdexcept>

BillboardService::BillboardService() {}

void BillboardService::init(VulkanApp* app) {
    vulkanApp = app;
}

void BillboardService::initializeTextures() {
    if (!vulkanApp || texturesInitialized) return;
    texturesInitialized = true;
}

void BillboardService::cleanup() {
    printf("[BillboardService] cleanup start: texturesInitialized=%d\n", texturesInitialized ? 1 : 0);
    if (texturesInitialized && vulkanApp) {
        VkDevice device = vulkanApp->getDevice();
        auto destroyAR = [&](VkImage& img, VmaAllocation& alloc, VkDeviceMemory& mem, VkImageView& view) {
            if (view != VK_NULL_HANDLE) { vulkanApp->resources.removeImageView(view); vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
            if (img  != VK_NULL_HANDLE) { vulkanApp->destroyImageWithVma(img, alloc, mem); img = VK_NULL_HANDLE; alloc = VK_NULL_HANDLE; mem = VK_NULL_HANDLE; }
        };
        destroyAR(billboardAlbedoArrayImage,  billboardAlbedoArrayAllocation,  billboardAlbedoArrayMemory,  billboardAlbedoArrayView);
        destroyAR(billboardNormalArrayImage,   billboardNormalArrayAllocation,  billboardNormalArrayMemory,  billboardNormalArrayView);
        destroyAR(billboardOpacityArrayImage,  billboardOpacityArrayAllocation, billboardOpacityArrayMemory, billboardOpacityArrayView);
        if (billboardArraySampler != VK_NULL_HANDLE) {
            vulkanApp->resources.removeSampler(billboardArraySampler);
            vkDestroySampler(device, billboardArraySampler, nullptr);
            billboardArraySampler = VK_NULL_HANDLE;
        }
        texturesInitialized = false;
    }
    printf("[BillboardService] cleanup done\n");
}

void BillboardService::invalidateImGuiDescriptors() {
}

void BillboardService::createBillboardArrayTextures() {
    if (!vulkanApp || !texturesInitialized) return;

    VkDevice device = vulkanApp->getDevice();

    auto destroyArrayResources = [&](VkImage& img, VmaAllocation& alloc, VkDeviceMemory& mem, VkImageView& view) {
        if (view != VK_NULL_HANDLE) {
            vulkanApp->resources.removeImageView(view);
            vkDestroyImageView(device, view, nullptr);
            view = VK_NULL_HANDLE;
        }
        if (img != VK_NULL_HANDLE) {
            vulkanApp->destroyImageWithVma(img, alloc, mem);
            img = VK_NULL_HANDLE;
        }
    };
    destroyArrayResources(billboardAlbedoArrayImage, billboardAlbedoArrayAllocation, billboardAlbedoArrayMemory, billboardAlbedoArrayView);
    destroyArrayResources(billboardNormalArrayImage, billboardNormalArrayAllocation, billboardNormalArrayMemory, billboardNormalArrayView);
    destroyArrayResources(billboardOpacityArrayImage, billboardOpacityArrayAllocation, billboardOpacityArrayMemory, billboardOpacityArrayView);
    if (billboardArraySampler != VK_NULL_HANDLE) {
        vulkanApp->resources.removeSampler(billboardArraySampler);
        vkDestroySampler(device, billboardArraySampler, nullptr);
        billboardArraySampler = VK_NULL_HANDLE;
    }
}

void BillboardService::bakeFromTextures(const std::array<EditableTexture, 3>& composedAlbedo,
                                         const std::array<EditableTexture, 3>& composedNormal,
                                         const std::array<EditableTexture, 3>& composedOpacity) {
    if (!vulkanApp || !texturesInitialized) return;

    VkDevice device = vulkanApp->getDevice();
    const uint32_t numLayers = static_cast<uint32_t>(composedAlbedo.size());
    const uint32_t w = composedAlbedo[0].getWidth();
    const uint32_t h = composedAlbedo[0].getHeight();
    const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;

    // Destroy any previously created array resources
    auto destroyArrayResources = [&](VkImage& img, VmaAllocation& alloc, VkDeviceMemory& mem, VkImageView& view) {
        if (view != VK_NULL_HANDLE) {
            vulkanApp->resources.removeImageView(view);
            vkDestroyImageView(device, view, nullptr);
            view = VK_NULL_HANDLE;
        }
        if (img != VK_NULL_HANDLE) {
            vulkanApp->destroyImageWithVma(img, alloc, mem);
            img = VK_NULL_HANDLE;
        }
    };
    destroyArrayResources(billboardAlbedoArrayImage, billboardAlbedoArrayAllocation, billboardAlbedoArrayMemory, billboardAlbedoArrayView);
    destroyArrayResources(billboardNormalArrayImage, billboardNormalArrayAllocation, billboardNormalArrayMemory, billboardNormalArrayView);
    destroyArrayResources(billboardOpacityArrayImage, billboardOpacityArrayAllocation, billboardOpacityArrayMemory, billboardOpacityArrayView);
    if (billboardArraySampler != VK_NULL_HANDLE) {
        vulkanApp->resources.removeSampler(billboardArraySampler);
        vkDestroySampler(device, billboardArraySampler, nullptr);
        billboardArraySampler = VK_NULL_HANDLE;
    }

    uint32_t mipCount = static_cast<uint32_t>(std::floor(std::log2(std::max(w, h)))) + 1;

    auto makeArrayImage = [&](VkImage& outImg, VmaAllocation& outAlloc, VkDeviceMemory& outMem, VkImageView& outView, const char* name) {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.extent = { w, h, 1 };
        imgInfo.mipLevels = mipCount;
        imgInfo.arrayLayers = numLayers;
        imgInfo.format = fmt;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;

        vulkanApp->createImageWithVma(imgInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outImg, outAlloc, outMem, name);

        vulkanApp->transitionImageLayout(outImg, fmt, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipCount, numLayers);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = outImg;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        viewInfo.format = fmt;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipCount;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = numLayers;
        if (vkCreateImageView(device, &viewInfo, nullptr, &outView) != VK_SUCCESS)
            throw std::runtime_error(std::string("BillboardService: failed to create array image view: ") + name);
        vulkanApp->resources.addImageView(outView, name);
    };

    makeArrayImage(billboardAlbedoArrayImage,  billboardAlbedoArrayAllocation,  billboardAlbedoArrayMemory,  billboardAlbedoArrayView,  "BillboardService: albedoArray");
    makeArrayImage(billboardNormalArrayImage,   billboardNormalArrayAllocation,  billboardNormalArrayMemory,  billboardNormalArrayView,  "BillboardService: normalArray");
    makeArrayImage(billboardOpacityArrayImage,  billboardOpacityArrayAllocation, billboardOpacityArrayMemory, billboardOpacityArrayView, "BillboardService: opacityArray");

    struct ChannelEntry { const std::array<EditableTexture, 3>* src; VkImage dst; };
    ChannelEntry channels[3] = {
        { &composedAlbedo,  billboardAlbedoArrayImage  },
        { &composedNormal,  billboardNormalArrayImage  },
        { &composedOpacity, billboardOpacityArrayImage },
    };
    for (auto& ch : channels) {
        for (uint32_t layer = 0; layer < numLayers; ++layer) {
            const EditableTexture& srcTex = (*ch.src)[layer];
            VkImage srcImage = srcTex.getImage();
            vulkanApp->runSingleTimeCommands([&](VkCommandBuffer cmd) {
                vulkanApp->recordTransitionImageLayoutLayer(cmd, srcImage, fmt,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 0, 1);
                VkImageCopy region{};
                region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                region.srcOffset = {0,0,0};
                region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1 };
                region.dstOffset = {0,0,0};
                region.extent = { w, h, 1 };
                vkCmdCopyImage(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               ch.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
                vulkanApp->recordTransitionImageLayoutLayer(cmd, srcImage, fmt,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
            });
        }
        // Keep array images in TRANSFER_DST for mipmap generation
    }

    // Generate mipmaps for each layer of each array
    for (auto& ch : channels) {
        if (mipCount > 1) {
            vulkanApp->generateMipmaps(ch.dst, fmt, static_cast<int32_t>(w), static_cast<int32_t>(h), mipCount, numLayers, 0);
        } else {
            vulkanApp->transitionImageLayout(ch.dst, fmt, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipCount, numLayers);
        }
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = (float)mipCount;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &billboardArraySampler) != VK_SUCCESS)
        throw std::runtime_error("BillboardService: failed to create array sampler");
    vulkanApp->resources.addSampler(billboardArraySampler, "BillboardService: arraysampler");
    std::cerr << "[BillboardService] billboard array textures created: albedo=" << (void*)billboardAlbedoArrayView
              << " normal=" << (void*)billboardNormalArrayView
              << " opacity=" << (void*)billboardOpacityArrayView << std::endl;
}
