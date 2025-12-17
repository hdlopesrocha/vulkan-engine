#include "TextureManager.hpp"
#include "VulkanApp.hpp"
#include <stdexcept>
#include <backends/imgui_impl_vulkan.h>
#include <cstdlib>

size_t TextureManager::loadTriple(const std::string &albedoFile, const std::string &normalFile, const std::string &heightFile) {
    if (!app) throw std::runtime_error("TextureManager not initialized with VulkanApp pointer");

    // Collect filenames for global arrays
    albedoFiles.push_back(albedoFile);
    normalFiles.push_back(normalFile);
    heightFiles.push_back(heightFile);

    Triple t;
    t.layerIndex = triples.size(); // Will be set after push, but for now use size
    // create array textures with a single layer each (for ImGui compatibility)
    t.albedo = app->createTextureImageArray({ albedoFile }, true);
    t.albedoSampler = app->createTextureSampler(t.albedo.mipLevels);
    // normal maps must be linear (UNORM) — do not use sRGB conversion
    t.normal = app->createTextureImageArray({ normalFile }, false);
    t.normalSampler = app->createTextureSampler(t.normal.mipLevels);
    // height maps also use linear sampling
    t.height = app->createTextureImageArray({ heightFile }, false);
    t.heightSampler = app->createTextureSampler(t.height.mipLevels);

    // Initialize material properties with reasonable defaults
    t.material.ambientFactor = 0.15f;
    t.material.specularStrength = 0.5f;
    t.material.shininess = 32.0f;

    triples.push_back(t);
    triples.back().layerIndex = triples.size() - 1; // Set correct index
    return triples.size() - 1;
}

size_t TextureManager::addTriple(const TextureImage& albedo, VkSampler albedoSampler,
                                  const TextureImage& normal, VkSampler normalSampler,
                                  const TextureImage& height, VkSampler heightSampler,
                                  uint32_t w, uint32_t h, uint32_t bpp,
                                  const uint8_t* albedoPixels, const uint8_t* normalPixels, const uint8_t* heightPixels) {
    Triple t;
    t.albedo = albedo;
    t.albedoSampler = albedoSampler;
    t.normal = normal;
    t.normalSampler = normalSampler;
    t.height = height;
    t.heightSampler = heightSampler;
    t.ownsResources = false;  // Externally managed resources

    // capture CPU-side pixel data if provided so we can rebuild global arrays later
    if (w > 0 && h > 0) {
        t.pixelWidth = w;
        t.pixelHeight = h;
        t.bytesPerPixel = bpp > 0 ? bpp : 4;
        if (albedoPixels) {
            t.hasCpuAlbedo = true;
            t.cpuAlbedo.assign(albedoPixels, albedoPixels + (size_t)w * h * t.bytesPerPixel);
        }
        if (normalPixels) {
            t.hasCpuNormal = true;
            t.cpuNormal.assign(normalPixels, normalPixels + (size_t)w * h * t.bytesPerPixel);
        }
        if (heightPixels) {
            t.hasCpuHeight = true;
            t.cpuHeight.assign(heightPixels, heightPixels + (size_t)w * h * t.bytesPerPixel);
        }
    }

    // Initialize material properties with reasonable defaults
    t.material.ambientFactor = 0.15f;
    t.material.specularStrength = 0.5f;
    t.material.shininess = 32.0f;

    triples.push_back(t);
    triples.back().layerIndex = triples.size() - 1;
    return triples.size() - 1;
}


void TextureManager::createGlobalArrays() {
    if (!app) throw std::runtime_error("TextureManager not initialized");
    const size_t layerCount = triples.size();
    if (layerCount == 0) return;

    // Helper lambda to build an array texture from per-layer CPU data or filenames
    auto buildArray = [&](bool srgb, const std::vector<std::string>& fileList, std::function<const uint8_t*(size_t,int&,int&,int&)> cpuGetter, TextureImage &outImage, VkSampler &outSampler) {
        std::vector<unsigned char*> stbiLoaded;
        std::vector<const unsigned char*> layerPtrs(layerCount, nullptr);

        int texW = 0, texH = 0, texChannels = 0;
        size_t fileIdx = 0;

        for (size_t i = 0; i < layerCount; ++i) {
            const Triple &t = triples[i];
            if (cpuGetter) {
                int w=0,h=0,ch=0;
                const uint8_t* cpu = cpuGetter(i, w, h, ch);
                if (cpu) {
                    if (texW == 0) { texW = w; texH = h; texChannels = ch; }
                    if (w != texW || h != texH) {
                        for (auto p : stbiLoaded) if (p) stbi_image_free(p);
                        throw std::runtime_error("TextureManager::createGlobalArrays - layer size mismatch");
                    }
                    layerPtrs[i] = (const unsigned char*)cpu;
                    continue;
                }
            }

            if (fileIdx >= fileList.size()) {
                for (auto p : stbiLoaded) if (p) stbi_image_free(p);
                throw std::runtime_error("TextureManager::createGlobalArrays - missing filename for layer");
            }
            unsigned char* pixels = stbi_load(fileList[fileIdx].c_str(), &texW, &texH, &texChannels, 4);
            if (!pixels) {
                for (auto p : stbiLoaded) if (p) stbi_image_free(p);
                throw std::runtime_error(std::string("failed to load texture image: ") + fileList[fileIdx]);
            }
            stbiLoaded.push_back(pixels);
            layerPtrs[i] = pixels;
            fileIdx++;
        }

        VkDeviceSize layerSize = (VkDeviceSize)texW * texH * 4;
        VkDeviceSize imageSize = layerSize * layerCount;
        Buffer staging = app->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* data;
        vkMapMemory(app->getDevice(), staging.memory, 0, imageSize, 0, &data);
        for (size_t i = 0; i < layerCount; ++i) {
            memcpy((unsigned char*)data + layerSize * i, layerPtrs[i], (size_t)layerSize);
        }
        vkUnmapMemory(app->getDevice(), staging.memory);

        // Create image with arrayLayers = layerCount and compute mipLevels
        outImage.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texW, texH)))) + 1;
        VkFormat chosenFormat = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.flags = 0;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = static_cast<uint32_t>(texW);
        imageInfo.extent.height = static_cast<uint32_t>(texH);
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = outImage.mipLevels;
        imageInfo.arrayLayers = static_cast<uint32_t>(layerCount);
        imageInfo.format = chosenFormat;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

        VkDevice device = app->getDevice();
        if (vkCreateImage(device, &imageInfo, nullptr, &outImage.image) != VK_SUCCESS) {
            vkDestroyBuffer(device, staging.buffer, nullptr);
            vkFreeMemory(device, staging.memory, nullptr);
            for (auto p : stbiLoaded) if (p) stbi_image_free(p);
            throw std::runtime_error("failed to create texture array image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, outImage.image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = app->findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &outImage.memory) != VK_SUCCESS) {
            vkDestroyImage(device, outImage.image, nullptr);
            vkDestroyBuffer(device, staging.buffer, nullptr);
            vkFreeMemory(device, staging.memory, nullptr);
            for (auto p : stbiLoaded) if (p) stbi_image_free(p);
            throw std::runtime_error("failed to allocate texture image memory!");
        }

        vkBindImageMemory(device, outImage.image, outImage.memory, 0);

        VkCommandBuffer commandBuffer = app->beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = outImage.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = static_cast<uint32_t>(layerCount);
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
            region.imageExtent = { static_cast<uint32_t>(texW), static_cast<uint32_t>(texH), 1 };
            regions[i] = region;
        }

        vkCmdCopyBufferToImage(commandBuffer, staging.buffer, outImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(regions.size()), regions.data());

        // finish copy command buffer; leave image in TRANSFER_DST_OPTIMAL so mip generation
        // can transition and blit correctly per-layer.
        app->endSingleTimeCommands(commandBuffer);

        // generate mipmaps for the array texture (per-layer) using public wrapper
        app->generateMipmapsForImage(outImage.image, chosenFormat, texW, texH, outImage.mipLevels, static_cast<uint32_t>(layerCount));

        vkDestroyBuffer(device, staging.buffer, nullptr);
        vkFreeMemory(device, staging.memory, nullptr);

        // create view for array texture
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = outImage.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        viewInfo.format = chosenFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = outImage.mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = static_cast<uint32_t>(layerCount);

        if (vkCreateImageView(device, &viewInfo, nullptr, &outImage.view) != VK_SUCCESS) {
            for (auto p : stbiLoaded) if (p) stbi_image_free(p);
            throw std::runtime_error("failed to create texture image view (array)!");
        }

        outSampler = app->createTextureSampler(outImage.mipLevels);

        for (auto p : stbiLoaded) if (p) stbi_image_free(p);
    };

    // Build albedo / normal / height arrays using per-triple cpu data when available
    try {
        buildArray(true, albedoFiles, [&](size_t idx,int& w,int& h,int& ch)->const uint8_t* {
            const Triple &t = triples[idx]; if (t.hasCpuAlbedo) { w = (int)t.pixelWidth; h = (int)t.pixelHeight; ch = (int)(t.bytesPerPixel); return t.cpuAlbedo.data(); } return nullptr; }, globalAlbedoArray, globalAlbedoSampler);
        buildArray(false, normalFiles, [&](size_t idx,int& w,int& h,int& ch)->const uint8_t* {
            const Triple &t = triples[idx]; if (t.hasCpuNormal) { w = (int)t.pixelWidth; h = (int)t.pixelHeight; ch = (int)(t.bytesPerPixel); return t.cpuNormal.data(); } return nullptr; }, globalNormalArray, globalNormalSampler);
        buildArray(false, heightFiles, [&](size_t idx,int& w,int& h,int& ch)->const uint8_t* {
            const Triple &t = triples[idx]; if (t.hasCpuHeight) { w = (int)t.pixelWidth; h = (int)t.pixelHeight; ch = (int)(t.bytesPerPixel); return t.cpuHeight.data(); } return nullptr; }, globalHeightArray, globalHeightSampler);
    } catch (const std::exception &e) {
        // If building arrays fails, leave existing arrays untouched and rethrow
        throw;
    }

    // assign layer indices
    for (size_t i = 0; i < triples.size(); ++i) triples[i].layerIndex = (int)i;
}

void TextureManager::destroyAll() {
    if (!app) return;
    VkDevice device = app->getDevice();
    printf("[TextureManager] destroyAll start: triples=%zu device=%p\n", triples.size(), (void*)device);
    if (device == VK_NULL_HANDLE) {
        // Device already destroyed; can't call Vulkan destroy functions or ImGui removal safely.
        // Just clear our local bookkeeping to avoid double-destruction from destructors later.
        printf("[TextureManager] device is VK_NULL_HANDLE - skipping Vulkan destroys and ImGui removals\n");
        triples.clear();
        return;
    }
    for (size_t i = 0; i < triples.size(); ++i) {
        auto &t = triples[i];
        printf("[TextureManager] triple %zu: ownsResources=%d albedoSampler=%p albedoView=%p albedoImage=%p albedoTexID=%p\n",
               i, t.ownsResources, (void*)t.albedoSampler, (void*)t.albedo.view, (void*)t.albedo.image, (void*)t.albedoTexID);
        // remove ImGui texture handles if created (check that descriptor set is valid)
        if (t.albedoTexID && (VkDescriptorSet)t.albedoTexID != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)t.albedoTexID);
            t.albedoTexID = nullptr;
        }
        if (t.normalTexID && (VkDescriptorSet)t.normalTexID != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)t.normalTexID);
            t.normalTexID = nullptr;
        }
        if (t.heightTexID && (VkDescriptorSet)t.heightTexID != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)t.heightTexID);
            t.heightTexID = nullptr;
        }

        // Only destroy resources we own
        if (t.ownsResources) {
            if (t.albedoSampler != VK_NULL_HANDLE) vkDestroySampler(device, t.albedoSampler, nullptr);
            if (t.albedo.view != VK_NULL_HANDLE) vkDestroyImageView(device, t.albedo.view, nullptr);
            if (t.albedo.image != VK_NULL_HANDLE) vkDestroyImage(device, t.albedo.image, nullptr);
            if (t.albedo.memory != VK_NULL_HANDLE) vkFreeMemory(device, t.albedo.memory, nullptr);

            if (t.normalSampler != VK_NULL_HANDLE) vkDestroySampler(device, t.normalSampler, nullptr);
            if (t.normal.view != VK_NULL_HANDLE) vkDestroyImageView(device, t.normal.view, nullptr);
            if (t.normal.image != VK_NULL_HANDLE) vkDestroyImage(device, t.normal.image, nullptr);
            if (t.normal.memory != VK_NULL_HANDLE) vkFreeMemory(device, t.normal.memory, nullptr);

            if (t.heightSampler != VK_NULL_HANDLE) vkDestroySampler(device, t.heightSampler, nullptr);
            if (t.height.view != VK_NULL_HANDLE) vkDestroyImageView(device, t.height.view, nullptr);
            if (t.height.image != VK_NULL_HANDLE) vkDestroyImage(device, t.height.image, nullptr);
            if (t.height.memory != VK_NULL_HANDLE) vkFreeMemory(device, t.height.memory, nullptr);
        }
    }
    triples.clear();

    // Destroy global arrays
    if (globalAlbedoSampler != VK_NULL_HANDLE) vkDestroySampler(device, globalAlbedoSampler, nullptr);
    if (globalAlbedoArray.view != VK_NULL_HANDLE) vkDestroyImageView(device, globalAlbedoArray.view, nullptr);
    if (globalAlbedoArray.image != VK_NULL_HANDLE) vkDestroyImage(device, globalAlbedoArray.image, nullptr);
    if (globalAlbedoArray.memory != VK_NULL_HANDLE) vkFreeMemory(device, globalAlbedoArray.memory, nullptr);

    if (globalNormalSampler != VK_NULL_HANDLE) vkDestroySampler(device, globalNormalSampler, nullptr);
    if (globalNormalArray.view != VK_NULL_HANDLE) vkDestroyImageView(device, globalNormalArray.view, nullptr);
    if (globalNormalArray.image != VK_NULL_HANDLE) vkDestroyImage(device, globalNormalArray.image, nullptr);
    if (globalNormalArray.memory != VK_NULL_HANDLE) vkFreeMemory(device, globalNormalArray.memory, nullptr);

    if (globalHeightSampler != VK_NULL_HANDLE) vkDestroySampler(device, globalHeightSampler, nullptr);
    if (globalHeightArray.view != VK_NULL_HANDLE) vkDestroyImageView(device, globalHeightArray.view, nullptr);
    if (globalHeightArray.image != VK_NULL_HANDLE) vkDestroyImage(device, globalHeightArray.image, nullptr);
    if (globalHeightArray.memory != VK_NULL_HANDLE) vkFreeMemory(device, globalHeightArray.memory, nullptr);

    printf("[TextureManager] destroyAll done\n");
}

void TextureManager::updateDescriptorsForTriple(size_t idx, const std::vector<VkDescriptorSet>& sets) {
    if (!app) return;
    if (idx >= triples.size()) return;
    const Triple &t = triples[idx];
    // Prepare image infos for albedo/normal/height
    VkDescriptorImageInfo albedoInfo{};
    albedoInfo.imageView = t.albedo.view;
    albedoInfo.sampler = t.albedoSampler;
    albedoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo normalInfo{};
    normalInfo.imageView = t.normal.view;
    normalInfo.sampler = t.normalSampler;
    normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo heightInfo{};
    heightInfo.imageView = t.height.view;
    heightInfo.sampler = t.heightSampler;
    heightInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    for (auto ds : sets) {
        if (ds == VK_NULL_HANDLE) continue;
        VkWriteDescriptorSet writes[3] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = ds;
        writes[0].dstBinding = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &albedoInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = ds;
        writes[1].dstBinding = 2;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &normalInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = ds;
        writes[2].dstBinding = 3;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &heightInfo;

        app->updateDescriptorSet(ds, { writes[0], writes[1], writes[2] });
    }
}

ImTextureID TextureManager::getImTexture(size_t idx, int map) {
    if (idx >= triples.size()) return nullptr;
    Triple &t = triples[idx];
    switch (map) {
        case 0:
            if (!t.albedoTexID) t.albedoTexID = ImGui_ImplVulkan_AddTexture(t.albedoSampler, t.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return t.albedoTexID;
        case 1:
            if (!t.normalTexID) t.normalTexID = ImGui_ImplVulkan_AddTexture(t.normalSampler, t.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return t.normalTexID;
        case 2:
            if (!t.heightTexID) t.heightTexID = ImGui_ImplVulkan_AddTexture(t.heightSampler, t.height.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return t.heightTexID;
        default:
            return nullptr;
    }
}

void TextureManager::init(VulkanApp* a) {
    this->app = a;
}

MaterialProperties& TextureManager::getMaterial(size_t idx) {
    return triples.at(idx).material;
}

const MaterialProperties& TextureManager::getMaterial(size_t idx) const {
    return triples.at(idx).material;
}

size_t TextureManager::count() const {
    return triples.size();
}

const TextureManager::Triple& TextureManager::getTriple(size_t idx) const {
    return triples.at(idx);
}

TextureManager::~TextureManager() {
    destroyAll();
}
