#pragma once

#include "vulkan.hpp"
#include <imgui.h>
#include <vector>
#include <cstring>

class EditableTexture {
public:
    EditableTexture() = default;
    
    void init(VulkanApp* app, uint32_t width, uint32_t height, VkFormat format, const char* name) {
        this->app = app;
        this->width = width;
        this->height = height;
        this->format = format;
        this->name = name;
        
        // Calculate bytes per pixel
        bytesPerPixel = (format == VK_FORMAT_R8_UNORM) ? 1 : 4;
        VkDeviceSize imageSize = width * height * bytesPerPixel;
        
        // Allocate CPU-side buffer
        cpuData.resize(imageSize);
        
        // Initialize with default gray color
        std::fill(cpuData.begin(), cpuData.end(), 128);
        
        // Create GPU image
        createImage();
        createImageView();
        createSampler();
        
        // Upload initial data
        updateGPU();
        
        // Create ImGui descriptor set
        createImGuiDescriptor();
    }
    
    void cleanup() {
        if (!app) return;
        VkDevice device = app->getDevice();
        
        if (sampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, sampler, nullptr);
            sampler = VK_NULL_HANDLE;
        }
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, view, nullptr);
            view = VK_NULL_HANDLE;
        }
        if (image != VK_NULL_HANDLE) {
            vkDestroyImage(device, image, nullptr);
            image = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }
    
    // Edit a pixel (RGBA format)
    void setPixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
        if (x >= width || y >= height || bytesPerPixel != 4) return;
        size_t index = (y * width + x) * 4;
        cpuData[index + 0] = r;
        cpuData[index + 1] = g;
        cpuData[index + 2] = b;
        cpuData[index + 3] = a;
        isDirty = true;
    }
    
    // Edit a pixel (single channel format)
    void setPixelGray(uint32_t x, uint32_t y, uint8_t value) {
        if (x >= width || y >= height || bytesPerPixel != 1) return;
        size_t index = y * width + x;
        cpuData[index] = value;
        isDirty = true;
    }
    
    // Fill entire texture with a color
    void fill(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
        if (bytesPerPixel == 4) {
            for (uint32_t i = 0; i < width * height; ++i) {
                cpuData[i * 4 + 0] = r;
                cpuData[i * 4 + 1] = g;
                cpuData[i * 4 + 2] = b;
                cpuData[i * 4 + 3] = a;
            }
        } else if (bytesPerPixel == 1) {
            std::fill(cpuData.begin(), cpuData.end(), r);
        }
        isDirty = true;
    }
    
    // Upload changes to GPU
    void updateGPU() {
        if (!app) return;
        
        VkDeviceSize imageSize = width * height * bytesPerPixel;
        
        // Create staging buffer
        Buffer stagingBuffer = app->createBuffer(
            imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        
        // Copy CPU data to staging buffer
        void* data;
        vkMapMemory(app->getDevice(), stagingBuffer.memory, 0, imageSize, 0, &data);
        memcpy(data, cpuData.data(), static_cast<size_t>(imageSize));
        vkUnmapMemory(app->getDevice(), stagingBuffer.memory);
        
        // Transition image layout and copy buffer to image
        transitionImageLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer.buffer);
        transitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        
        // Cleanup staging buffer
        vkDestroyBuffer(app->getDevice(), stagingBuffer.buffer, nullptr);
        vkFreeMemory(app->getDevice(), stagingBuffer.memory, nullptr);
        
        isDirty = false;
    }
    
    // Render ImGui widget
    void renderImGui() {
        if (!app || imguiDescSet == VK_NULL_HANDLE) return;
        
        ImGui::Begin(name.c_str());
        
        ImGui::Text("Size: %dx%d", width, height);
        ImGui::Text("Format: %s", bytesPerPixel == 4 ? "RGBA8" : "R8");
        
        if (isDirty) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Texture has unsaved changes");
            if (ImGui::Button("Upload to GPU")) {
                updateGPU();
            }
        } else {
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "Texture is up to date");
        }
        
        ImGui::Separator();
        
        // Color picker for filling
        static float fillColor[4] = {0.5f, 0.5f, 0.5f, 1.0f};
        ImGui::ColorEdit4("Fill Color", fillColor);
        
        if (ImGui::Button("Fill Texture")) {
            fill(
                static_cast<uint8_t>(fillColor[0] * 255),
                static_cast<uint8_t>(fillColor[1] * 255),
                static_cast<uint8_t>(fillColor[2] * 255),
                static_cast<uint8_t>(fillColor[3] * 255)
            );
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Clear to Black")) {
            fill(0, 0, 0, 255);
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Clear to White")) {
            fill(255, 255, 255, 255);
        }
        
        ImGui::Separator();
        
        // Display texture preview
        float previewSize = 512.0f;
        ImVec2 imageSize(previewSize, previewSize);
        ImGui::Text("Preview:");
        ImGui::Image((ImTextureID)imguiDescSet, imageSize);
        
        // Interactive drawing on the texture
        if (ImGui::IsItemHovered() && ImGui::IsMouseDown(0)) {
            ImVec2 mousePos = ImGui::GetMousePos();
            ImVec2 imageMin = ImGui::GetItemRectMin();
            ImVec2 imageMax = ImGui::GetItemRectMax();
            
            // Calculate texture coordinates
            float relX = (mousePos.x - imageMin.x) / (imageMax.x - imageMin.x);
            float relY = (mousePos.y - imageMin.y) / (imageMax.y - imageMin.y);
            
            if (relX >= 0.0f && relX <= 1.0f && relY >= 0.0f && relY <= 1.0f) {
                uint32_t texX = static_cast<uint32_t>(relX * width);
                uint32_t texY = static_cast<uint32_t>(relY * height);
                
                // Draw with current fill color
                uint8_t r = static_cast<uint8_t>(fillColor[0] * 255);
                uint8_t g = static_cast<uint8_t>(fillColor[1] * 255);
                uint8_t b = static_cast<uint8_t>(fillColor[2] * 255);
                uint8_t a = static_cast<uint8_t>(fillColor[3] * 255);
                
                // Draw a small brush (3x3 pixels)
                int brushSize = 3;
                for (int dy = -brushSize; dy <= brushSize; ++dy) {
                    for (int dx = -brushSize; dx <= brushSize; ++dx) {
                        int px = static_cast<int>(texX) + dx;
                        int py = static_cast<int>(texY) + dy;
                        if (px >= 0 && px < static_cast<int>(width) && 
                            py >= 0 && py < static_cast<int>(height)) {
                            if (bytesPerPixel == 4) {
                                setPixel(px, py, r, g, b, a);
                            } else {
                                setPixelGray(px, py, r);
                            }
                        }
                    }
                }
            }
        }
        
        ImGui::Text("Tip: Click and drag on the preview to paint!");
        
        ImGui::End();
    }
    
    VkImageView getView() const { return view; }
    VkSampler getSampler() const { return sampler; }
    VkImage getImage() const { return image; }
    bool getDirty() const { return isDirty; }
    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }
    uint32_t getBytesPerPixel() const { return bytesPerPixel; }
    VkDescriptorSet getImGuiDescriptorSet() const { return imguiDescSet; }
    
    // Get as TextureImage for TextureManager compatibility
    TextureImage getTextureImage() const {
        TextureImage tex;
        tex.image = image;
        tex.memory = memory;
        tex.view = view;
        tex.mipLevels = 1;
        return tex;
    }
    
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
    
    void createImage() {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        
        if (vkCreateImage(app->getDevice(), &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create editable image");
        }
        
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(app->getDevice(), image, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = app->findMemoryType(memRequirements.memoryTypeBits, 
                                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        if (vkAllocateMemory(app->getDevice(), &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory");
        }
        
        vkBindImageMemory(app->getDevice(), image, memory, 0);
    }
    
    void createImageView() {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        
        if (vkCreateImageView(app->getDevice(), &viewInfo, nullptr, &view) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image view");
        }
    }
    
    void createSampler() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        
        if (vkCreateSampler(app->getDevice(), &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler");
        }
    }
    
    void createImGuiDescriptor() {
        imguiDescSet = ImGui_ImplVulkan_AddTexture(sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    
    void transitionImageLayout(VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = app->beginSingleTimeCommands();
        
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        
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
            throw std::invalid_argument("unsupported layout transition");
        }
        
        vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        app->endSingleTimeCommands(commandBuffer);
    }
    
    void copyBufferToImage(VkBuffer buffer) {
        VkCommandBuffer commandBuffer = app->beginSingleTimeCommands();
        
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};
        
        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        app->endSingleTimeCommands(commandBuffer);
    }
};
