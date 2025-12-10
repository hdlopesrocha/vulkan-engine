#pragma once

#include "EditableTexture.hpp"
#include <imgui.h>
#include <random>

// Push constants for compute shader
struct PerlinPushConstants {
    float scale;
    float octaves;
    float persistence;
    float lacunarity;
    uint32_t seed;
    float brightness; // Added brightness adjustment (-1.0 to 1.0)
    float contrast;   // Added contrast adjustment (0.0 to 2.0)
    float padding;    // Padding for alignment
};

class EditableTextureSet {
public:
    EditableTextureSet() = default;
    
    void init(VulkanApp* app, uint32_t width, uint32_t height, const char* windowName = "Editable Textures") {
        this->app = app;
        this->windowName = windowName;
        
        // Initialize all three textures (all RGBA8 for compute shader compatibility)
        albedo.init(app, width, height, VK_FORMAT_R8G8B8A8_UNORM, "Albedo");
        normal.init(app, width, height, VK_FORMAT_R8G8B8A8_UNORM, "Normal");
        bump.init(app, width, height, VK_FORMAT_R8G8B8A8_UNORM, "Bump"); // Changed from R8 to RGBA8
        
        // Initialize compute pipeline for Perlin noise
        createComputePipeline();
    }
    
    void cleanup() {
        albedo.cleanup();
        normal.cleanup();
        bump.cleanup();
        
        // Cleanup compute pipeline resources
        if (computePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(app->getDevice(), computePipeline, nullptr);
            computePipeline = VK_NULL_HANDLE;
        }
        if (computePipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(app->getDevice(), computePipelineLayout, nullptr);
            computePipelineLayout = VK_NULL_HANDLE;
        }
        if (computeDescriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(app->getDevice(), computeDescriptorSetLayout, nullptr);
            computeDescriptorSetLayout = VK_NULL_HANDLE;
        }
        if (computeDescriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(app->getDevice(), computeDescriptorPool, nullptr);
            computeDescriptorPool = VK_NULL_HANDLE;
        }
    }
    
    // Render a single ImGui window with tabs
    void renderImGui() {
        ImGui::Begin(windowName.c_str(), nullptr, ImGuiWindowFlags_None);
        
        if (ImGui::BeginTabBar("TextureTabBar")) {
            // Albedo Tab
            if (ImGui::BeginTabItem("Albedo")) {
                renderTextureTab(albedo);
                ImGui::EndTabItem();
            }
            
            // Normal Tab
            if (ImGui::BeginTabItem("Normal")) {
                renderTextureTab(normal);
                ImGui::EndTabItem();
            }
            
            // Bump Tab
            if (ImGui::BeginTabItem("Bump")) {
                renderTextureTab(bump);
                ImGui::EndTabItem();
            }
            
            ImGui::EndTabBar();
        }
        
        ImGui::End();
    }
    
    // Getters for individual textures
    EditableTexture& getAlbedo() { return albedo; }
    EditableTexture& getNormal() { return normal; }
    EditableTexture& getBump() { return bump; }
    
    const EditableTexture& getAlbedo() const { return albedo; }
    const EditableTexture& getNormal() const { return normal; }
    const EditableTexture& getBump() const { return bump; }
    
private:
    VulkanApp* app = nullptr;
    std::string windowName = "Editable Textures";
    EditableTexture albedo;
    EditableTexture normal;
    EditableTexture bump;
    
    // Compute pipeline for Perlin noise generation
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VkPipelineLayout computePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool computeDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet albedoComputeDescSet = VK_NULL_HANDLE;
    VkDescriptorSet normalComputeDescSet = VK_NULL_HANDLE;
    VkDescriptorSet bumpComputeDescSet = VK_NULL_HANDLE;
    
    // Perlin noise parameters
    float perlinScale = 8.0f;
    float perlinOctaves = 4.0f;
    float perlinPersistence = 0.5f;
    float perlinLacunarity = 2.0f;
    float perlinBrightness = 0.0f;  // -1.0 to 1.0
    float perlinContrast = 1.0f;    // 0.0 to 2.0
    uint32_t perlinSeed = 12345;    // Fixed seed for consistent generation
    
    // Previous parameter values for change detection
    float prevPerlinScale = 8.0f;
    float prevPerlinOctaves = 4.0f;
    float prevPerlinPersistence = 0.5f;
    float prevPerlinLacunarity = 2.0f;
    float prevPerlinBrightness = 0.0f;
    float prevPerlinContrast = 1.0f;
    
    void renderTextureTab(EditableTexture& texture) {
        ImGui::Text("Size: %dx%d", texture.getWidth(), texture.getHeight());
        
        const char* formatName = "Unknown";
        if (texture.getBytesPerPixel() == 4) {
            formatName = "RGBA8";
        } else if (texture.getBytesPerPixel() == 1) {
            formatName = "R8";
        }
        ImGui::Text("Format: %s", formatName);
        
        if (texture.getDirty()) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Texture has unsaved changes");
            if (ImGui::Button("Upload to GPU")) {
                texture.updateGPU();
            }
        } else {
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "Texture is up to date");
        }
        
        ImGui::Separator();
        
        // Perlin Noise Generation Section
        ImGui::Text("Perlin Noise Generator");
        
        bool paramsChanged = false;
        
        if (ImGui::SliderFloat("Scale", &perlinScale, 1.0f, 32.0f)) {
            paramsChanged = true;
        }
        if (ImGui::SliderFloat("Octaves", &perlinOctaves, 1.0f, 8.0f)) {
            paramsChanged = true;
        }
        if (ImGui::SliderFloat("Persistence", &perlinPersistence, 0.0f, 1.0f)) {
            paramsChanged = true;
        }
        if (ImGui::SliderFloat("Lacunarity", &perlinLacunarity, 1.0f, 4.0f)) {
            paramsChanged = true;
        }
        
        ImGui::Separator();
        ImGui::Text("Adjustments");
        
        if (ImGui::SliderFloat("Brightness", &perlinBrightness, -1.0f, 1.0f)) {
            paramsChanged = true;
        }
        if (ImGui::SliderFloat("Contrast", &perlinContrast, 0.0f, 2.0f)) {
            paramsChanged = true;
        }
        
        // Auto-generate when parameters change
        if (paramsChanged) {
            generatePerlinNoise(texture);
            // Update previous values
            prevPerlinScale = perlinScale;
            prevPerlinOctaves = perlinOctaves;
            prevPerlinPersistence = perlinPersistence;
            prevPerlinLacunarity = perlinLacunarity;
            prevPerlinBrightness = perlinBrightness;
            prevPerlinContrast = perlinContrast;
        }
        
        if (ImGui::Button("Generate Perlin Noise")) {
            generatePerlinNoise(texture);
        }
        ImGui::SameLine();
        if (ImGui::Button("Randomize Seed")) {
            // Only randomize the seed to generate a new pattern
            std::random_device rd;
            perlinSeed = rd();
            generatePerlinNoise(texture);
        }
        
        ImGui::Separator();
        
        // Display texture preview
        float previewSize = 512.0f;
        ImVec2 imageSize(previewSize, previewSize);
        ImGui::Text("Preview:");
        
        VkDescriptorSet descSet = texture.getImGuiDescriptorSet();
        if (descSet != VK_NULL_HANDLE) {
            ImGui::Image((ImTextureID)descSet, imageSize);
        } else {
            ImGui::Text("Texture preview not available");
        }
    }
    
    void createComputePipeline() {
        // Create descriptor set layout
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &binding;
        
        if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
        
        // Create pipeline layout with push constants
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PerlinPushConstants);
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        
        if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }
        
        // Load compute shader
        auto computeShaderCode = FileReader::readFile("shaders/perlin_noise.comp.spv");
        VkShaderModule computeShaderModule = app->createShaderModule(computeShaderCode);
        
        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";
        
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = computeShaderStageInfo;
        pipelineInfo.layout = computePipelineLayout;
        
        if (vkCreateComputePipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }
        
        vkDestroyShaderModule(app->getDevice(), computeShaderModule, nullptr);
        
        // Create descriptor pool
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSize.descriptorCount = 3; // One for each texture
        
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 3;
        
        if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor pool!");
        }
        
        // Create descriptor sets for each texture
        createComputeDescriptorSet(albedo, albedoComputeDescSet);
        createComputeDescriptorSet(normal, normalComputeDescSet);
        createComputeDescriptorSet(bump, bumpComputeDescSet);
    }
    
    void createComputeDescriptorSet(EditableTexture& texture, VkDescriptorSet& descSet) {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = computeDescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &computeDescriptorSetLayout;
        
        if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, &descSet) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate compute descriptor set!");
        }
        
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = texture.getView();
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        
        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descSet;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pImageInfo = &imageInfo;
        
        vkUpdateDescriptorSets(app->getDevice(), 1, &descriptorWrite, 0, nullptr);
    }
    
    void generatePerlinNoise(EditableTexture& texture) {
        // Determine which descriptor set to use
        VkDescriptorSet descSet = VK_NULL_HANDLE;
        if (&texture == &albedo) {
            descSet = albedoComputeDescSet;
        } else if (&texture == &normal) {
            descSet = normalComputeDescSet;
        } else if (&texture == &bump) {
            descSet = bumpComputeDescSet;
        }
        
        if (descSet == VK_NULL_HANDLE) {
            return;
        }
        
        // Prepare push constants
        PerlinPushConstants pushConstants;
        pushConstants.scale = perlinScale;
        pushConstants.octaves = perlinOctaves;
        pushConstants.persistence = perlinPersistence;
        pushConstants.lacunarity = perlinLacunarity;
        pushConstants.brightness = perlinBrightness;
        pushConstants.contrast = perlinContrast;
        pushConstants.padding = 0.0f;
        pushConstants.seed = perlinSeed; // Use the fixed seed
        
        // Debug output
        printf("Generating Perlin noise: scale=%.2f, octaves=%.0f, persistence=%.2f, lacunarity=%.2f, brightness=%.2f, contrast=%.2f, seed=%u\n",
               pushConstants.scale, pushConstants.octaves, pushConstants.persistence, pushConstants.lacunarity, 
               pushConstants.brightness, pushConstants.contrast, pushConstants.seed);
        printf("Texture size: %dx%d, dispatch groups: %dx%d\n", 
               texture.getWidth(), texture.getHeight(), 
               (texture.getWidth() + 15) / 16, (texture.getHeight() + 15) / 16);
        
        // Begin command buffer
        VkCommandBuffer commandBuffer = app->beginSingleTimeCommands();
        
        // Transition image to GENERAL layout for compute shader write
        // Note: Image should be in SHADER_READ_ONLY_OPTIMAL after initialization
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = texture.getImage();
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
        
        // Bind pipeline and descriptor set
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &descSet, 0, nullptr);
        
        // Push constants
        vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PerlinPushConstants), &pushConstants);
        
        // Dispatch compute shader (16x16 local size)
        uint32_t groupCountX = (texture.getWidth() + 15) / 16;
        uint32_t groupCountY = (texture.getHeight() + 15) / 16;
        vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
        
        // Transition back to shader read layout
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
        
        // End and submit
        app->endSingleTimeCommands(commandBuffer);
        
        printf("Perlin noise generation complete!\n");
    }
};
