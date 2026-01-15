#pragma once

#include "VulkanApp.hpp"
#include "DescriptorSetBuilder.hpp"
#include "TextureArrayManager.hpp"
#include "TextureTriple.hpp"
#include "MaterialManager.hpp"
#include "ShaderStage.hpp"
#include "../utils/FileReader.hpp"
#include "../math/Vertex.hpp"
#include <unordered_map>
#include <memory>
#include "Model3DVersion.hpp"
#include "SkyRenderer.hpp"
#include "SolidRenderer.hpp"
#include "IndirectRenderer.hpp"
#include "ShadowMapper.hpp"
#include "WaterRenderer.hpp"
#include "../math/SphereModel.hpp"
#include "../Uniforms.hpp"
#include "../utils/Scene.hpp"
// forward declare SkyWidget to avoid including widget headers here
class SkyWidget;

class SceneRenderer {
public:
    VulkanApp* app = nullptr;

    std::unique_ptr<SkyRenderer> skyRenderer;
    ShadowMapper shadowMapper;
    WaterRenderer waterRenderer;
    std::unique_ptr<SolidRenderer> solidRenderer;

    // Create graphics pipelines. This mirrors the previous logic in main.cpp
    void createPipelines() {
        if (!app) return;

        ShaderStage vertexShader = ShaderStage(
            app->createShaderModule(FileReader::readFile("shaders/main.vert.spv")),
            VK_SHADER_STAGE_VERTEX_BIT
        );

        ShaderStage fragmentShader = ShaderStage(
            app->createShaderModule(FileReader::readFile("shaders/main.frag.spv")),
            VK_SHADER_STAGE_FRAGMENT_BIT
        );

        if (!solidRenderer) {
            solidRenderer = std::make_unique<SolidRenderer>(app);
            solidRenderer->init(app);
        }
        solidRenderer->createPipelines();
        // Pipelines, descriptor pools and descriptor sets are now delegated to the individual renderers.
        return;
    }

    // Create descriptor pools/sets. Minimal implementation for now —
    // the detailed per-triple descriptor setup is handled elsewhere.
    void createDescriptorSets(MaterialManager &materialManager, TextureArrayManager &textureArrayManager, VkDescriptorSet &outDescriptorSet, VkDescriptorSet &outShadowPassDescriptorSet, size_t tripleCount) {
        if (!app) {
            outDescriptorSet = VK_NULL_HANDLE;
            outShadowPassDescriptorSet = VK_NULL_HANDLE;
            return;
        }

        // Roughly estimate pool sizes and create descriptor pool
        uint32_t uboCount = static_cast<uint32_t>(tripleCount * 2 + 8);
        uint32_t samplerCount = static_cast<uint32_t>(tripleCount * 4 + 16);
        app->createDescriptorPool(uboCount, samplerCount);

        // Create global material descriptor set if materials exist
        VkDescriptorSet globalMatDS = VK_NULL_HANDLE;
        if (materialManager.count() > 0) {
            globalMatDS = app->createDescriptorSet(app->getMaterialDescriptorSetLayout());
            VkDescriptorBufferInfo materialBufInfo{ materialManager.getBuffer().buffer, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet matWrite{}; matWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; matWrite.dstSet = globalMatDS; matWrite.dstBinding = 5; matWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; matWrite.descriptorCount = 1; matWrite.pBufferInfo = &materialBufInfo;
            app->updateDescriptorSet(globalMatDS, { matWrite });
            app->setMaterialDescriptorSet(globalMatDS);
            app->registerDescriptorSet(globalMatDS);
        }

        outDescriptorSet = VK_NULL_HANDLE;
        outShadowPassDescriptorSet = VK_NULL_HANDLE;

        DescriptorSetBuilder dsBuilder(app, &shadowMapper);
        for (size_t i = 0; i < tripleCount; ++i) {
            Triple tr;
            tr.albedo.view = textureArrayManager.albedoArray.view;
            tr.albedoSampler = textureArrayManager.albedoSampler;
            tr.normal.view = textureArrayManager.normalArray.view;
            tr.normalSampler = textureArrayManager.normalSampler;
            tr.height.view = textureArrayManager.bumpArray.view;
            tr.heightSampler = textureArrayManager.bumpSampler;

            if (outDescriptorSet == VK_NULL_HANDLE) {
                VkDescriptorSet ds = dsBuilder.createMainDescriptorSet(tr, mainPassUBO.buffer, false, nullptr, 0);

                // Attach sky UBO (binding 6)
                VkDescriptorBufferInfo skyBufInfo{ mainPassUBO.buffer.buffer, 0, sizeof(UniformObject) };
                VkWriteDescriptorSet skyWrite{}; skyWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; skyWrite.dstSet = ds; skyWrite.dstBinding = 6; skyWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; skyWrite.descriptorCount = 1; skyWrite.pBufferInfo = &skyBufInfo;

                // Attach water params UBO (binding 7)
                VkDescriptorBufferInfo waterParamsBufInfo{ waterPassUBO.buffer.buffer, 0, sizeof(WaterParamsGPU) };
                VkWriteDescriptorSet waterParamsWrite{}; waterParamsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; waterParamsWrite.dstSet = ds; waterParamsWrite.dstBinding = 7; waterParamsWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; waterParamsWrite.descriptorCount = 1; waterParamsWrite.pBufferInfo = &waterParamsBufInfo;

                app->updateDescriptorSet(ds, { skyWrite, waterParamsWrite });
                outDescriptorSet = ds;
                app->registerDescriptorSet(outDescriptorSet);
            }

            if (outShadowPassDescriptorSet == VK_NULL_HANDLE) {
                VkDescriptorSet ds = dsBuilder.createShadowDescriptorSet(tr, shadowPassUBO.buffer, false, nullptr, 0);
                outShadowPassDescriptorSet = ds;
                app->registerDescriptorSet(outShadowPassDescriptorSet);
            }
        }

        // If texture arrays are present, update descriptor sets with combined image samplers
        if (textureArrayManager.layerAmount > 0) {
            VkDescriptorImageInfo albedoArrayInfo{ textureArrayManager.albedoSampler, textureArrayManager.albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo normalArrayInfo{ textureArrayManager.normalSampler, textureArrayManager.normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo bumpArrayInfo{ textureArrayManager.bumpSampler, textureArrayManager.bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

            if (outDescriptorSet != VK_NULL_HANDLE) {
                VkWriteDescriptorSet w1{}; w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w1.dstSet = outDescriptorSet; w1.dstBinding = 1; w1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w1.descriptorCount = 1; w1.pImageInfo = &albedoArrayInfo;
                VkWriteDescriptorSet w2{}; w2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w2.dstSet = outDescriptorSet; w2.dstBinding = 2; w2.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w2.descriptorCount = 1; w2.pImageInfo = &normalArrayInfo;
                VkWriteDescriptorSet w3{}; w3.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w3.dstSet = outDescriptorSet; w3.dstBinding = 3; w3.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w3.descriptorCount = 1; w3.pImageInfo = &bumpArrayInfo;
                app->updateDescriptorSet(outDescriptorSet, { w1, w2, w3 });
            }
            if (outShadowPassDescriptorSet != VK_NULL_HANDLE) {
                VkWriteDescriptorSet w1{}; w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w1.dstSet = outShadowPassDescriptorSet; w1.dstBinding = 1; w1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w1.descriptorCount = 1; w1.pImageInfo = &albedoArrayInfo;
                VkWriteDescriptorSet w2{}; w2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w2.dstSet = outShadowPassDescriptorSet; w2.dstBinding = 2; w2.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w2.descriptorCount = 1; w2.pImageInfo = &normalArrayInfo;
                VkWriteDescriptorSet w3{}; w3.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w3.dstSet = outShadowPassDescriptorSet; w3.dstBinding = 3; w3.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w3.descriptorCount = 1; w3.pImageInfo = &bumpArrayInfo;
                app->updateDescriptorSet(outShadowPassDescriptorSet, { w1, w2, w3 });
            }
        }
    }

    // Render pass helpers moved from main.cpp. These methods perform the recorded
    // commands for each pass and require some app-provided parameters.
    void shadowPass(VkCommandBuffer &commandBuffer, VkQueryPool queryPool, VkDescriptorSet shadowPassDescriptorSet, const UniformObject &uboStatic, bool shadowsEnabled, bool shadowTessellationEnabled) {
        if (!app) return;

        if (shadowsEnabled) {
            if (solidRenderer) solidRenderer->getIndirectRenderer().prepareCull(commandBuffer, glm::mat4(1.0f)); // caller should set up cull matrix externally if needed
            if (queryPool != VK_NULL_HANDLE) {
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 1);
            }

            shadowMapper.beginShadowPass(commandBuffer, uboStatic.lightSpaceMatrix);

            shadowPassUBO.data = uboStatic;
            shadowPassUBO.data.viewProjection = uboStatic.lightSpaceMatrix;
            shadowPassUBO.data.materialFlags = glm::vec4(0.0f);
            shadowPassUBO.data.passParams = glm::vec4(1.0f, shadowTessellationEnabled ? 1.0f : 0.0f, 0.0f, 0.0f);
            app->updateUniformBuffer(shadowPassUBO.buffer, &shadowPassUBO.data, sizeof(UniformObject));

            {
                VkDescriptorSet setsToBind[2];
                uint32_t bindCount = 0;
                VkDescriptorSet matDs = app->getMaterialDescriptorSet();
                if (matDs != VK_NULL_HANDLE) setsToBind[bindCount++] = matDs;
                if (shadowPassDescriptorSet != VK_NULL_HANDLE) setsToBind[bindCount++] = shadowPassDescriptorSet;
                if (bindCount > 0) vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->getPipelineLayout(), 0, bindCount, setsToBind, 0, nullptr);
            }

            if (solidRenderer) solidRenderer->getIndirectRenderer().drawPrepared(commandBuffer, app);

            shadowMapper.endShadowPass(commandBuffer);

            if (queryPool != VK_NULL_HANDLE) {
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 2);
            }
        } else {
            if (queryPool != VK_NULL_HANDLE) {
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 1);
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 2);
            }
        }
    }

    void depthPrePass(VkCommandBuffer &commandBuffer, VkQueryPool queryPool) {
        if (!app) return;
            if (solidRenderer) solidRenderer->depthPrePass(commandBuffer, queryPool);
        if (queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, queryPool, 5);
        }
    }

    void mainPass(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &mainPassInfo, uint32_t frameIdx, bool hasWater, VkDescriptorSet perTextureDescriptorSet, bool wireframeEnabled, bool profilingEnabled, VkQueryPool queryPool, const glm::mat4 &viewProj,
                  const UniformObject &uboStatic, const WaterParams &waterParams, float waterTime, bool normalMappingEnabled, bool tessellationEnabled, bool shadowsEnabled, int debugMode, float triplanarThreshold, float triplanarExponent) {
        if (!app) return;

        // Update main pass UBO from provided static and per-frame parameters
        mainPassUBO.data = uboStatic;
        mainPassUBO.data.viewProjection = viewProj;
        mainPassUBO.data.materialFlags.w = normalMappingEnabled ? 1.0f : 0.0f;
        mainPassUBO.data.shadowEffects = glm::vec4(0.0f, 0.0f, 0.0f, shadowsEnabled ? 1.0f : 0.0f);
        mainPassUBO.data.debugParams = glm::vec4((float)debugMode, 0.0f, 0.0f, 0.0f);
        mainPassUBO.data.triplanarSettings = glm::vec4(triplanarThreshold, triplanarExponent, 0.0f, 0.0f);
        mainPassUBO.data.passParams = glm::vec4(
            waterTime,
            tessellationEnabled ? 1.0f : 0.0f,
            waterParams.waveScale,
            waterParams.noiseScale
        );
        mainPassUBO.data.tessParams = glm::vec4(
            waterParams.waveSpeed,
            waterParams.refractionStrength,
            waterParams.fresnelPower,
            waterParams.transparency
        );
        app->updateUniformBuffer(mainPassUBO.buffer, &mainPassUBO.data, sizeof(UniformObject));

        // Ensure host writes to UBO are visible to subsequent shader stages
        VkMemoryBarrier memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            1, &memBarrier,
            0, nullptr,
            0, nullptr
        );

        vkCmdBeginRenderPass(commandBuffer, &mainPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)app->getWidth();
        viewport.height = (float)app->getHeight();
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        if (skyRenderer && perTextureDescriptorSet != VK_NULL_HANDLE) {
            skyRenderer->update();
            SkyMode skyMode = SkyMode::Gradient; // caller-controlled elsewhere
            skyRenderer->render(commandBuffer, perTextureDescriptorSet, mainPassUBO.buffer, mainPassUBO.data, viewProj, skyMode);
        }

        if (queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, queryPool, 3);
        }

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = { (uint32_t)app->getWidth(), (uint32_t)app->getHeight() };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        if (solidRenderer) solidRenderer->getIndirectRenderer().bindBuffers(commandBuffer);

        {
            VkDescriptorSet setsToBind[2];
            uint32_t bindCount = 0;
            VkDescriptorSet matDs = app->getMaterialDescriptorSet();
            if (matDs != VK_NULL_HANDLE) setsToBind[bindCount++] = matDs;
            if (perTextureDescriptorSet != VK_NULL_HANDLE) setsToBind[bindCount++] = perTextureDescriptorSet;
            if (bindCount > 0) vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->getPipelineLayout(), 0, bindCount, setsToBind, 0, nullptr);
        }

        // Depth pre-pass
        depthPrePass(commandBuffer, queryPool);

        // Delegate pipeline binding and draw to SolidRenderer
        if (solidRenderer) solidRenderer->draw(commandBuffer, app, perTextureDescriptorSet, wireframeEnabled);

        if (solidRenderer) solidRenderer->getIndirectRenderer().drawIndirectOnly(commandBuffer, app);

        if (profilingEnabled && queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 6);
        }
    }

    void waterPass(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo, uint32_t frameIdx, VkDescriptorSet perTextureDescriptorSet, bool profilingEnabled, VkQueryPool queryPool, const WaterParams &waterParams, float waterTime) {
        if (!app) return;
        VkPipeline waterGeomPipeline = waterRenderer.getWaterGeometryPipeline();

        // Update water UBO (GPU-side) from provided CPU params
        waterPassUBO.data.params1 = glm::vec4(
            waterParams.refractionStrength,
            waterParams.fresnelPower,
            waterParams.transparency,
            waterParams.foamDepthThreshold
        );
        waterPassUBO.data.params2 = glm::vec4(
            waterParams.waterTint,
            1.0f / waterParams.noiseScale,
            static_cast<float>(waterParams.noiseOctaves),
            waterParams.noisePersistence
        );
        waterPassUBO.data.params3 = glm::vec4(waterParams.noiseTimeSpeed, waterTime, waterParams.shoreStrength, waterParams.shoreFalloff);
        waterPassUBO.data.shallowColor = glm::vec4(waterParams.shallowColor, 0.0f);
        waterPassUBO.data.deepColor = glm::vec4(waterParams.deepColor, waterParams.foamIntensity);
        waterPassUBO.data.foamParams = glm::vec4(1.0f/waterParams.foamNoiseScale, static_cast<float>(waterParams.foamNoiseOctaves), waterParams.foamNoisePersistence, waterParams.foamTintIntensity);
        waterPassUBO.data.foamParams2 = glm::vec4(waterParams.foamBrightness, waterParams.foamContrast, 0.0f, 0.0f);
        waterPassUBO.data.foamTint = waterParams.foamTint;
        app->updateUniformBuffer(waterPassUBO.buffer, &waterPassUBO.data, sizeof(WaterParamsGPU));

        // Ensure host writes to water UBO are visible to fragment shaders
        VkMemoryBarrier waterMemBarrier{};
        waterMemBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        waterMemBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        waterMemBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            1, &waterMemBarrier,
            0, nullptr,
            0, nullptr
        );
        // End offscreen render pass
        vkCmdEndRenderPass(commandBuffer);

        // Find the current swapchain image
        uint32_t imageIndex = 0;
        const auto& framebuffers = app->getSwapchainFramebuffers();
        for (uint32_t i = 0; i < framebuffers.size(); ++i) {
            if (framebuffers[i] == renderPassInfo.framebuffer) {
                imageIndex = i; break;
            }
        }
        VkImage currentSwapchainImage = app->getSwapchainImages()[imageIndex];

        VkImageMemoryBarrier sceneColorBarrier{};
        sceneColorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        sceneColorBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        sceneColorBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        sceneColorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sceneColorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sceneColorBarrier.image = waterRenderer.getSceneColorImage(frameIdx);
        sceneColorBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        sceneColorBarrier.subresourceRange.baseMipLevel = 0;
        sceneColorBarrier.subresourceRange.levelCount = 1;
        sceneColorBarrier.subresourceRange.baseArrayLayer = 0;
        sceneColorBarrier.subresourceRange.layerCount = 1;
        sceneColorBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        sceneColorBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        VkImageMemoryBarrier sceneDepthBarrier{};
        sceneDepthBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        sceneDepthBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        sceneDepthBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        sceneDepthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sceneDepthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sceneDepthBarrier.image = waterRenderer.getSceneDepthImage(frameIdx);
        sceneDepthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        sceneDepthBarrier.subresourceRange.baseMipLevel = 0;
        sceneDepthBarrier.subresourceRange.levelCount = 1;
        sceneDepthBarrier.subresourceRange.baseArrayLayer = 0;
        sceneDepthBarrier.subresourceRange.layerCount = 1;
        sceneDepthBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        sceneDepthBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        VkImageMemoryBarrier swapBarrier{};
        swapBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        swapBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        swapBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        swapBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapBarrier.image = currentSwapchainImage;
        swapBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        swapBarrier.subresourceRange.baseMipLevel = 0;
        swapBarrier.subresourceRange.levelCount = 1;
        swapBarrier.subresourceRange.baseArrayLayer = 0;
        swapBarrier.subresourceRange.layerCount = 1;
        swapBarrier.srcAccessMask = 0;
        swapBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        VkImageMemoryBarrier mainDepthBarrier{};
        mainDepthBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        mainDepthBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        mainDepthBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        mainDepthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mainDepthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mainDepthBarrier.image = app->getDepthImage();
        mainDepthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        mainDepthBarrier.subresourceRange.baseMipLevel = 0;
        mainDepthBarrier.subresourceRange.levelCount = 1;
        mainDepthBarrier.subresourceRange.baseArrayLayer = 0;
        mainDepthBarrier.subresourceRange.layerCount = 1;
        mainDepthBarrier.srcAccessMask = 0;
        mainDepthBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        std::array<VkImageMemoryBarrier, 4> preBlitBarriers = {sceneColorBarrier, sceneDepthBarrier, swapBarrier, mainDepthBarrier};
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr,
            static_cast<uint32_t>(preBlitBarriers.size()), preBlitBarriers.data());

        VkImageBlit blitRegion{};
        blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.srcSubresource.mipLevel = 0;
        blitRegion.srcSubresource.baseArrayLayer = 0;
        blitRegion.srcSubresource.layerCount = 1;
        blitRegion.srcOffsets[0] = {0, 0, 0};
        blitRegion.srcOffsets[1] = {(int32_t)app->getWidth(), (int32_t)app->getHeight(), 1};
        blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.dstSubresource.mipLevel = 0;
        blitRegion.dstSubresource.baseArrayLayer = 0;
        blitRegion.dstSubresource.layerCount = 1;
        blitRegion.dstOffsets[0] = {0, 0, 0};
        blitRegion.dstOffsets[1] = {(int32_t)app->getWidth(), (int32_t)app->getHeight(), 1};

        vkCmdBlitImage(commandBuffer,
            waterRenderer.getSceneColorImage(frameIdx), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            currentSwapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blitRegion, VK_FILTER_NEAREST);

        VkImageBlit depthBlitRegion{};
        depthBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthBlitRegion.srcSubresource.mipLevel = 0;
        depthBlitRegion.srcSubresource.baseArrayLayer = 0;
        depthBlitRegion.srcSubresource.layerCount = 1;
        depthBlitRegion.srcOffsets[0] = {0, 0, 0};
        depthBlitRegion.srcOffsets[1] = {(int32_t)app->getWidth(), (int32_t)app->getHeight(), 1};
        depthBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthBlitRegion.dstSubresource.mipLevel = 0;
        depthBlitRegion.dstSubresource.baseArrayLayer = 0;
        depthBlitRegion.dstSubresource.layerCount = 1;
        depthBlitRegion.dstOffsets[0] = {0, 0, 0};
        depthBlitRegion.dstOffsets[1] = {(int32_t)app->getWidth(), (int32_t)app->getHeight(), 1};

        vkCmdBlitImage(commandBuffer,
            waterRenderer.getSceneDepthImage(frameIdx), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            app->getDepthImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &depthBlitRegion, VK_FILTER_NEAREST);

        swapBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        swapBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        swapBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        swapBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        mainDepthBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        mainDepthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        mainDepthBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mainDepthBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

        VkImageMemoryBarrier sceneColorReadBarrier{};
        sceneColorReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        sceneColorReadBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        sceneColorReadBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        sceneColorReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sceneColorReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sceneColorReadBarrier.image = waterRenderer.getSceneColorImage(frameIdx);
        sceneColorReadBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        sceneColorReadBarrier.subresourceRange.baseMipLevel = 0;
        sceneColorReadBarrier.subresourceRange.levelCount = 1;
        sceneColorReadBarrier.subresourceRange.baseArrayLayer = 0;
        sceneColorReadBarrier.subresourceRange.layerCount = 1;
        sceneColorReadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        sceneColorReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkImageMemoryBarrier sceneDepthReadBarrier{};
        sceneDepthReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        sceneDepthReadBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        sceneDepthReadBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        sceneDepthReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sceneDepthReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sceneDepthReadBarrier.image = waterRenderer.getSceneDepthImage(frameIdx);
        sceneDepthReadBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        sceneDepthReadBarrier.subresourceRange.baseMipLevel = 0;
        sceneDepthReadBarrier.subresourceRange.levelCount = 1;
        sceneDepthReadBarrier.subresourceRange.baseArrayLayer = 0;
        sceneDepthReadBarrier.subresourceRange.layerCount = 1;
        sceneDepthReadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        sceneDepthReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        std::array<VkImageMemoryBarrier, 4> postBlitBarriers = {swapBarrier, mainDepthBarrier, sceneColorReadBarrier, sceneDepthReadBarrier};
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr,
            static_cast<uint32_t>(postBlitBarriers.size()), postBlitBarriers.data());

        // Begin swapchain render pass for water (load existing color, load depth)
        VkRenderPassBeginInfo waterRenderPassInfo{};
        waterRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        waterRenderPassInfo.renderPass = app->getContinuationRenderPass();
        waterRenderPassInfo.framebuffer = renderPassInfo.framebuffer;
        waterRenderPassInfo.renderArea.offset = {0, 0};
        waterRenderPassInfo.renderArea.extent = {static_cast<uint32_t>(app->getWidth()), static_cast<uint32_t>(app->getHeight())};
        waterRenderPassInfo.clearValueCount = 0;
        waterRenderPassInfo.pClearValues = nullptr;

        vkCmdBeginRenderPass(commandBuffer, &waterRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport waterViewport{};
        waterViewport.x = 0.0f;
        waterViewport.y = 0.0f;
        waterViewport.width = (float)app->getWidth();
        waterViewport.height = (float)app->getHeight();
        waterViewport.minDepth = 0.0f;
        waterViewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &waterViewport);

        VkRect2D waterScissor{};
        waterScissor.offset = {0, 0};
        waterScissor.extent = {static_cast<uint32_t>(app->getWidth()), static_cast<uint32_t>(app->getHeight())};
        vkCmdSetScissor(commandBuffer, 0, 1, &waterScissor);

        int debugMode = int(mainPassUBO.data.debugParams.x + 0.5f);
        if (waterGeomPipeline != VK_NULL_HANDLE && (debugMode == 0 || debugMode == 32)) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, waterGeomPipeline);
            waterRenderer.getIndirectRenderer().bindBuffers(commandBuffer);

            VkDescriptorSet matDs = app->getMaterialDescriptorSet();
            VkDescriptorSet sceneDs = waterRenderer.getWaterDepthDescriptorSet(frameIdx);

            if (matDs != VK_NULL_HANDLE && perTextureDescriptorSet != VK_NULL_HANDLE) {
                VkDescriptorSet sets01[2] = { matDs, perTextureDescriptorSet };
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    waterRenderer.getWaterGeometryPipelineLayout(), 0, 2, sets01, 0, nullptr);
            }

            if (sceneDs != VK_NULL_HANDLE) {
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    waterRenderer.getWaterGeometryPipelineLayout(), 2, 1, &sceneDs, 0, nullptr);
            }

            waterRenderer.getIndirectRenderer().drawIndirectOnly(commandBuffer, app);
        }

        // Render ImGui is handled by the app after this call
        // NOTE: do not end the swapchain render pass here so the caller can render ImGui
        // (ImGui_ImplVulkan_RenderDrawData expects to be called inside an active render pass).
    }

    template<typename T>
    struct PassUBO {
        Buffer buffer;
        T data;

        PassUBO() = default;
        PassUBO(VulkanApp* app, size_t size) {
            buffer = app->createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
    };

    PassUBO<UniformObject> mainPassUBO{};
    PassUBO<UniformObject> shadowPassUBO{};
    PassUBO<WaterParamsGPU> waterPassUBO{};

    SceneRenderer(VulkanApp* app_) : app(app_), shadowMapper(app_, 8192), waterRenderer(app_) {}

    void init(VulkanApp* app_, SkyWidget* skyWidget = nullptr, VkDescriptorSet descriptorSet = VK_NULL_HANDLE) {
        if (app_) app = app_;
        if (!app) return;

        // Sky renderer
        if (!skyRenderer) {
            skyRenderer = std::make_unique<SkyRenderer>(app);
            skyRenderer->init();
        }

        // Initialize sky renderer internals if needed
        // Sky VBO and SkySphere are managed by SkyRenderer

        // Shadow mapper
        // Note: init() on shadowMapper is safe to call if already initialized
        shadowMapper.init();

        // Indirect renderer (compute culling + merged buffers)
        if (!solidRenderer) {
            solidRenderer = std::make_unique<SolidRenderer>(app);
        }
        solidRenderer->init(app);

        // Create global UBOs if not already created
        if (mainPassUBO.buffer.buffer == VK_NULL_HANDLE) mainPassUBO = PassUBO<UniformObject>(app, sizeof(UniformObject));
        if (shadowPassUBO.buffer.buffer == VK_NULL_HANDLE) shadowPassUBO = PassUBO<UniformObject>(app, sizeof(UniformObject));
        if (waterPassUBO.buffer.buffer == VK_NULL_HANDLE) waterPassUBO = PassUBO<WaterParamsGPU>(app, sizeof(WaterParamsGPU));

        // Initialize water renderer with its UBO buffer
        waterRenderer.init(waterPassUBO.buffer);

        // If skyWidget and descriptorSet are provided, initialize SkyRenderer's sky
        if (skyWidget && descriptorSet != VK_NULL_HANDLE) {
            skyRenderer->initSky(skyWidget, descriptorSet);
        }
    }

    void cleanup() {
        if (!app) return;

        if (skyRenderer) skyRenderer->cleanup();

        // Cleanup solid renderer (handles opaque meshes and its indirect renderer)
        if (solidRenderer) solidRenderer->cleanup();

        // Remove water meshes registered in WaterRenderer
        waterRenderer.removeAllRegisteredMeshes();

        // Cleanup water renderer
        waterRenderer.cleanup();

        // Destroy UBO buffers
        if (mainPassUBO.buffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(app->getDevice(), mainPassUBO.buffer.buffer, nullptr);
        if (mainPassUBO.buffer.memory != VK_NULL_HANDLE) vkFreeMemory(app->getDevice(), mainPassUBO.buffer.memory, nullptr);
        if (shadowPassUBO.buffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(app->getDevice(), shadowPassUBO.buffer.buffer, nullptr);
        if (shadowPassUBO.buffer.memory != VK_NULL_HANDLE) vkFreeMemory(app->getDevice(), shadowPassUBO.buffer.memory, nullptr);
        if (waterPassUBO.buffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(app->getDevice(), waterPassUBO.buffer.buffer, nullptr);
        if (waterPassUBO.buffer.memory != VK_NULL_HANDLE) vkFreeMemory(app->getDevice(), waterPassUBO.buffer.memory, nullptr);

        // Shadow mapper cleanup (SkyRenderer handles its own sky resources)
        shadowMapper.cleanup();
    }

    // initSky removed; use init(app, skyWidget, descriptorSet) instead
};
