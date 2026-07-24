#include "RenderTargetsWidget.hpp"

#include "../utils/Settings.hpp"
#include "../vulkan/VulkanApp.hpp"
#include "../vulkan/renderer/SceneRenderer.hpp"
#include "../vulkan/renderer/SolidRenderer.hpp"
#include "../vulkan/renderer/SkyRenderer.hpp"
#include "../vulkan/renderer/ShadowRenderer.hpp"
#include "../utils/ShadowParams.hpp"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <string>
#include <cstdio>
#include <stdexcept>
#include "../utils/FileReader.hpp"
#include <vector>
#include <array>
#include "components/ImGuiHelpers.hpp"


RenderTargetsWidget::RenderTargetsWidget(VulkanApp* app_, SceneRenderer* scene, SolidRenderer* solid, SkyRenderer* sky,
                                                                                 ShadowRenderer* shadow, ShadowParams* shadowParams_, Settings* settings_)
        : Widget("Render Targets", u8"\uf5b0"), app(app_), sceneRenderer(scene), solidRenderer(solid), skyRenderer(sky),
            shadowMapper(shadow), shadowParams(shadowParams_), settings(settings_) {
    // Initialize static GPU resources used by this widget (run once)
    init(app_, 512, 512);
}

void RenderTargetsWidget::init(VulkanApp* app_, int width, int height) {
    if (!app_) return;
    VkDevice device = app->getDevice();

    // Require application-provided sampler for widget previews; do not create fallbacks
    if (widgetSampler == VK_NULL_HANDLE) {
        // Prefer a sampler owned by the SceneRenderer's PostProcessRenderer if present
        if (sceneRenderer && sceneRenderer->postProcessRenderer) {
            widgetSampler = sceneRenderer->postProcessRenderer->getLinearSampler();
        }
        if (widgetSampler == VK_NULL_HANDLE) {
            throw std::runtime_error("RenderTargetsWidget requires a valid sampler (no fallback allowed)");
        }
    }

    // Descriptor set layout for combined sampler (binding 0)
    if (linearizeDescriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        li.bindingCount = 1;
        li.pBindings = &b;
        if (vkCreateDescriptorSetLayout(device, &li, nullptr, &linearizeDescriptorSetLayout) == VK_SUCCESS) {
            app->resources.addDescriptorSetLayout(linearizeDescriptorSetLayout, "RenderTargetsWidget: linearizeDescriptorSetLayout");
        } else linearizeDescriptorSetLayout = VK_NULL_HANDLE;
    }

    // Pipeline layout with push constants (near, far)
    if (linearizePipelineLayout == VK_NULL_HANDLE && linearizeDescriptorSetLayout != VK_NULL_HANDLE) {
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(float) * 3;
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &linearizeDescriptorSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(device, &pli, nullptr, &linearizePipelineLayout) == VK_SUCCESS) {
            app->resources.addPipelineLayout(linearizePipelineLayout, "RenderTargetsWidget: linearizePipelineLayout");
        } else linearizePipelineLayout = VK_NULL_HANDLE;
    }

    // Create graphics pipeline from SPV files (if available)
    if (linearizePipeline == VK_NULL_HANDLE && linearizePipelineLayout != VK_NULL_HANDLE) {
        std::vector<char> vertCode, fragCode;
        VkShaderModule vert = VK_NULL_HANDLE;
        VkShaderModule frag = VK_NULL_HANDLE;
        try { vert = app->getOrCreateShaderModule("shaders/depth_linearize.vert.spv"); } catch (...) { }
        try { frag = app->getOrCreateShaderModule("shaders/depth_linearize.frag.spv"); } catch (...) { }
        if (vert != VK_NULL_HANDLE && frag != VK_NULL_HANDLE) {

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            stages[1].pName = "main";

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vs{};
            vs.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vs.viewportCount = 1;
            vs.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.depthClampEnable = VK_FALSE;
            rs.rasterizerDiscardEnable = VK_FALSE;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.lineWidth = 1.0f;
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.depthBiasEnable = VK_FALSE;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable = VK_FALSE;
            ds.depthWriteEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState ca{};
            ca.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            ca.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.logicOpEnable = VK_FALSE;
            cb.attachmentCount = 1;
            cb.pAttachments = &ca;

            std::array<VkDynamicState,2> dyn = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynInfo{};
            dynInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynInfo.dynamicStateCount = static_cast<uint32_t>(dyn.size());
            dynInfo.pDynamicStates = dyn.data();

            VkGraphicsPipelineCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pi.stageCount = 2;
            pi.pStages = stages;
            pi.pVertexInputState = &vi;
            pi.pInputAssemblyState = &ia;
            pi.pViewportState = &vs;
            pi.pRasterizationState = &rs;
            pi.pMultisampleState = &ms;
            pi.pDepthStencilState = &ds;
            pi.pColorBlendState = &cb;
            pi.pDynamicState = &dynInfo;
            pi.layout = linearizePipelineLayout;
            pi.renderPass = VK_NULL_HANDLE;

            VkFormat colorFmt = VK_FORMAT_R8G8B8A8_UNORM;
            VkPipelineRenderingCreateInfo pri{};
            pri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            pri.colorAttachmentCount = 1;
            pri.pColorAttachmentFormats = &colorFmt;
            pri.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
            pi.pNext = &pri;

            if (vkCreateGraphicsPipelines(device, app->getPipelineCache(), 1, &pi, nullptr, &linearizePipeline) == VK_SUCCESS) {
                app->resources.addPipeline(linearizePipeline, "RenderTargetsWidget: linearizePipeline");
                std::cout << "[RenderTargetsWidget] Created linearizePipeline=" << (void*)linearizePipeline << " layout=" << (void*)linearizePipelineLayout << std::endl;
            } else linearizePipeline = VK_NULL_HANDLE;
        }
    }

    // Allocate descriptor set
    if (linearizeDescriptorSet == VK_NULL_HANDLE && linearizeDescriptorSetLayout != VK_NULL_HANDLE) {
        linearizeDescriptorSet = app->createDescriptorSet(linearizeDescriptorSetLayout);
    }
    // If a size is provided, create the linearized target images, views
    // and framebuffers once here (they are size-dependent). This method is
    // idempotent and can be called again after a resize.
    if (width > 0 && height > 0) {
        // Create target images + views (RGBA8) if missing
        if (linearSceneDepthImage == VK_NULL_HANDLE) {
            app->createImage(static_cast<uint32_t>(width), static_cast<uint32_t>(height), VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_TILING_OPTIMAL, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, linearSceneDepthImage, linearSceneDepthAllocation, linearSceneDepthMemory, "RenderTargetsWidget: linearSceneDepthImage");
            VkImageViewCreateInfo iv{};
            iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            iv.format = VK_FORMAT_R8G8B8A8_UNORM;
            iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            iv.subresourceRange.baseMipLevel = 0;
            iv.subresourceRange.levelCount = 1;
            iv.subresourceRange.baseArrayLayer = 0;
            iv.subresourceRange.layerCount = 1;
            iv.image = linearSceneDepthImage;
            if (vkCreateImageView(device, &iv, nullptr, &linearSceneDepthView) == VK_SUCCESS) {
                app->resources.addImageView(linearSceneDepthView, "RenderTargetsWidget: linearSceneDepthView");
            } else linearSceneDepthView = VK_NULL_HANDLE;
        }

        if (linearBackFaceDepthImage == VK_NULL_HANDLE) {
            app->createImage(static_cast<uint32_t>(width), static_cast<uint32_t>(height), VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_TILING_OPTIMAL, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, linearBackFaceDepthImage, linearBackFaceDepthAllocation, linearBackFaceDepthMemory, "RenderTargetsWidget: linearBackFaceDepthImage");
            VkImageViewCreateInfo iv{};
            iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            iv.format = VK_FORMAT_R8G8B8A8_UNORM;
            iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            iv.subresourceRange.baseMipLevel = 0;
            iv.subresourceRange.levelCount = 1;
            iv.subresourceRange.baseArrayLayer = 0;
            iv.subresourceRange.layerCount = 1;
            iv.image = linearBackFaceDepthImage;
            if (vkCreateImageView(device, &iv, nullptr, &linearBackFaceDepthView) == VK_SUCCESS) {
                app->resources.addImageView(linearBackFaceDepthView, "RenderTargetsWidget: linearBackFaceDepthView");
            } else linearBackFaceDepthView = VK_NULL_HANDLE;
        }

        if (waterDepthLinearImage == VK_NULL_HANDLE) {
            app->createImage(static_cast<uint32_t>(width), static_cast<uint32_t>(height), VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_TILING_OPTIMAL, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, waterDepthLinearImage, waterDepthLinearAllocation, waterDepthLinearMemory, "RenderTargetsWidget: waterDepthLinearImage");
            VkImageViewCreateInfo wiv{};
            wiv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            wiv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            wiv.format = VK_FORMAT_R8G8B8A8_UNORM;
            wiv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            wiv.subresourceRange.baseMipLevel = 0;
            wiv.subresourceRange.levelCount = 1;
            wiv.subresourceRange.baseArrayLayer = 0;
            wiv.subresourceRange.layerCount = 1;
            wiv.image = waterDepthLinearImage;
            if (vkCreateImageView(device, &wiv, nullptr, &waterDepthLinearView) == VK_SUCCESS) {
                app->resources.addImageView(waterDepthLinearView, "RenderTargetsWidget: waterDepthLinearView");
            } else waterDepthLinearView = VK_NULL_HANDLE;
        }

        if (linearBrushBackFaceDepthImage == VK_NULL_HANDLE) {
            app->createImage(static_cast<uint32_t>(width), static_cast<uint32_t>(height), VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_TILING_OPTIMAL, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, linearBrushBackFaceDepthImage, linearBrushBackFaceDepthAllocation, linearBrushBackFaceDepthMemory, "RenderTargetsWidget: linearBrushBackFaceDepthImage");
            VkImageViewCreateInfo biv{};
            biv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            biv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            biv.format = VK_FORMAT_R8G8B8A8_UNORM;
            biv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            biv.subresourceRange.baseMipLevel = 0;
            biv.subresourceRange.levelCount = 1;
            biv.subresourceRange.baseArrayLayer = 0;
            biv.subresourceRange.layerCount = 1;
            biv.image = linearBrushBackFaceDepthImage;
            if (vkCreateImageView(device, &biv, nullptr, &linearBrushBackFaceDepthView) == VK_SUCCESS) {
                app->resources.addImageView(linearBrushBackFaceDepthView, "RenderTargetsWidget: linearBrushBackFaceDepthView");
            } else linearBrushBackFaceDepthView = VK_NULL_HANDLE;
        }

        // Framebuffers are no longer needed - using dynamic rendering

        // Create per-face linearized targets for cubemap depth previews
        for (int face = 0; face < 6; ++face) {
            if (linearCubeFaceDepthImage[face] == VK_NULL_HANDLE) {
                app->createImage(static_cast<uint32_t>(width), static_cast<uint32_t>(height), VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_IMAGE_TILING_OPTIMAL, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, linearCubeFaceDepthImage[face], linearCubeFaceDepthAllocation[face], linearCubeFaceDepthMemory[face], "RenderTargetsWidget: linearCubeFaceDepthImage");
                VkImageViewCreateInfo iv{};
                iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
                iv.format = VK_FORMAT_R8G8B8A8_UNORM;
                iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                iv.subresourceRange.baseMipLevel = 0;
                iv.subresourceRange.levelCount = 1;
                iv.subresourceRange.baseArrayLayer = 0;
                iv.subresourceRange.layerCount = 1;
                iv.image = linearCubeFaceDepthImage[face];
                if (vkCreateImageView(device, &iv, nullptr, &linearCubeFaceDepthView[face]) == VK_SUCCESS) {
                    app->resources.addImageView(linearCubeFaceDepthView[face], "RenderTargetsWidget: linearCubeFaceDepthView");
                } else linearCubeFaceDepthView[face] = VK_NULL_HANDLE;
            }

            // No framebuffer needed with dynamic rendering
        }
    }

    // Create per-cascade linear shadow targets if a shadow mapper exists.
    if (shadowMapper) {
        uint32_t shadowSize = shadowMapper->getShadowMapSize();
        for (int c = 0; c < SHADOW_CASCADE_COUNT; ++c) {
            if (linearShadowDepthImage[c] == VK_NULL_HANDLE) {
                app->createImage(shadowSize, shadowSize, VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_IMAGE_TILING_OPTIMAL, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, linearShadowDepthImage[c], linearShadowDepthAllocation[c], linearShadowDepthMemory[c], "RenderTargetsWidget: linearShadowDepthImage");
                VkImageViewCreateInfo iv{};
                iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
                iv.format = VK_FORMAT_R8G8B8A8_UNORM;
                iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                iv.subresourceRange.baseMipLevel = 0;
                iv.subresourceRange.levelCount = 1;
                iv.subresourceRange.baseArrayLayer = 0;
                iv.subresourceRange.layerCount = 1;
                iv.image = linearShadowDepthImage[c];
                if (vkCreateImageView(device, &iv, nullptr, &linearShadowDepthView[c]) == VK_SUCCESS) {
                    app->resources.addImageView(linearShadowDepthView[c], "RenderTargetsWidget: linearShadowDepthView");
                } else linearShadowDepthView[c] = VK_NULL_HANDLE;
                linearShadowSize[c] = static_cast<int>(shadowSize);
            }

            // No framebuffer needed with dynamic rendering
        }
    }
}

bool RenderTargetsWidget::runLinearizePass(VulkanApp* app_, VkImage srcImage, VkImageView srcView, VkSampler srcSampler, VkSampler previewSampler,
                                          VkImageView dstView,
                                          uint32_t width, uint32_t height,
                                          float zNear, float zFar, float mode,
                                          uint32_t srcBaseArrayLayer) {
    if (!app_ || srcView == VK_NULL_HANDLE || dstView == VK_NULL_HANDLE) return false;
    if (linearizePipeline == VK_NULL_HANDLE || linearizeDescriptorSet == VK_NULL_HANDLE || linearizePipelineLayout == VK_NULL_HANDLE) return false;

    // std::cerr << "[RenderTargetsWidget] runLinearizePass: src=" << (void*)srcView << " dst=" << (void*)dstView << " fb=" << (void*)dstFb << " size=" << width << "x" << height << " mode=" << mode << std::endl;

    VkDescriptorImageInfo di{};
    di.sampler = srcSampler;
    di.imageView = srcView;
    di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    // Reuse the persistent linearize descriptor set (allocated once at init).
    // This pass submits synchronously and waits for completion before
    // returning, so no other command buffer can be reading this set while we
    // update and bind it — no per-call temporary allocation is needed.
    VkDescriptorSet dsToUse = linearizeDescriptorSet;
    w.dstSet = dsToUse;
    w.dstBinding = 0;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.descriptorCount = 1;
    w.pImageInfo = &di;
    app->updateDescriptorSet({ w });

    // Allocate a primary command buffer and record the linearize pass and
    // the required image layout transitions into it, then submit asynchronously.
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    try {
        cmd = app->allocatePrimaryCommandBuffer();
    } catch (...) {
        return false;
    }

    VkCommandBufferBeginInfo binfo{};
    binfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    binfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &binfo) != VK_SUCCESS) {
        // Best-effort cleanup on failure
        app->freeCommandBuffer(cmd);
        return false;
    }

    // Determine the pre-linearization layout. VulkanApp's authoritative tracked
    // layout (imageLayerLayouts) is consulted first — it reflects the last value
    // applied at submit time and is more reliable than renderer-local tracking
    // which can lag by one frame due to pending-layout deferred application.
    // Renderer-local tracking is kept as a fallback for images not yet seen by VulkanApp.
    VkImageLayout trackedOld = VK_IMAGE_LAYOUT_UNDEFINED;
    if (srcImage != VK_NULL_HANDLE) {
        // Prefer the VulkanApp authoritative tracked layout (updated at submit time).
        trackedOld = app->getImageLayoutTracked(srcImage, srcBaseArrayLayer);

        // Renderer-local fallback when VulkanApp has no entry yet.
        if (trackedOld == VK_IMAGE_LAYOUT_UNDEFINED) {
            // Solid 360 (per-face array)
            if (sceneRenderer && sceneRenderer->solid360Renderer && srcImage == sceneRenderer->solid360Renderer->getCube360DepthImage()) {
                trackedOld = sceneRenderer->solid360Renderer->getCube360DepthLayout(srcBaseArrayLayer);
            }
            // Main solid renderer (per-frame depth images)
            if (trackedOld == VK_IMAGE_LAYOUT_UNDEFINED && solidRenderer) {
                for (uint32_t f = 0; f < 2; ++f) {
                    if (srcImage == solidRenderer->getDepthImage(f)) {
                        trackedOld = solidRenderer->getDepthLayout(f);
                        break;
                    }
                }
            }
            // Back-face renderer (per-frame)
            if (trackedOld == VK_IMAGE_LAYOUT_UNDEFINED && sceneRenderer && sceneRenderer->backFaceRenderer) {
                for (uint32_t f = 0; f < 2; ++f) {
                    if (srcImage == sceneRenderer->backFaceRenderer->getBackFaceDepthImage(f)) {
                        trackedOld = sceneRenderer->backFaceRenderer->getBackFaceDepthLayout(f);
                        break;
                    }
                }
            }
            // Water renderer (water geometry depth, per-frame)
            if (trackedOld == VK_IMAGE_LAYOUT_UNDEFINED && sceneRenderer && sceneRenderer->waterRenderer) {
                for (uint32_t f = 0; f < 2; ++f) {
                    if (srcImage == sceneRenderer->waterRenderer->getWaterGeomDepthImage(f)) {
                        trackedOld = sceneRenderer->waterRenderer->getWaterGeomDepthLayout(f);
                        break;
                    }
                }
            }
            // Shadow cascades
            if (trackedOld == VK_IMAGE_LAYOUT_UNDEFINED && shadowMapper) {
                for (uint32_t sc = 0; sc < SHADOW_CASCADE_COUNT; ++sc) {
                    if (srcImage == shadowMapper->getDepthImage(sc)) {
                        trackedOld = shadowMapper->getDepthLayout(sc);
                        break;
                    }
                }
            }
        }
    }

    // Ensure the source depth image is in a shader-readable layout before sampling (recorded into cmd)
    if (srcImage != VK_NULL_HANDLE) {
        // Force-correct VulkanApp's stale imageLayerLayouts using the renderer's
        // authoritative local tracking. This avoids barriers being recorded with
        // wrong oldLayouts when the deferred applyPending hasn't run yet.
        if (trackedOld != VK_IMAGE_LAYOUT_UNDEFINED) {
            app->setImageLayoutTracked(srcImage, trackedOld, srcBaseArrayLayer, 1);
        }
        // Record transition to SHADER_READ_ONLY for sampling.
        // VulkanApp will now see the corrected tracked layout.
        app->recordTransitionImageLayoutLayer(cmd, srcImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, srcBaseArrayLayer, 1);
    }

    // Transition dst to COLOR_ATTACHMENT_OPTIMAL before rendering
    // We need to find the VkImage for this view - look it up from our known images
    VkImage dstImage = VK_NULL_HANDLE;
    if (dstView == linearSceneDepthView) { dstImage = linearSceneDepthImage; }
    else if (dstView == linearBackFaceDepthView) { dstImage = linearBackFaceDepthImage; }
    else if (dstView == linearBrushBackFaceDepthView) { dstImage = linearBrushBackFaceDepthImage; }
    else if (dstView == waterDepthLinearView) { dstImage = waterDepthLinearImage; }
    if (dstImage == VK_NULL_HANDLE) {
        for (int i = 0; i < 6; ++i) {
            if (dstView == linearCubeFaceDepthView[i]) { dstImage = linearCubeFaceDepthImage[i]; break; }
        }
    }
    if (dstImage == VK_NULL_HANDLE) {
        for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
            if (dstView == linearShadowDepthView[i]) { dstImage = linearShadowDepthImage[i]; break; }
        }
    }
    if (dstImage != VK_NULL_HANDLE) {
        VkImageMemoryBarrier2 dstBarrier{};
        dstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        dstBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        dstBarrier.srcAccessMask = VK_ACCESS_2_NONE;
        dstBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        dstBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        dstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        dstBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        dstBarrier.image = dstImage;
        dstBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &dstBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView = dstView;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = {width, height};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAtt;

    vkCmdBeginRendering(cmd, &renderingInfo);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, linearizePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, linearizePipelineLayout, 0, 1, &dsToUse, 0, nullptr);
    VkViewport vp{}; vp.x = 0.0f; vp.y = 0.0f; vp.width = static_cast<float>(width); vp.height = static_cast<float>(height); vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{}; sc.offset = {0,0}; sc.extent = { width, height };
    vkCmdSetScissor(cmd, 0, 1, &sc);
    float pc[3] = { zNear, zFar, mode };
    vkCmdPushConstants(cmd, linearizePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);

    // Transition dst back to SHADER_READ_ONLY_OPTIMAL for sampling by ImGui
    if (dstImage != VK_NULL_HANDLE) {
        VkImageMemoryBarrier2 readBarrier{};
        readBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        readBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        readBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        readBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        readBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        readBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        readBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        readBarrier.image = dstImage;
        readBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &readBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // Revert source depth image back to its pre-linearization layout.
    // Using trackedOld as desiredFinal ensures we restore the exact layout the
    // rendering system expects (e.g. DEPTH_STENCIL_READ_ONLY for shadow maps,
    // SHADER_READ_ONLY for solid depth that already ends in read-only).
    if (srcImage != VK_NULL_HANDLE) {
        // desiredFinal mirrors the layout the image was in before linearization.
        // If trackedOld is UNDEFINED fall back to DEPTH_STENCIL_ATTACHMENT.
        VkImageLayout desiredFinal = (trackedOld != VK_IMAGE_LAYOUT_UNDEFINED)
                                        ? trackedOld
                                        : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        // Pass UNDEFINED so VulkanApp resolves the old layout from the pending
        // barrier recorded above (which left the image in SHADER_READ_ONLY).
        app->recordTransitionImageLayoutLayer(cmd, srcImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, desiredFinal, 1, srcBaseArrayLayer, 1);
    }

    // Submit the recorded command buffer synchronously and wait for completion.
    // Synchronous submission avoids races where descriptor sets or image
    // layouts are updated while other command buffers are still recording.
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        app->freeCommandBuffer(cmd);
        return false;
    }
    try {
        app->submitCommandBufferAndWait(cmd);
    } catch (...) {
        // Free the command buffer on failure
        app->freeCommandBuffer(cmd);
        return false;
    }

    // Free the command buffer now that synchronous submit completed
    app->freeCommandBuffer(cmd);

    // After the synchronous submit completed, update all renderer-local layout
    // tracking to reflect the image's actual final state (trackedOld, since we
    // reverted the image back to its pre-linearization layout).
    if (srcImage != VK_NULL_HANDLE) {
        VkImageLayout finalTrackedLayout = (trackedOld != VK_IMAGE_LAYOUT_UNDEFINED)
                                               ? trackedOld
                                               : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        if (sceneRenderer && sceneRenderer->solid360Renderer && srcImage == sceneRenderer->solid360Renderer->getCube360DepthImage()) {
            sceneRenderer->solid360Renderer->setCube360DepthLayout(srcBaseArrayLayer, finalTrackedLayout);
        }
        if (solidRenderer) {
            for (uint32_t f = 0; f < 2; ++f) {
                if (srcImage == solidRenderer->getDepthImage(f)) {
                    solidRenderer->setDepthLayout(f, finalTrackedLayout);
                    break;
                }
            }
        }
        if (sceneRenderer && sceneRenderer->backFaceRenderer) {
            for (uint32_t f = 0; f < 2; ++f) {
                if (srcImage == sceneRenderer->backFaceRenderer->getBackFaceDepthImage(f)) {
                    sceneRenderer->backFaceRenderer->setBackFaceDepthLayout(f, finalTrackedLayout);
                    break;
                }
            }
        }
        if (sceneRenderer && sceneRenderer->waterRenderer) {
            for (uint32_t f = 0; f < 2; ++f) {
                if (srcImage == sceneRenderer->waterRenderer->getWaterGeomDepthImage(f)) {
                    sceneRenderer->waterRenderer->setWaterGeomDepthLayout(f, finalTrackedLayout);
                    break;
                }
            }
        }
        if (shadowMapper) {
            for (uint32_t c = 0; c < SHADOW_CASCADE_COUNT; ++c) {
                if (srcImage == shadowMapper->getDepthImage(c)) {
                    shadowMapper->setDepthLayout(c, finalTrackedLayout);
                    break;
                }
            }
        }
    }

    // ImGui descriptor is created lazily by ImTextureManager; no longer
    // managed here.

    return true;
}


RenderTargetsWidget::~RenderTargetsWidget() {
    // ImGui descriptors are managed by ImTextureManager — no cleanup needed here.
}

void RenderTargetsWidget::setFrameInfo(uint32_t frameIndex, int width, int height) {
    currentFrame = static_cast<int>(frameIndex);
    // If the size changed, destroy linear preview targets so they are recreated
    // at the new size (avoids renderArea vs framebuffer extent mismatches).
    if (cachedWidth != width || cachedHeight != height) {
        destroyLinearTargets();
        cachedWidth = width;
        cachedHeight = height;
        // Create size-dependent linear targets once via init()
        init(app, width, height);
        return;
    }
}

void RenderTargetsWidget::destroyLinearTargets() {
    VulkanApp* a = app;
    if (!a) return;

    // ImGui descriptors are managed by ImTextureManager — invalidated during
    // swapchain recreation.  Only destroy Vulkan image resources here.

    // Destroy images, views, memories and framebuffers created for linearization.
    // Always defer via deferDestroyUntilAllPending to ensure in-flight graphics
    // frames complete before freeing memory.
    auto destroyImageAndMemory = [&](VkImageView &iv, VkImage &img, VmaAllocation &alloc, VkDeviceMemory &mem) {
        if (iv == VK_NULL_HANDLE && img == VK_NULL_HANDLE && mem == VK_NULL_HANDLE) return;
        VkImageView tmp_iv = iv;
        VkImage tmp_img = img;
        VmaAllocation tmp_alloc = alloc;
        VkDeviceMemory tmp_mem = mem;
        a->deferDestroyUntilAllPending([tmp_iv, tmp_img, tmp_alloc, tmp_mem, a]() {
            if (tmp_iv != VK_NULL_HANDLE) {
                if (a->resources.removeImageView(tmp_iv))
                    vkDestroyImageView(a->getDevice(), tmp_iv, nullptr);
            }
            a->destroyImageWithVma(tmp_img, tmp_alloc, tmp_mem);
        });
        iv = VK_NULL_HANDLE;
        img = VK_NULL_HANDLE;
        alloc = VK_NULL_HANDLE;
        mem = VK_NULL_HANDLE;
    };

    auto destroyFramebuffer = [&](VkFramebuffer &fb) {
        if (fb == VK_NULL_HANDLE) return;
        VkFramebuffer tmp = fb;
        a->deferDestroyUntilAllPending([tmp, a]() {
            if (a->resources.removeFramebuffer(tmp)) vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
        });
        fb = VK_NULL_HANDLE;
    };

    for (int i = 0; i < 6; ++i) destroyFramebuffer(linearCubeFaceFramebuffer[i]);
    destroyImageAndMemory(linearSceneDepthView, linearSceneDepthImage, linearSceneDepthAllocation, linearSceneDepthMemory);
    destroyImageAndMemory(linearBackFaceDepthView, linearBackFaceDepthImage, linearBackFaceDepthAllocation, linearBackFaceDepthMemory);
    destroyImageAndMemory(linearBrushBackFaceDepthView, linearBrushBackFaceDepthImage, linearBrushBackFaceDepthAllocation, linearBrushBackFaceDepthMemory);
    destroyImageAndMemory(waterDepthLinearView, waterDepthLinearImage, waterDepthLinearAllocation, waterDepthLinearMemory);
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        destroyImageAndMemory(linearShadowDepthView[i], linearShadowDepthImage[i], linearShadowDepthAllocation[i], linearShadowDepthMemory[i]);
    }
    for (int i = 0; i < 6; ++i) {
        destroyImageAndMemory(linearCubeFaceDepthView[i], linearCubeFaceDepthImage[i], linearCubeFaceDepthAllocation[i], linearCubeFaceDepthMemory[i]);
    }

    // Keep pipeline/renderpass/layout/descriptor set until full cleanup()

    // Reset tracked sizes
    linearSceneWidth = 0;
    linearSceneHeight = 0;
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) linearShadowSize[i] = 0;
}

void RenderTargetsWidget::cleanup() {
    // ImGui descriptors are managed by ImTextureManager — no cleanup needed here.
    cube360EquirectRenderer.cleanup(app);
    // Destroy persistent staging buffers (VulkanApp::createBuffer registers them with resource manager)
    // Unmap persistent staging buffers; if GPU work is pending, defer unmap until safe
    if (stagingReadPtr && app && stagingReadBuffer.memory != VK_NULL_HANDLE) {
        stagingReadBuffer.unmap(); // VMA persistent mapping
        stagingReadPtr = nullptr;
    }
    if (stagingUploadPtr && app && stagingUploadBuffer.memory != VK_NULL_HANDLE) {
        stagingUploadBuffer.unmap(); // VMA persistent mapping
        stagingUploadPtr = nullptr;
    }
    // Drop local buffer handles; actual destruction managed by VulkanResourceManager
    stagingReadBuffer = {};
    stagingUploadBuffer = {};

    // Destroy any images / image views and persistent staging buffers that
    // this widget created. Always defer via deferDestroyUntilAllPending to
    // ensure in-flight graphics frames complete before freeing memory.
    VulkanApp* a = app;
    auto destroyImageAndMemory = [&](VkImageView &iv, VkImage &img, VmaAllocation &alloc, VkDeviceMemory &mem) {
        if (iv == VK_NULL_HANDLE && img == VK_NULL_HANDLE && mem == VK_NULL_HANDLE) return;
        VkImageView tmp_iv = iv;
        VkImage tmp_img = img;
        VmaAllocation tmp_alloc = alloc;
        VkDeviceMemory tmp_mem = mem;
        if (a) a->deferDestroyUntilAllPending([tmp_iv, tmp_img, tmp_alloc, tmp_mem, a](){
            if (tmp_iv != VK_NULL_HANDLE) {
                if (a->resources.removeImageView(tmp_iv))
                    vkDestroyImageView(a->getDevice(), tmp_iv, nullptr);
            }
            a->destroyImageWithVma(tmp_img, tmp_alloc, tmp_mem);
        });
        iv = VK_NULL_HANDLE;
        img = VK_NULL_HANDLE;
        alloc = VK_NULL_HANDLE;
        mem = VK_NULL_HANDLE;
    };

    auto destroyBufferAndMemory = [&](Buffer &buf) {
        if (buf.buffer == VK_NULL_HANDLE) return;
        VkDevice device = a ? a->getDevice() : VK_NULL_HANDLE;
        VkBuffer tmpBuf = buf.buffer;
        VmaAllocation tmpAlloc = buf.allocation;
        VkDeviceMemory tmpMem = buf.memory;
        if (tmpAlloc && a) {
            a->deferDestroyUntilAllPending([a, tmpBuf, tmpAlloc](){
                a->resources.removeBuffer(tmpBuf);
                vmaDestroyBuffer(a->getVmaAllocator(), tmpBuf, tmpAlloc);
            });
        } else if (a) {
            a->deferDestroyUntilAllPending([device, tmpBuf, tmpMem, a](){
                if (a->resources.removeBuffer(tmpBuf)) vkDestroyBuffer(device, tmpBuf, nullptr);
                if (a->resources.removeDeviceMemory(tmpMem)) vkFreeMemory(device, tmpMem, nullptr);
            });
        }
        buf = {};
    };

    // Destroy linear debug images / views
    destroyImageAndMemory(linearSceneDepthView, linearSceneDepthImage, linearSceneDepthAllocation, linearSceneDepthMemory);
    destroyImageAndMemory(linearBackFaceDepthView, linearBackFaceDepthImage, linearBackFaceDepthAllocation, linearBackFaceDepthMemory);
    destroyImageAndMemory(linearBrushBackFaceDepthView, linearBrushBackFaceDepthImage, linearBrushBackFaceDepthAllocation, linearBrushBackFaceDepthMemory);
    destroyImageAndMemory(waterDepthLinearView, waterDepthLinearImage, waterDepthLinearAllocation, waterDepthLinearMemory);
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        destroyImageAndMemory(linearShadowDepthView[i], linearShadowDepthImage[i], linearShadowDepthAllocation[i], linearShadowDepthMemory[i]);
    }

    // Destroy persistent staging buffers
    if (stagingReadPtr && app && stagingReadBuffer.memory != VK_NULL_HANDLE) {
        stagingReadBuffer.unmap(); // VMA persistent mapping
        stagingReadPtr = nullptr;
    }
    if (stagingUploadPtr && app && stagingUploadBuffer.memory != VK_NULL_HANDLE) {
        stagingUploadBuffer.unmap(); // VMA persistent mapping
        stagingUploadPtr = nullptr;
    }
    destroyBufferAndMemory(stagingReadBuffer);
    destroyBufferAndMemory(stagingUploadBuffer);

    // Destroy linearization pipeline, pipeline layout, descriptor set/layout,
    // framebuffers and renderpass created by this widget.
    // Always defer to ensure in-flight graphics frames complete.
    // auto destroyIf = [&](auto &handle, auto removeFn, auto destroyFn) {
    //     if (handle == VK_NULL_HANDLE) return;
    //     auto tmp = handle;
    //     VkDevice device = a ? a->getDevice() : VK_NULL_HANDLE;
    //     if (a) a->deferDestroyUntilAllPending([device, tmp, a, removeFn, destroyFn]() mutable {
    //         if ((a->resources.*removeFn)(tmp)) destroyFn(device, tmp);
    //     });
    //     handle = VK_NULL_HANDLE;
    // };

    // Framebuffers removed - using dynamic rendering

    // Pipeline
    if (linearizePipeline != VK_NULL_HANDLE) {
        VkPipeline tmp = linearizePipeline;
        a->deferDestroyUntilAllPending([tmp, a]() {
            if (a->resources.removePipeline(tmp)) vkDestroyPipeline(a->getDevice(), tmp, nullptr);
        });
        linearizePipeline = VK_NULL_HANDLE;
    }

    // Pipeline layout
    if (linearizePipelineLayout != VK_NULL_HANDLE) {
        VkPipelineLayout tmp = linearizePipelineLayout;
        a->deferDestroyUntilAllPending([tmp, a]() {
            if (a->resources.removePipelineLayout(tmp)) vkDestroyPipelineLayout(a->getDevice(), tmp, nullptr);
        });
        linearizePipelineLayout = VK_NULL_HANDLE;
    }

    // Descriptor set + layout
    if (linearizeDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorSet tmp = linearizeDescriptorSet;
        a->deferDestroyUntilAllPending([tmp, a]() {
            a->resources.removeDescriptorSet(tmp);
            // Descriptor sets are freed via pool management elsewhere; leave as-is
        });
        linearizeDescriptorSet = VK_NULL_HANDLE;
    }
    if (linearizeDescriptorSetLayout != VK_NULL_HANDLE) {
        VkDescriptorSetLayout tmp = linearizeDescriptorSetLayout;
        a->deferDestroyUntilAllPending([tmp, a]() {
            if (a->resources.removeDescriptorSetLayout(tmp)) vkDestroyDescriptorSetLayout(a->getDevice(), tmp, nullptr);
        });
        linearizeDescriptorSetLayout = VK_NULL_HANDLE;
    }

    // Render pass removed - using dynamic rendering

    // Destroy widget-owned sampler
    if (widgetSampler != VK_NULL_HANDLE) {
        VkSampler tmp = widgetSampler;
        a->deferDestroyUntilAllPending([tmp, a]() {
            if (a->resources.removeSampler(tmp)) vkDestroySampler(a->getDevice(), tmp, nullptr);
        });
        widgetSampler = VK_NULL_HANDLE;
    }
}

bool RenderTargetsWidget::isSolid360Preview() const {
    switch (selectedPreview) {
        case PreviewTarget::Solid360Cube:
        case PreviewTarget::Solid360DepthCube:
        case PreviewTarget::Solid360Equirect:
            return true;
        default:
            return false;
    }
}

// ImGui descriptors are managed centrally by ImTextureManager.
// No per-widget invalidation needed.

void RenderTargetsWidget::updateDescriptors(uint32_t frameIndex) {
    if (!sceneRenderer || !sceneRenderer->waterRenderer) return;

    // ImGui texture descriptors are lazily created by ImTextureManager.
    // For per-frame previews, the manager caches by (view, sampler) — since
    // render target views cycle through a small set (frames in flight), the
    // cache naturally bounds itself.  No explicit release is needed.

    // Run linearize passes for depth previews (do this first so the
    // linearized views are available for getOrCreate below).
    if (app && cachedWidth > 0 && cachedHeight > 0) {
        if (solidRenderer && linearizePipeline != VK_NULL_HANDLE && linearSceneDepthView != VK_NULL_HANDLE) {
            uint32_t producerFrame = (frameIndex + 1) % 2;
            VkImageView src = solidRenderer->getDepthView(producerFrame);
            if (src != VK_NULL_HANDLE) {
                float nearP = 0.1f, farP = 1000.0f;
                if (settings) { nearP = settings->nearPlane; farP = settings->farPlane; }
                runLinearizePass(app, solidRenderer->getDepthImage(producerFrame), src, widgetSampler, widgetSampler,
                                 linearSceneDepthView,
                                 static_cast<uint32_t>(cachedWidth), static_cast<uint32_t>(cachedHeight), nearP, farP, 0.0f);
            }
        }

        if (sceneRenderer && sceneRenderer->waterRenderer && linearizePipeline != VK_NULL_HANDLE) {
            uint32_t producerFrame = (frameIndex + 1) % 2;
            VkImageView src = (sceneRenderer && sceneRenderer->backFaceRenderer) ? sceneRenderer->backFaceRenderer->getBackFaceDepthView(producerFrame) : VK_NULL_HANDLE;
            if (src != VK_NULL_HANDLE) {
                float nearP = 0.1f, farP = 1000.0f;
                if (settings) { nearP = settings->nearPlane; farP = settings->farPlane; }
                runLinearizePass(app, sceneRenderer->backFaceRenderer->getBackFaceDepthImage(producerFrame), src, widgetSampler, widgetSampler,
                                 linearBackFaceDepthView,
                                 static_cast<uint32_t>(cachedWidth), static_cast<uint32_t>(cachedHeight), nearP, farP, 0.0f);
            }
        }

        if (sceneRenderer && sceneRenderer->waterRenderer && linearizePipeline != VK_NULL_HANDLE && waterDepthLinearView != VK_NULL_HANDLE) {
            uint32_t producerFrame = (frameIndex + 1) % 2;
            VkImageView src = sceneRenderer->waterRenderer->getWaterGeomDepthView(producerFrame);
            if (src != VK_NULL_HANDLE) {
                float nearP = 0.1f, farP = 1000.0f;
                if (settings) { nearP = settings->nearPlane; farP = settings->farPlane; }
                runLinearizePass(app, sceneRenderer->waterRenderer->getWaterGeomDepthImage(producerFrame), src, widgetSampler, widgetSampler,
                                 waterDepthLinearView,
                                 static_cast<uint32_t>(cachedWidth), static_cast<uint32_t>(cachedHeight), nearP, farP, 1.0f);
            }
        }

        if (shadowMapper && linearizePipeline != VK_NULL_HANDLE) {
            uint32_t shadowSize = shadowMapper->getShadowMapSize();
            for (int c = 0; c < SHADOW_CASCADE_COUNT; ++c) {
                VkImageView src = shadowMapper->getShadowMapView(c);
                if (src != VK_NULL_HANDLE && linearShadowDepthView[c] != VK_NULL_HANDLE) {
                    float nearP = 0.0f, farP = 1.0f;
                    runLinearizePass(app, shadowMapper->getDepthImage(c), src, widgetSampler, widgetSampler, linearShadowDepthView[c],
                                     shadowSize, shadowSize, nearP, farP, 1.0f);
                }
            }
        }
    }

    if (!app) { previewTextureID = 0; return; }

    // Choose a single preview texture ID according to the current selection.
    auto& mgr = app->imTextureManager;
    previewTextureID = 0;
    switch (selectedPreview) {
        case PreviewTarget::Sky: {
            VkImageView v = skyRenderer ? skyRenderer->getSkyView(frameIndex) : VK_NULL_HANDLE;
            if (v) previewTextureID = mgr.getOrCreate(v, widgetSampler);
        } break;
        case PreviewTarget::Solid360Cube: {
            uint32_t f = static_cast<uint32_t>(this->selectedCubeFaceIndex);
            VkImageView v = (sceneRenderer && sceneRenderer->solid360Renderer) ? sceneRenderer->solid360Renderer->getCube360FaceView(f) : VK_NULL_HANDLE;
            if (v) previewTextureID = mgr.getOrCreate(v, widgetSampler);
        } break;
        case PreviewTarget::Solid360DepthCube: {
            uint32_t f = static_cast<uint32_t>(this->selectedCubeFaceIndex);
            VkImageView v = linearCubeFaceDepthView[f];
            if (v) previewTextureID = mgr.getOrCreate(v, widgetSampler);
        } break;
        case PreviewTarget::Solid360Equirect: {
            cube360EquirectRenderer.render(app, widgetSampler, sceneRenderer->solid360Renderer->getSolid360View());
            VkImageView v = cube360EquirectRenderer.getEquirectView();
            if (v) previewTextureID = mgr.getOrCreate(v, widgetSampler);
        } break;
        case PreviewTarget::SolidColor: {
            VkImageView v = solidRenderer ? solidRenderer->getColorView(frameIndex) : VK_NULL_HANDLE;
            if (v) previewTextureID = mgr.getOrCreate(v, widgetSampler);
        } break;
        case PreviewTarget::SolidDepth: {
            uint32_t producerFrame = (frameIndex + 1) % 2;
            VkImageView v = solidRenderer ? solidRenderer->getDepthView(producerFrame) : VK_NULL_HANDLE;
            if (v) previewTextureID = mgr.getOrCreate(v, widgetSampler);
        } break;
        case PreviewTarget::WaterColor: {
            VkImageView v = (sceneRenderer && sceneRenderer->waterRenderer) ? sceneRenderer->waterRenderer->getWaterDepthView(frameIndex) : VK_NULL_HANDLE;
            if (v) previewTextureID = mgr.getOrCreate(v, widgetSampler);
        } break;
        case PreviewTarget::WaterDepth: {
            if (waterDepthLinearView) previewTextureID = mgr.getOrCreate(waterDepthLinearView, widgetSampler);
        } break;
        case PreviewTarget::BackFaceColor: {
            VkImageView v = (sceneRenderer && sceneRenderer->backFaceRenderer) ? sceneRenderer->backFaceRenderer->getBackFaceDepthView(frameIndex) : VK_NULL_HANDLE;
            if (v) previewTextureID = mgr.getOrCreate(v, widgetSampler);
        } break;
        case PreviewTarget::BackFaceDepth:
            if (linearBackFaceDepthView) previewTextureID = mgr.getOrCreate(linearBackFaceDepthView, widgetSampler);
            break;
        case PreviewTarget::BrushColor: {
            VkImageView v = solidRenderer ? solidRenderer->getColorView(frameIndex) : VK_NULL_HANDLE;
            if (v) previewTextureID = mgr.getOrCreate(v, widgetSampler);
        } break;
        case PreviewTarget::BrushDepth: {
            uint32_t producerFrame = (frameIndex + 1) % 2;
            VkImageView v = solidRenderer ? solidRenderer->getDepthView(producerFrame) : VK_NULL_HANDLE;
            if (v) previewTextureID = mgr.getOrCreate(v, widgetSampler);
        } break;
        case PreviewTarget::BrushBackFaceDepth:
            if (linearBrushBackFaceDepthView) previewTextureID = mgr.getOrCreate(linearBrushBackFaceDepthView, widgetSampler);
            break;
        case PreviewTarget::LinearSceneDepth:
            if (linearSceneDepthView) previewTextureID = mgr.getOrCreate(linearSceneDepthView, widgetSampler);
            break;
        case PreviewTarget::ShadowCascade:
            if (shadowViewMode == RenderTargetsWidget::ShadowViewMode::Linearized) {
                if (linearShadowDepthView[selectedShadowCascade])
                    previewTextureID = mgr.getOrCreate(linearShadowDepthView[selectedShadowCascade], widgetSampler);
                else if (shadowMapper)
                    previewTextureID = shadowMapper->getImTextureID(app, selectedShadowCascade);
            } else {
                if (shadowMapper)
                    previewTextureID = shadowMapper->getImTextureID(app, selectedShadowCascade);
            }
            break;
        default:
            break;
    }
}

void RenderTargetsWidget::render() {
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen, ImGuiWindowFlags_AlwaysAutoResize);
    if (!wg.visible()) return;

    if (!sceneRenderer || !sceneRenderer->waterRenderer || !solidRenderer) {
        ImGui::TextUnformatted("Renderers not available.");
        return;
    }

    

    // Fixed preview width (px)
    const float PREVIEW_WIDTH = 800.0f;
    ImGui::Separator();

    // Preview selector: show only one image at a time
    ImGui::Text("Preview Selector");
    if (ImGui::BeginPopupContextItem("preview_selector")) {
        ImGui::TextUnformatted("Choose preview");
        ImGui::EndPopup();
    }

    // Auto-advance: cycle through all targets every N frames
    ImGui::Checkbox("Auto-advance", &autoAdvance);
    if (autoAdvance) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80.0f);
        ImGui::InputInt("frames/step", &autoAdvanceInterval);
        if (autoAdvanceInterval < 1) autoAdvanceInterval = 1;
        ++autoAdvanceFrameCounter;
        if (autoAdvanceFrameCounter >= autoAdvanceInterval) {
            autoAdvanceFrameCounter = 0;
            int next = (static_cast<int>(selectedPreview) + 1) % static_cast<int>(PreviewTarget::Count);
            selectedPreview = static_cast<PreviewTarget>(next);
            // Reset shadow cascade on wrap-around so all cascades get exercised over time
            if (selectedPreview == PreviewTarget::ShadowCascade) {
                selectedShadowCascade = (selectedShadowCascade + 1) % SHADOW_CASCADE_COUNT;
            }
        }
    }
    // Add preview items array (prepare for dropdown selector)
    // Items must be in the same order as RenderTargetsWidget::PreviewTarget
    const char* previewItems[] = {
        "Sky",
        "Solid360Cube",
        "Solid360DepthCube",
        "Solid360Equirect",
        "SolidColor",
        "SolidDepth",
        "BackFaceColor",
        "BackFaceDepth",
        "BrushColor",
        "BrushDepth",
        "BrushBackFaceDepth",
        "WaterColor",
        "WaterDepth",
        "LinearSceneDepth",
        "ShadowCascade"
    };
    int previewIndex = static_cast<int>(selectedPreview);

    // Replace radio buttons with a dropdown combo using the prepared array
    if (ImGui::Combo("Preview", &previewIndex, previewItems, static_cast<int>(sizeof(previewItems)/sizeof(previewItems[0])))) {
        selectedPreview = static_cast<RenderTargetsWidget::PreviewTarget>(previewIndex);
    }
    if (selectedPreview == PreviewTarget::ShadowCascade) {
        ImGui::SliderInt("Cascade", &selectedShadowCascade, 0, SHADOW_CASCADE_COUNT - 1);
    }
    ImGui::Text("Shadow View"); ImGui::SameLine();
    if (ImGui::RadioButton("Linearized", shadowViewMode == RenderTargetsWidget::ShadowViewMode::Linearized)) {
        shadowViewMode = RenderTargetsWidget::ShadowViewMode::Linearized;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Raw", shadowViewMode == RenderTargetsWidget::ShadowViewMode::Raw)) {
        shadowViewMode = RenderTargetsWidget::ShadowViewMode::Raw;
    }
    ImGui::SameLine();
    ImGui::Checkbox("Show All Cascades", &showAllCascades);
    ImGui::Separator();

    float aspect = 1.0f;
    if (cachedWidth > 0 && cachedHeight > 0) aspect = static_cast<float>(cachedHeight) / static_cast<float>(cachedWidth);
    ImVec2 previewSize(PREVIEW_WIDTH, PREVIEW_WIDTH * aspect);

    if (selectedPreview == PreviewTarget::Solid360Cube || selectedPreview == PreviewTarget::Solid360DepthCube) {
        const char* faceLabels[6] = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};
        ImGui::Text("Cube face");
        ImGui::SameLine();
        if (ImGui::ArrowButton("##cube_face_prev", ImGuiDir_Left)) {
            this->selectedCubeFaceIndex = (this->selectedCubeFaceIndex + 5) % 6;
        }
        ImGui::SameLine();
        ImGui::Text("%s (%d/6)", faceLabels[this->selectedCubeFaceIndex], this->selectedCubeFaceIndex + 1);
        ImGui::SameLine();
        if (ImGui::ArrowButton("##cube_face_next", ImGuiDir_Right)) {
            this->selectedCubeFaceIndex = (this->selectedCubeFaceIndex + 1) % 6;
        }
    }

        // Update/create descriptors and run linearize passes now that the
        // UI selection (including cube-face arrows and shadow cascade sliders)
        // has been applied so the displayed preview matches the current UI
        // state in the same frame.
        updateDescriptors(currentFrame);

        // Debug: print current selection to stderr for quick diagnostics
        // std::cerr << "RenderTargetsWidget: selectedPreview=" << static_cast<int>(selectedPreview)
        //           << " selectedShadowCascade=" << selectedShadowCascade
        //           << " showAllCascades=" << showAllCascades
        //           << " selectedCubeFace=" << selectedCubeFaceIndex << std::endl;

        // Render only the selected preview using a single preview texture ID
        ImVec2 imgSize = previewSize;
        ImGuiHelpers::ImageOrUnavailable(previewTextureID, imgSize, "Preview unavailable");
    ImGui::Separator();



    // Optionally show all cascades (in selected shadow view mode)
    // Only show the full cascade grid when the shadow cascade preview is selected.
    if (selectedPreview == PreviewTarget::ShadowCascade && showAllCascades && shadowMapper && app) {
        float shadowSize = PREVIEW_WIDTH;
        for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
            ImGui::Text("Shadow Cascade %d", i);
            ImTextureID tid = 0;
            if (shadowViewMode == RenderTargetsWidget::ShadowViewMode::Linearized) {
                if (linearShadowDepthView[i])
                    tid = app->imTextureManager.getOrCreate(linearShadowDepthView[i], widgetSampler);
                else
                    tid = shadowMapper->getImTextureID(app, i);
            } else {
                tid = shadowMapper->getImTextureID(app, i);
            }
            ImGuiHelpers::ImageOrUnavailable(tid, ImVec2(shadowSize, shadowSize));
            ImGui::Separator();
        }
    }
}
