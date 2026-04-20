#include "RenderTargetsWidget.hpp"
#include "Settings.hpp"
#include "../vulkan/VulkanApp.hpp"
#include "../vulkan/SceneRenderer.hpp"
#include "../vulkan/SolidRenderer.hpp"
#include "../vulkan/SkyRenderer.hpp"
#include "../vulkan/ShadowRenderer.hpp"
#include "../utils/ShadowParams.hpp"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <string>
#include <cstdio>
#include <stdexcept>
#include "../utils/FileReader.hpp"
#include <vector>
#include <array>


RenderTargetsWidget::RenderTargetsWidget(VulkanApp* app, SceneRenderer* scene, SolidRenderer* solid, SkyRenderer* sky,
                                                                                 ShadowRenderer* shadow, ShadowParams* shadowParams, Settings* settings)
        : Widget("Render Targets"), app(app), sceneRenderer(scene), solidRenderer(solid), skyRenderer(sky),
            shadowMapper(shadow), shadowParams(shadowParams), settings(settings) {
    // Initialize static GPU resources used by this widget (run once)
    init(app, 512, 512);
}

void RenderTargetsWidget::init(VulkanApp* app, int width, int height) {
    if (!app) return;
    VkDevice device = app->getDevice();

    // Create a simple sampler used as a fallback for widget previews
    if (widgetSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.anisotropyEnable = VK_FALSE;
        sci.maxAnisotropy = 1.0f;
        sci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sci.unnormalizedCoordinates = VK_FALSE;
        sci.compareEnable = VK_FALSE;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        if (vkCreateSampler(device, &sci, nullptr, &widgetSampler) == VK_SUCCESS) {
            app->resources.addSampler(widgetSampler, "RenderTargetsWidget: widgetSampler");
        } else {
            widgetSampler = VK_NULL_HANDLE;
        }
    }

    // Create a tiny renderpass for the linearize write (color output)
    if (linearizeRenderPass == VK_NULL_HANDLE) {
        VkAttachmentDescription att{};
        att.format = VK_FORMAT_R8G8B8A8_UNORM;
        att.samples = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpci{};
        rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpci.attachmentCount = 1;
        rpci.pAttachments = &att;
        rpci.subpassCount = 1;
        rpci.pSubpasses = &subpass;
        rpci.dependencyCount = 1;
        rpci.pDependencies = &dep;

        if (vkCreateRenderPass(device, &rpci, nullptr, &linearizeRenderPass) == VK_SUCCESS) {
            app->resources.addRenderPass(linearizeRenderPass, "RenderTargetsWidget: linearizeRenderPass");
        } else linearizeRenderPass = VK_NULL_HANDLE;
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
    if (linearizePipeline == VK_NULL_HANDLE && linearizePipelineLayout != VK_NULL_HANDLE && linearizeRenderPass != VK_NULL_HANDLE) {
        std::vector<char> vertCode, fragCode;
        try { vertCode = FileReader::readFile("shaders/depth_linearize.vert.spv"); } catch (...) { }
        try { fragCode = FileReader::readFile("shaders/depth_linearize.frag.spv"); } catch (...) { }
        if (!vertCode.empty() && !fragCode.empty()) {
            VkShaderModule vert = app->createShaderModule(vertCode);
            VkShaderModule frag = app->createShaderModule(fragCode);

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
            pi.renderPass = linearizeRenderPass;
            pi.subpass = 0;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &linearizePipeline) == VK_SUCCESS) {
                app->resources.addPipeline(linearizePipeline, "RenderTargetsWidget: linearizePipeline");
                fprintf(stderr, "[RenderTargetsWidget] Created linearizePipeline=%p layout=%p renderPass=%p\n", (void*)linearizePipeline, (void*)linearizePipelineLayout, (void*)linearizeRenderPass);
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
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, linearSceneDepthImage, linearSceneDepthMemory, "RenderTargetsWidget: linearSceneDepthImage");
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
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, linearBackFaceDepthImage, linearBackFaceDepthMemory, "RenderTargetsWidget: linearBackFaceDepthImage");
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

        // Create framebuffers for the two linearized targets
        if (linearSceneFramebuffer == VK_NULL_HANDLE && linearSceneDepthView != VK_NULL_HANDLE && linearizeRenderPass != VK_NULL_HANDLE) {
            VkImageView atts[] = { linearSceneDepthView };
            VkFramebufferCreateInfo fb{};
            fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb.renderPass = linearizeRenderPass;
            fb.attachmentCount = 1;
            fb.pAttachments = atts;
            fb.width = static_cast<uint32_t>(width);
            fb.height = static_cast<uint32_t>(height);
            fb.layers = 1;
            if (vkCreateFramebuffer(device, &fb, nullptr, &linearSceneFramebuffer) == VK_SUCCESS) {
                app->resources.addFramebuffer(linearSceneFramebuffer, "RenderTargetsWidget: linearSceneFramebuffer");
            } else linearSceneFramebuffer = VK_NULL_HANDLE;
        }
        if (linearBackFaceFramebuffer == VK_NULL_HANDLE && linearBackFaceDepthView != VK_NULL_HANDLE && linearizeRenderPass != VK_NULL_HANDLE) {
            VkImageView atts[] = { linearBackFaceDepthView };
            VkFramebufferCreateInfo fb{};
            fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb.renderPass = linearizeRenderPass;
            fb.attachmentCount = 1;
            fb.pAttachments = atts;
            fb.width = static_cast<uint32_t>(width);
            fb.height = static_cast<uint32_t>(height);
            fb.layers = 1;
            if (vkCreateFramebuffer(device, &fb, nullptr, &linearBackFaceFramebuffer) == VK_SUCCESS) {
                app->resources.addFramebuffer(linearBackFaceFramebuffer, "RenderTargetsWidget: linearBackFaceFramebuffer");
            } else linearBackFaceFramebuffer = VK_NULL_HANDLE;
        }

        // Create per-face linearized targets for cubemap depth previews
        for (int face = 0; face < 6; ++face) {
            if (linearCubeFaceDepthImage[face] == VK_NULL_HANDLE) {
                app->createImage(static_cast<uint32_t>(width), static_cast<uint32_t>(height), VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_IMAGE_TILING_OPTIMAL, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, linearCubeFaceDepthImage[face], linearCubeFaceDepthMemory[face], "RenderTargetsWidget: linearCubeFaceDepthImage");
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

            if (linearCubeFaceFramebuffer[face] == VK_NULL_HANDLE && linearCubeFaceDepthView[face] != VK_NULL_HANDLE && linearizeRenderPass != VK_NULL_HANDLE) {
                VkImageView atts[] = { linearCubeFaceDepthView[face] };
                VkFramebufferCreateInfo fb{};
                fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                fb.renderPass = linearizeRenderPass;
                fb.attachmentCount = 1;
                fb.pAttachments = atts;
                fb.width = static_cast<uint32_t>(width);
                fb.height = static_cast<uint32_t>(height);
                fb.layers = 1;
                if (vkCreateFramebuffer(device, &fb, nullptr, &linearCubeFaceFramebuffer[face]) == VK_SUCCESS) {
                    app->resources.addFramebuffer(linearCubeFaceFramebuffer[face], "RenderTargetsWidget: linearCubeFaceFramebuffer");
                } else linearCubeFaceFramebuffer[face] = VK_NULL_HANDLE;
            }
        }
    }

    // Create per-cascade linear shadow targets if a shadow mapper exists.
    if (shadowMapper && linearizeRenderPass != VK_NULL_HANDLE) {
        uint32_t shadowSize = shadowMapper->getShadowMapSize();
        for (int c = 0; c < SHADOW_CASCADE_COUNT; ++c) {
            if (linearShadowDepthImage[c] == VK_NULL_HANDLE) {
                app->createImage(shadowSize, shadowSize, VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_IMAGE_TILING_OPTIMAL, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, linearShadowDepthImage[c], linearShadowDepthMemory[c], "RenderTargetsWidget: linearShadowDepthImage");
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

            if (linearShadowFramebuffer[c] == VK_NULL_HANDLE && linearShadowDepthView[c] != VK_NULL_HANDLE && linearizeRenderPass != VK_NULL_HANDLE) {
                VkImageView atts[] = { linearShadowDepthView[c] };
                VkFramebufferCreateInfo fb{};
                fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                fb.renderPass = linearizeRenderPass;
                fb.attachmentCount = 1;
                fb.pAttachments = atts;
                fb.width = shadowSize;
                fb.height = shadowSize;
                fb.layers = 1;
                if (vkCreateFramebuffer(device, &fb, nullptr, &linearShadowFramebuffer[c]) == VK_SUCCESS) {
                    app->resources.addFramebuffer(linearShadowFramebuffer[c], "RenderTargetsWidget: linearShadowFramebuffer");
                } else linearShadowFramebuffer[c] = VK_NULL_HANDLE;
            }
        }
    }
}

bool RenderTargetsWidget::runLinearizePass(VulkanApp* app, VkImage srcImage, VkImageView srcView, VkSampler srcSampler, VkSampler previewSampler,
                                          VkImageView dstView, VkFramebuffer dstFb,
                                          VkDescriptorSet &dstDescriptor, bool &dstDescriptorOwned,
                                          uint32_t width, uint32_t height,
                                          float zNear, float zFar, float mode,
                                          uint32_t srcBaseArrayLayer) {
    if (!app || srcView == VK_NULL_HANDLE || dstView == VK_NULL_HANDLE || dstFb == VK_NULL_HANDLE) return false;
    if (linearizePipeline == VK_NULL_HANDLE || linearizeDescriptorSet == VK_NULL_HANDLE || linearizePipelineLayout == VK_NULL_HANDLE) return false;

    fprintf(stderr, "[RenderTargetsWidget] runLinearizePass: src=%p dst=%p fb=%p size=%ux%u mode=%f\n", (void*)srcView, (void*)dstView, (void*)dstFb, (unsigned)width, (unsigned)height, mode);

    VkDescriptorImageInfo di{};
    di.sampler = srcSampler;
    di.imageView = srcView;
    di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    // Always allocate a temporary descriptor set for this pass to avoid
    // updating a descriptor set that may be currently bound by another
    // command buffer (including one in RECORDING state). The temporary
    // set will be freed after the synchronous submit below.
    VkDescriptorSet dsToUse = VK_NULL_HANDLE;
    try {
        dsToUse = app->createDescriptorSet(linearizeDescriptorSetLayout);
    } catch (...) {
        // Fall back to the persistent set if allocation fails (rare)
        dsToUse = linearizeDescriptorSet;
    }
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

    // Ensure the source depth image is in a shader-readable layout before sampling (recorded into cmd)
    if (srcImage != VK_NULL_HANDLE) {
        // Consult renderer-tracked layouts when available so we emit a barrier
        // with the correct oldLayout instead of guessing. Check known renderers
        // (solid360, back-face, water) across both frames.
        VkImageLayout trackedOld = VK_IMAGE_LAYOUT_UNDEFINED;
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
        // Shadow cascades (single-image per cascade)
        if (trackedOld == VK_IMAGE_LAYOUT_UNDEFINED && shadowMapper) {
            for (uint32_t sc = 0; sc < SHADOW_CASCADE_COUNT; ++sc) {
                if (srcImage == shadowMapper->getDepthImage(sc)) {
                    VkImageLayout l = shadowMapper->getDepthLayout(sc);
                    if (l == VK_IMAGE_LAYOUT_UNDEFINED) l = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                    trackedOld = l;
                    break;
                }
            }
        }
        fprintf(stderr, "[RenderTargetsWidget] runLinearizePass: srcImage=%p baseLayer=%u trackedOld=%d\n",
                (void*)srcImage, (unsigned)srcBaseArrayLayer, (int)trackedOld);
        // Record transition using the tracked old layout (no widget fallbacks).
        app->recordTransitionImageLayoutLayer(cmd, srcImage, VK_FORMAT_D32_SFLOAT, trackedOld, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, srcBaseArrayLayer, 1);
    }

    VkClearValue clr{}; clr.color = {{0.0f,0.0f,0.0f,1.0f}};
    VkRenderPassBeginInfo rbi{};
    rbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rbi.renderPass = linearizeRenderPass;
    rbi.framebuffer = dstFb;
    rbi.renderArea.offset = {0,0};
    rbi.renderArea.extent = { width, height };
    rbi.clearValueCount = 1;
    rbi.pClearValues = &clr;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, linearizePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, linearizePipelineLayout, 0, 1, &dsToUse, 0, nullptr);
    VkViewport vp{}; vp.x = 0.0f; vp.y = 0.0f; vp.width = static_cast<float>(width); vp.height = static_cast<float>(height); vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{}; sc.offset = {0,0}; sc.extent = { width, height };
    vkCmdSetScissor(cmd, 0, 1, &sc);
    float pc[3] = { zNear, zFar, mode };
    vkCmdPushConstants(cmd, linearizePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);

    // Revert source depth image back to a depth-stencil attachment layout (recorded into cmd)
    if (srcImage != VK_NULL_HANDLE) {
        // Use the canonical final layout: depth-stencil attachment.
        VkImageLayout desiredFinal = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        fprintf(stderr, "[RenderTargetsWidget] runLinearizePass (pre-revert): srcImage=%p baseLayer=%u desiredFinal=%d\n", (void*)srcImage, (unsigned)srcBaseArrayLayer, (int)desiredFinal);
        app->recordTransitionImageLayoutLayer(cmd, srcImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, desiredFinal, 1, srcBaseArrayLayer, 1);
        // Best-effort: do not change the renderer's tracked layout here because
        // recorded barriers are not yet submitted. Update tracked layout only
        // after submit so other record-time logic sees a consistent state.
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

    // Free the temporary descriptor set immediately — GPU work is complete.
    if (dsToUse != linearizeDescriptorSet && dsToUse != VK_NULL_HANDLE) {
        VkDevice dev = app->getDevice();
        VkDescriptorPool pool = app->getDescriptorPool();
        vkFreeDescriptorSets(dev, pool, 1, &dsToUse);
        app->resources.removeDescriptorSet(dsToUse);
    }

    // Free the command buffer now that synchronous submit completed
    app->freeCommandBuffer(cmd);

    // After the synchronous submit completed, update the renderer's per-face
    // tracking for cubemaps so future passes will record correct oldLayout values.
    if (srcImage != VK_NULL_HANDLE) {
        if (sceneRenderer && sceneRenderer->solid360Renderer && srcImage == sceneRenderer->solid360Renderer->getCube360DepthImage()) {
            sceneRenderer->solid360Renderer->setCube360DepthLayout(srcBaseArrayLayer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            fprintf(stderr, "[RenderTargetsWidget] runLinearizePass: updated cube360 tracked layout for srcImage=%p baseLayer=%u -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL\n", (void*)srcImage, (unsigned)srcBaseArrayLayer);
        }
        // Update main solid renderer tracked layouts if we operated on its depth image
        if (solidRenderer) {
            for (uint32_t f = 0; f < 2; ++f) {
                if (srcImage == solidRenderer->getDepthImage(f)) {
                    solidRenderer->setDepthLayout(f, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
                    fprintf(stderr, "[RenderTargetsWidget] runLinearizePass: updated solid renderer tracked layout for srcImage=%p frame=%u -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL\n", (void*)srcImage, (unsigned)f);
                    break;
                }
            }
        }
    }

    if (dstView != VK_NULL_HANDLE && dstDescriptor == VK_NULL_HANDLE) {
        dstDescriptor = ImGui_ImplVulkan_AddTexture(previewSampler, dstView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        dstDescriptorOwned = true;
    }

    return true;
}


RenderTargetsWidget::~RenderTargetsWidget() {
    // Remove only descriptors that this widget created. Some descriptor sets
    // (e.g. those returned by other renderers) must not be removed here.
    auto removeOwnedDesc = [&](VkDescriptorSet &ds, bool &owned) {
        if (ds == VK_NULL_HANDLE) return;
        if (!owned) { ds = VK_NULL_HANDLE; return; }
        // Defer removal until the in-flight fence for the current frame to
        // avoid destroying descriptor sets that may be bound by command
        // buffers still being recorded or submitted.
        if (app) {
            VkDescriptorSet tmp = ds;
            // Defer until all pending GPU work completes to avoid destroying
            // descriptor sets that may be bound by other command buffers.
            app->deferDestroyUntilAllPending([tmp]() { ImGui_ImplVulkan_RemoveTexture(tmp); });
        } else {
            ImGui_ImplVulkan_RemoveTexture(ds);
        }
        ds = VK_NULL_HANDLE;
        owned = false;
    };
    removeOwnedDesc(skyDescriptor, skyDescriptorOwned);
    removeOwnedDesc(solidColorDescriptor, solidColorDescriptorOwned);
    removeOwnedDesc(solidDepthDescriptor, solidDepthDescriptorOwned);
    removeOwnedDesc(waterColorDescriptor, waterColorDescriptorOwned);
    removeOwnedDesc(solid360Descriptor, solid360DescriptorOwned);
    removeOwnedDesc(cube360EquirectDescriptor, cube360EquirectDescriptorOwned);
    for (int i = 0; i < 6; ++i) removeOwnedDesc(cube360FaceDescriptor[i], cube360FaceDescriptorOwned[i]);
    for (int i = 0; i < 6; ++i) removeOwnedDesc(cube360FaceDepthDescriptor[i], cube360FaceDepthDescriptorOwned[i]);
    removeOwnedDesc(backFaceDepthDescriptor, backFaceDepthDescriptorOwned);
    removeOwnedDesc(waterDepthLinearDescriptor, waterDepthLinearDescriptorOwned);
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

    // Remove ImGui descriptors owned by this widget and destroy associated
    // images, views, framebuffers and pipeline. Defer if GPU work is pending.
    auto removeDescIfOwned = [&](VkDescriptorSet &ds, bool &owned) {
        if (ds == VK_NULL_HANDLE) return;
        if (!owned) { ds = VK_NULL_HANDLE; return; }
        {
            VkDescriptorSet tmp = ds;
            a->deferDestroyUntilAllPending([tmp]() { ImGui_ImplVulkan_RemoveTexture(tmp); });
        }
        ds = VK_NULL_HANDLE;
        owned = false;
    };

    removeDescIfOwned(linearSceneDepthDescriptor, linearSceneDepthDescriptorOwned);
    removeDescIfOwned(linearBackFaceDepthDescriptor, linearBackFaceDepthDescriptorOwned);
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        removeDescIfOwned(linearShadowDepthDescriptor[i], linearShadowDepthDescriptorOwned[i]);
    }

    // Destroy images, views, memories and framebuffers created for linearization
    auto destroyImageAndMemory = [&](VkImageView &iv, VkImage &img, VkDeviceMemory &mem) {
        if (iv == VK_NULL_HANDLE && img == VK_NULL_HANDLE && mem == VK_NULL_HANDLE) return;
        VkImageView tmp_iv = iv;
        VkImage tmp_img = img;
        VkDeviceMemory tmp_mem = mem;
        VkDevice device = a->getDevice();
        if (a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [device, tmp_iv, tmp_img, tmp_mem, a]() {
                if (tmp_iv != VK_NULL_HANDLE) {
                    if (a->resources.removeImageView(tmp_iv))
                        vkDestroyImageView(device, tmp_iv, nullptr);
                }
                if (tmp_img != VK_NULL_HANDLE) {
                    if (a->resources.removeImage(tmp_img))
                        vkDestroyImage(device, tmp_img, nullptr);
                }
                if (tmp_mem != VK_NULL_HANDLE) {
                    if (a->resources.removeDeviceMemory(tmp_mem))
                        vkFreeMemory(device, tmp_mem, nullptr);
                }
            });
        } else {
            if (tmp_iv != VK_NULL_HANDLE) {
                if (a->resources.removeImageView(tmp_iv))
                    vkDestroyImageView(device, tmp_iv, nullptr);
            }
            if (tmp_img != VK_NULL_HANDLE) {
                if (a->resources.removeImage(tmp_img))
                    vkDestroyImage(device, tmp_img, nullptr);
            }
            if (tmp_mem != VK_NULL_HANDLE) {
                if (a->resources.removeDeviceMemory(tmp_mem))
                    vkFreeMemory(device, tmp_mem, nullptr);
            }
        }
        iv = VK_NULL_HANDLE;
        img = VK_NULL_HANDLE;
        mem = VK_NULL_HANDLE;
    };

    auto destroyFramebuffer = [&](VkFramebuffer &fb) {
        if (fb == VK_NULL_HANDLE) return;
        VkFramebuffer tmp = fb;
        VkDevice device = a->getDevice();
        if (a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                if (a->resources.removeFramebuffer(tmp)) vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
            });
        } else {
            if (a->resources.removeFramebuffer(tmp)) vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
        }
        fb = VK_NULL_HANDLE;
    };

    destroyFramebuffer(linearSceneFramebuffer);
    destroyFramebuffer(linearBackFaceFramebuffer);
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) destroyFramebuffer(linearShadowFramebuffer[i]);
    for (int i = 0; i < 6; ++i) destroyFramebuffer(linearCubeFaceFramebuffer[i]);

    destroyImageAndMemory(linearSceneDepthView, linearSceneDepthImage, linearSceneDepthMemory);
    destroyImageAndMemory(linearBackFaceDepthView, linearBackFaceDepthImage, linearBackFaceDepthMemory);
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        destroyImageAndMemory(linearShadowDepthView[i], linearShadowDepthImage[i], linearShadowDepthMemory[i]);
    }
    for (int i = 0; i < 6; ++i) {
        destroyImageAndMemory(linearCubeFaceDepthView[i], linearCubeFaceDepthImage[i], linearCubeFaceDepthMemory[i]);
    }

    // Keep pipeline/renderpass/layout/descriptor set until full cleanup()

    // Reset tracked sizes
    linearSceneWidth = 0;
    linearSceneHeight = 0;
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) linearShadowSize[i] = 0;
}

void RenderTargetsWidget::cleanup() {
    auto removeOwnedDesc = [&](VkDescriptorSet &ds, bool &owned) {
        if (ds == VK_NULL_HANDLE) return;
        if (!owned) { ds = VK_NULL_HANDLE; return; }
        if (app) {
            VkDescriptorSet tmp = ds;
            app->deferDestroyUntilAllPending([tmp](){ ImGui_ImplVulkan_RemoveTexture(tmp); });
        } else {
            ImGui_ImplVulkan_RemoveTexture(ds);
        }
        ds = VK_NULL_HANDLE;
        owned = false;
    };
    removeOwnedDesc(skyDescriptor, skyDescriptorOwned);
    removeOwnedDesc(solidColorDescriptor, solidColorDescriptorOwned);
    removeOwnedDesc(waterColorDescriptor, waterColorDescriptorOwned);
    removeOwnedDesc(solidDepthDescriptor, solidDepthDescriptorOwned);
    removeOwnedDesc(solid360Descriptor, solid360DescriptorOwned);
    removeOwnedDesc(cube360EquirectDescriptor, cube360EquirectDescriptorOwned);
    for (int i = 0; i < 6; ++i) removeOwnedDesc(cube360FaceDescriptor[i], cube360FaceDescriptorOwned[i]);
    for (int i = 0; i < 6; ++i) removeOwnedDesc(cube360FaceDepthDescriptor[i], cube360FaceDepthDescriptorOwned[i]);
    removeOwnedDesc(backFaceDepthDescriptor, backFaceDepthDescriptorOwned);
    removeOwnedDesc(waterDepthLinearDescriptor, waterDepthLinearDescriptorOwned);
    removeOwnedDesc(linearSceneDepthDescriptor, linearSceneDepthDescriptorOwned);
    removeOwnedDesc(linearBackFaceDepthDescriptor, linearBackFaceDepthDescriptorOwned);
    cube360EquirectRenderer.cleanup(app);
    // Destroy persistent staging buffers (VulkanApp::createBuffer registers them with resource manager)
    // Unmap persistent staging buffers; if GPU work is pending, defer unmap until safe
    if (stagingReadPtr && app && stagingReadBuffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(app->getDevice(), stagingReadBuffer.memory);
        stagingReadPtr = nullptr;
    }
    if (stagingUploadPtr && app && stagingUploadBuffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(app->getDevice(), stagingUploadBuffer.memory);
        stagingUploadPtr = nullptr;
    }
    // Drop local buffer handles; actual destruction managed by VulkanResourceManager
    stagingReadBuffer = {};
    stagingUploadBuffer = {};
    // Shadow cascade linear descriptors (use removeDesc to defer when necessary)
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        removeOwnedDesc(linearShadowDepthDescriptor[i], linearShadowDepthDescriptorOwned[i]);
    }
    // Note: `previewDescriptor` may reference descriptor sets owned by other
    // renderers (e.g. shadowMapper->getImGuiDescriptorSet). We must not call
    // ImGui_ImplVulkan_RemoveTexture() on descriptor sets we don't own. The
    // owned per-cascade descriptors are removed above in the loop.

    // Destroy any images / image views and persistent staging buffers that
    // this widget created. If GPU work is pending, defer destruction until
    // it is safe to free these Vulkan objects.
    VulkanApp* a = app;
    auto destroyImageAndMemory = [&](VkImageView &iv, VkImage &img, VkDeviceMemory &mem) {
        if (iv == VK_NULL_HANDLE && img == VK_NULL_HANDLE && mem == VK_NULL_HANDLE) return;
        VkImageView tmp_iv = iv;
        VkImage tmp_img = img;
        VkDeviceMemory tmp_mem = mem;
        VkDevice device = a ? a->getDevice() : VK_NULL_HANDLE;
        if (a && a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [device, tmp_iv, tmp_img, tmp_mem, a](){
                if (tmp_iv != VK_NULL_HANDLE) {
                    if (a->resources.removeImageView(tmp_iv))
                        vkDestroyImageView(device, tmp_iv, nullptr);
                }
                if (tmp_img != VK_NULL_HANDLE) {
                    if (a->resources.removeImage(tmp_img))
                        vkDestroyImage(device, tmp_img, nullptr);
                }
                if (tmp_mem != VK_NULL_HANDLE) {
                    if (a->resources.removeDeviceMemory(tmp_mem))
                        vkFreeMemory(device, tmp_mem, nullptr);
                }
            });
        } else {
            if (tmp_iv != VK_NULL_HANDLE) {
                if (a->resources.removeImageView(tmp_iv))
                    vkDestroyImageView(device, tmp_iv, nullptr);
            }
            if (tmp_img != VK_NULL_HANDLE) {
                if (a->resources.removeImage(tmp_img))
                    vkDestroyImage(device, tmp_img, nullptr);
            }
            if (tmp_mem != VK_NULL_HANDLE) {
                if (a->resources.removeDeviceMemory(tmp_mem))
                    vkFreeMemory(device, tmp_mem, nullptr);
            }
        }
        iv = VK_NULL_HANDLE;
        img = VK_NULL_HANDLE;
        mem = VK_NULL_HANDLE;
    };

    auto destroyBufferAndMemory = [&](Buffer &buf) {
        if (buf.buffer == VK_NULL_HANDLE) return;
        VkBuffer tmpBuf = buf.buffer;
        VkDeviceMemory tmpMem = buf.memory;
        VkDevice device = a ? a->getDevice() : VK_NULL_HANDLE;
        if (a && a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [device, tmpBuf, tmpMem, a](){
                if (a->resources.removeBuffer(tmpBuf)) vkDestroyBuffer(device, tmpBuf, nullptr);
                if (a->resources.removeDeviceMemory(tmpMem)) vkFreeMemory(device, tmpMem, nullptr);
            });
        } else {
            if (a->resources.removeBuffer(tmpBuf)) vkDestroyBuffer(device, tmpBuf, nullptr);
            if (a->resources.removeDeviceMemory(tmpMem)) vkFreeMemory(device, tmpMem, nullptr);
        }
        buf = {};
    };

    // Destroy linear debug images / views
    destroyImageAndMemory(linearSceneDepthView, linearSceneDepthImage, linearSceneDepthMemory);
    destroyImageAndMemory(linearBackFaceDepthView, linearBackFaceDepthImage, linearBackFaceDepthMemory);
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        destroyImageAndMemory(linearShadowDepthView[i], linearShadowDepthImage[i], linearShadowDepthMemory[i]);
    }

    // Destroy persistent staging buffers
    if (stagingReadPtr && app && stagingReadBuffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(app->getDevice(), stagingReadBuffer.memory);
        stagingReadPtr = nullptr;
    }
    if (stagingUploadPtr && app && stagingUploadBuffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(app->getDevice(), stagingUploadBuffer.memory);
        stagingUploadPtr = nullptr;
    }
    destroyBufferAndMemory(stagingReadBuffer);
    destroyBufferAndMemory(stagingUploadBuffer);

    // Destroy linearization pipeline, pipeline layout, descriptor set/layout,
    // framebuffers and renderpass created by this widget.
    auto destroyIf = [&](auto &handle, auto removeFn, auto destroyFn) {
        if (handle == VK_NULL_HANDLE) return;
        auto tmp = handle;
        VkDevice device = a ? a->getDevice() : VK_NULL_HANDLE;
        if (a && a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [device, tmp, a, removeFn, destroyFn]() mutable {
                if ((a->resources.*removeFn)(tmp)) destroyFn(device, tmp);
            });
        } else {
            if ((a->resources.*removeFn)(tmp)) destroyFn(a->getDevice(), tmp);
        }
        handle = VK_NULL_HANDLE;
    };

    // Framebuffers
    if (linearSceneFramebuffer != VK_NULL_HANDLE) {
        VkFramebuffer tmp = linearSceneFramebuffer;
        if (a && a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                if (a->resources.removeFramebuffer(tmp)) vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
            });
        } else {
            if (a->resources.removeFramebuffer(tmp)) vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
        }
        linearSceneFramebuffer = VK_NULL_HANDLE;
    }
    if (linearBackFaceFramebuffer != VK_NULL_HANDLE) {
        VkFramebuffer tmp = linearBackFaceFramebuffer;
        if (a && a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                if (a->resources.removeFramebuffer(tmp)) vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
            });
        } else {
            if (a->resources.removeFramebuffer(tmp)) vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
        }
        linearBackFaceFramebuffer = VK_NULL_HANDLE;
    }

    // Pipeline
    if (linearizePipeline != VK_NULL_HANDLE) {
        VkPipeline tmp = linearizePipeline;
        if (a && a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                if (a->resources.removePipeline(tmp)) vkDestroyPipeline(a->getDevice(), tmp, nullptr);
            });
        } else {
            if (a->resources.removePipeline(tmp)) vkDestroyPipeline(a->getDevice(), tmp, nullptr);
        }
        linearizePipeline = VK_NULL_HANDLE;
    }

    // Pipeline layout
    if (linearizePipelineLayout != VK_NULL_HANDLE) {
        VkPipelineLayout tmp = linearizePipelineLayout;
        if (a && a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                if (a->resources.removePipelineLayout(tmp)) vkDestroyPipelineLayout(a->getDevice(), tmp, nullptr);
            });
        } else {
            if (a->resources.removePipelineLayout(tmp)) vkDestroyPipelineLayout(a->getDevice(), tmp, nullptr);
        }
        linearizePipelineLayout = VK_NULL_HANDLE;
    }

    // Descriptor set + layout
    if (linearizeDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorSet tmp = linearizeDescriptorSet;
        if (a && a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                a->resources.removeDescriptorSet(tmp);
                // Descriptor sets are freed via pool management elsewhere; leave as-is
            });
        } else {
            a->resources.removeDescriptorSet(tmp);
        }
        linearizeDescriptorSet = VK_NULL_HANDLE;
    }
    if (linearizeDescriptorSetLayout != VK_NULL_HANDLE) {
        VkDescriptorSetLayout tmp = linearizeDescriptorSetLayout;
        if (a && a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                if (a->resources.removeDescriptorSetLayout(tmp)) vkDestroyDescriptorSetLayout(a->getDevice(), tmp, nullptr);
            });
        } else {
            if (a->resources.removeDescriptorSetLayout(tmp)) vkDestroyDescriptorSetLayout(a->getDevice(), tmp, nullptr);
        }
        linearizeDescriptorSetLayout = VK_NULL_HANDLE;
    }

    // Render pass
    if (linearizeRenderPass != VK_NULL_HANDLE) {
        VkRenderPass tmp = linearizeRenderPass;
        if (a && a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                if (a->resources.removeRenderPass(tmp)) vkDestroyRenderPass(a->getDevice(), tmp, nullptr);
            });
        } else {
            if (a->resources.removeRenderPass(tmp)) vkDestroyRenderPass(a->getDevice(), tmp, nullptr);
        }
        linearizeRenderPass = VK_NULL_HANDLE;
    }

    // Destroy widget-owned sampler
    if (widgetSampler != VK_NULL_HANDLE) {
        VkSampler tmp = widgetSampler;
        if (a && a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                if (a->resources.removeSampler(tmp)) vkDestroySampler(a->getDevice(), tmp, nullptr);
            });
        } else {
            if (a->resources.removeSampler(tmp)) vkDestroySampler(a->getDevice(), tmp, nullptr);
        }
        widgetSampler = VK_NULL_HANDLE;
    }
}

void RenderTargetsWidget::updateDescriptors(uint32_t frameIndex) {
    if (!sceneRenderer || !sceneRenderer->waterRenderer) return;

    // Debug: print resource counts before cleanup (helps track leaks)
    if (app) {
        auto &rm = app->resources;
        size_t imgCount = rm.getImageMap().size();
        size_t ivCount = rm.getImageViewMap().size();
        size_t bufCount = rm.getBufferMap().size();
        size_t memCount = rm.getDeviceMemoryMap().size();
        size_t descSetCount = rm.getDescriptorSetMap().size();
        fprintf(stderr, "[RenderTargetsWidget] updateDescriptors START frame=%d resources pre-cleanup images=%zu imageViews=%zu buffers=%zu memories=%zu descSets=%zu\n",
                currentFrame, imgCount, ivCount, bufCount, memCount, descSetCount);
    }

    // Keep existing descriptor sets alive across frames. Create ImGui texture
    // descriptors only when the corresponding descriptor is null to avoid
    // allocating a new descriptor every frame (which was causing the growth
    // seen in the logs).



    switch (selectedPreview) {
        case PreviewTarget::Sky: {
            if (skyDescriptor == VK_NULL_HANDLE) {
                skyDescriptor = ImGui_ImplVulkan_AddTexture(widgetSampler, skyRenderer->getSkyView(frameIndex), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                skyDescriptorOwned = true;
            }
            
        } break;

        case PreviewTarget::SolidColor: {
            if (solidColorDescriptor == VK_NULL_HANDLE) {
                solidColorDescriptor = ImGui_ImplVulkan_AddTexture(widgetSampler, solidRenderer->getColorView(frameIndex), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                solidColorDescriptorOwned = true;
            }
            
        } break;

        case PreviewTarget::SolidDepth: {
            if (solidDepthDescriptor == VK_NULL_HANDLE) {
                VkSampler depthSampler = widgetSampler;
                // Sample the previously-produced frame's depth (one-frame latency)
                // to avoid binding the current-frame attachment before it's rendered.
                uint32_t producerFrame = (frameIndex + 1) % 2;
                solidDepthDescriptor = ImGui_ImplVulkan_AddTexture(depthSampler,  solidRenderer->getDepthView(producerFrame), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                solidDepthDescriptorOwned = true;
            }
            
        } break;

        case PreviewTarget::Solid360Equirect: {
            cube360EquirectRenderer.render(app, widgetSampler, sceneRenderer->solid360Renderer->getSolid360View());
            if (cube360EquirectDescriptor == VK_NULL_HANDLE) {
                cube360EquirectDescriptor = ImGui_ImplVulkan_AddTexture(widgetSampler, cube360EquirectRenderer.getEquirectView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                cube360EquirectDescriptorOwned = true;
                
            }
            
        } break;

        case PreviewTarget::Solid360Cube: {
            uint32_t f = static_cast<uint32_t>(this->selectedCubeFaceIndex);
            VkImageView faceView = (sceneRenderer && sceneRenderer->solid360Renderer) ? sceneRenderer->solid360Renderer->getCube360FaceView(f) : VK_NULL_HANDLE;
            if (faceView != VK_NULL_HANDLE && cube360FaceDescriptor[f] == VK_NULL_HANDLE) {
                cube360FaceDescriptor[f] = ImGui_ImplVulkan_AddTexture(widgetSampler, faceView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                cube360FaceDescriptorOwned[f] = true;
            }
        } break;

        case PreviewTarget::Solid360DepthCube: {
            uint32_t f = static_cast<uint32_t>(this->selectedCubeFaceIndex);
            VkImageView depthView = (sceneRenderer && sceneRenderer->solid360Renderer) ? sceneRenderer->solid360Renderer->getCube360DepthView(f) : VK_NULL_HANDLE;
            if (depthView != VK_NULL_HANDLE && cube360FaceDepthDescriptor[f] == VK_NULL_HANDLE) {
                VkSampler depthSampler = widgetSampler;
                float nearP = 0.1f, farP = 1000.0f;
                if (settings) { nearP = settings->nearPlane; farP = settings->farPlane; }
                // Linearize the depth for this cubemap face into its own per-face linear target
                runLinearizePass(app, sceneRenderer->solid360Renderer->getCube360DepthImage(), depthView, widgetSampler, widgetSampler,
                                 linearCubeFaceDepthView[f], linearCubeFaceFramebuffer[f],
                                 cube360FaceDepthDescriptor[f], cube360FaceDepthDescriptorOwned[f],
                                 static_cast<uint32_t>(cachedWidth), static_cast<uint32_t>(cachedHeight), nearP, farP, 0.0f, f);
            }
        } break;

        case PreviewTarget::WaterColor: {
            VkImageView waterView = (sceneRenderer && sceneRenderer->waterRenderer) ? sceneRenderer->waterRenderer->getWaterDepthView(frameIndex) : VK_NULL_HANDLE;
            if (waterView != VK_NULL_HANDLE && waterColorDescriptor == VK_NULL_HANDLE) {
                waterColorDescriptor = ImGui_ImplVulkan_AddTexture(widgetSampler, waterView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                waterColorDescriptorOwned = true;
            }
        } break;

        default:
            break;
    }


    // Water linear depth preview: instead of using the alpha-swizzle view
    // (which is a hack), run the GPU linearize pass on the actual water
    // geometry depth image (D32_SFLOAT) and expose the resulting color view
    // to ImGui. We reuse the `linearSceneDepthView` target for this simple
    // preview to avoid duplicating target creation.

    // CPU readback removed: previews use renderer-provided GPU image views.
    // Full GPU linearization pass: when possible render depth -> R8 target
    // via a small fullscreen pass and expose that R8 image to ImGui. This
    // is preferred because it avoids CPU readback while producing a proper
    // linearized grayscale depth preview for perspective projections.
    if (app && cachedWidth > 0 && cachedHeight > 0) {
        VkDevice device = app->getDevice();

        // Size-dependent linearize targets (images/views/framebuffers) are
        // created in `init(app, width, height)` so they are not recreated
        // every frame. Ensure `setFrameInfo()` calls `init()` when the size
        // changes so these targets are available here.

        // linearize descriptor set layout created once in init()

        // linearize pipeline layout created once in init()

        // pipeline and descriptor set are created once in init()

        // Prepare a sampler for sampling depth textures (use non-compare widget sampler)
        VkSampler depthSampler = widgetSampler;

        // Run the pass for scene depth (use perspective linearization)
        if (solidRenderer && linearizePipeline != VK_NULL_HANDLE && linearSceneFramebuffer != VK_NULL_HANDLE) {
            // Sample the previously-produced frame's depth (one-frame latency)
            // to avoid sampling an attachment that will be written later in
            // the same frame while ImGui is being built.
            uint32_t producerFrame = (frameIndex + 1) % 2;
            VkImageView src = solidRenderer->getDepthView(producerFrame);
            fprintf(stderr, "[RenderTargetsWidget] Scene linearize check: pipeline=%p descSet=%p fb=%p view=%p src=%p producerFrame=%u\n",
                    (void*)linearizePipeline, (void*)linearizeDescriptorSet, (void*)linearSceneFramebuffer, (void*)linearSceneDepthView, (void*)src, (unsigned)producerFrame);
                if (src != VK_NULL_HANDLE) {
                float nearP = 0.1f, farP = 1000.0f;
                if (settings) { nearP = settings->nearPlane; farP = settings->farPlane; }
                runLinearizePass(app, solidRenderer->getDepthImage(producerFrame), src, widgetSampler, widgetSampler, linearSceneDepthView, linearSceneFramebuffer,
                                 linearSceneDepthDescriptor, linearSceneDepthDescriptorOwned,
                                 static_cast<uint32_t>(cachedWidth), static_cast<uint32_t>(cachedHeight), nearP, farP, 0.0f);
                fprintf(stderr, "[RenderTargetsWidget] Scene linearize: pass completed\n");
            }
        }

        // Back-face depth pass
        // Back-face depth pass (use perspective linearization)
        if (sceneRenderer && sceneRenderer->waterRenderer && linearizePipeline != VK_NULL_HANDLE && linearBackFaceFramebuffer != VK_NULL_HANDLE) {
            // Use previous producer frame for the back-face source to avoid
            // sampling images that may still be in-flight.
            uint32_t producerFrame = (frameIndex + 1) % 2;
            VkImageView src = (sceneRenderer && sceneRenderer->backFaceRenderer) ? sceneRenderer->backFaceRenderer->getBackFaceDepthView(producerFrame) : VK_NULL_HANDLE;
            fprintf(stderr, "[RenderTargetsWidget] Backface linearize check: pipeline=%p descSet=%p fb=%p view=%p src=%p producerFrame=%u\n",
                    (void*)linearizePipeline, (void*)linearizeDescriptorSet, (void*)linearBackFaceFramebuffer, (void*)linearBackFaceDepthView, (void*)src, (unsigned)producerFrame);
                if (src != VK_NULL_HANDLE) {
                float nearP = 0.1f, farP = 1000.0f;
                if (settings) { nearP = settings->nearPlane; farP = settings->farPlane; }
                runLinearizePass(app, sceneRenderer->backFaceRenderer->getBackFaceDepthImage(producerFrame), src, widgetSampler, widgetSampler, linearBackFaceDepthView, linearBackFaceFramebuffer,
                                 linearBackFaceDepthDescriptor, linearBackFaceDepthDescriptorOwned,
                                 static_cast<uint32_t>(cachedWidth), static_cast<uint32_t>(cachedHeight), nearP, farP, 0.0f);
                fprintf(stderr, "[RenderTargetsWidget] Backface linearize: pass completed\n");
            }
        }

        // Water front-face depth pass (linearize the water geometry depth buffer)
        // Water front-face depth pass (linearize the water geometry depth buffer)
        if (sceneRenderer && sceneRenderer->waterRenderer && linearizePipeline != VK_NULL_HANDLE && linearSceneFramebuffer != VK_NULL_HANDLE) {
            // The water geometry depth is written during the main render pass for
            // the current CPU frame. The widget runs its linearize pass while
            // building ImGui (before the main render pass is submitted), so
            // sampling the *current* frame's depth can read undefined/attachment
            // layout contents. Use the previous producer frame's depth view so
            // we sample a completed image (one-frame latency) and avoid hazards.
            uint32_t producerFrame = (frameIndex + 1) % 2;
            VkImageView src = sceneRenderer->waterRenderer->getWaterGeomDepthView(producerFrame);
            fprintf(stderr, "[RenderTargetsWidget] Water linearize check: pipeline=%p descSet=%p fb=%p view=%p src=%p producerFrame=%u\n",
                    (void*)linearizePipeline, (void*)linearizeDescriptorSet, (void*)linearSceneFramebuffer, (void*)linearSceneDepthView, (void*)src, (unsigned)producerFrame);
            if (src != VK_NULL_HANDLE) {
                float nearP = 0.1f, farP = 1000.0f;
                if (settings) { nearP = settings->nearPlane; farP = settings->farPlane; }
                runLinearizePass(app, sceneRenderer->waterRenderer->getWaterGeomDepthImage(producerFrame), src, widgetSampler, widgetSampler, linearSceneDepthView, linearSceneFramebuffer,
                                 linearSceneDepthDescriptor, linearSceneDepthDescriptorOwned,
                                 static_cast<uint32_t>(cachedWidth), static_cast<uint32_t>(cachedHeight), nearP, farP, 1.0f);
                fprintf(stderr, "[RenderTargetsWidget] Water linearize: pass completed\n");

                // Expose the linearized image as the water-depth preview descriptor.
                if (waterDepthLinearDescriptor == VK_NULL_HANDLE && linearSceneDepthDescriptor != VK_NULL_HANDLE) {
                    // Point to the same descriptor but mark as not-owned to avoid double-free
                    waterDepthLinearDescriptor = linearSceneDepthDescriptor;
                    waterDepthLinearDescriptorOwned = false;
                }
            }
        }
        // Shadow cascade linearization: create per-cascade RGBA targets and
        // run the same linearize pass used for scene/backface/water so all
        // depth previews are produced consistently.
        if (shadowMapper && linearizePipeline != VK_NULL_HANDLE && linearizeRenderPass != VK_NULL_HANDLE) {
            uint32_t shadowSize = shadowMapper->getShadowMapSize();
            for (int c = 0; c < SHADOW_CASCADE_COUNT; ++c) {
                VkImageView src = shadowMapper->getShadowMapView(c);
                if (src != VK_NULL_HANDLE && linearShadowFramebuffer[c] != VK_NULL_HANDLE) {
                    float nearP = 0.0f, farP = 1.0f;
                    runLinearizePass(app, shadowMapper->getDepthImage(c), src, widgetSampler, widgetSampler, linearShadowDepthView[c], linearShadowFramebuffer[c],
                                     linearShadowDepthDescriptor[c], linearShadowDepthDescriptorOwned[c], shadowSize, shadowSize, nearP, farP, 1.0f);
                    fprintf(stderr, "[RenderTargetsWidget] Shadow linearize: cascade=%d done\n", c);
                }
            }
        }
    }
    // Water back-face depth (volume thickness pre-pass — D32_SFLOAT)
    if (shadowMapper) {
        VkImageView bfView = (sceneRenderer && sceneRenderer->backFaceRenderer) ? sceneRenderer->backFaceRenderer->getBackFaceDepthView(frameIndex) : VK_NULL_HANDLE;
        if (bfView != VK_NULL_HANDLE && backFaceDepthDescriptor == VK_NULL_HANDLE) {
            backFaceDepthDescriptor = ImGui_ImplVulkan_AddTexture(
                widgetSampler, bfView,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            backFaceDepthDescriptorOwned = true;
        }
    }

    // Alias linear-depth previews to renderer-provided depth image views so
    // the widget displays depth entirely via GPU sampling (no CPU readback).
    // Scene linear depth: if we didn't create a GPU-linearized image above,
    // alias to the solid renderer depth view so we still show something.
    if (linearSceneDepthDescriptor == VK_NULL_HANDLE && solidRenderer) {
        // Try to produce a GPU-linearized RGBA preview first. If linearization
        // fails or is unavailable, fall back to aliasing the raw depth view.
        uint32_t producerFrame = (frameIndex + 1) % 2;
        VkImageView sceneDepthView = solidRenderer->getDepthView(producerFrame);
        VkImage sceneDepthImage = solidRenderer->getDepthImage(producerFrame);
        if (sceneDepthView != VK_NULL_HANDLE) {
            bool linearized = false;
            // Only attempt linearize if we have the pipeline and target framebuffer
            if (linearizePipeline != VK_NULL_HANDLE && linearSceneFramebuffer != VK_NULL_HANDLE && sceneDepthImage != VK_NULL_HANDLE) {
                float nearP = 0.1f, farP = 1000.0f;
                if (settings) { nearP = settings->nearPlane; farP = settings->farPlane; }
                // runLinearizePass will set `linearSceneDepthDescriptor` on success
                linearized = runLinearizePass(app, sceneDepthImage, sceneDepthView, widgetSampler, widgetSampler,
                                              linearSceneDepthView, linearSceneFramebuffer,
                                              linearSceneDepthDescriptor, linearSceneDepthDescriptorOwned,
                                              static_cast<uint32_t>(cachedWidth), static_cast<uint32_t>(cachedHeight), nearP, farP, 0.0f);
            }
            if (!linearized) {
                // Fallback: alias raw depth view (may appear incorrect on some GPUs)
                linearSceneDepthDescriptor = ImGui_ImplVulkan_AddTexture(widgetSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                linearSceneDepthDescriptorOwned = true;
            }
        }
    }
    // Back-face depth (water): alias to water back-face depth view
    VkImageView bfView2 = (sceneRenderer && sceneRenderer->backFaceRenderer) ? sceneRenderer->backFaceRenderer->getBackFaceDepthView(frameIndex) : VK_NULL_HANDLE;
    if (linearBackFaceDepthDescriptor == VK_NULL_HANDLE && bfView2 != VK_NULL_HANDLE) {
        VkSampler depthSampler = widgetSampler;
        linearBackFaceDepthDescriptor = ImGui_ImplVulkan_AddTexture(depthSampler, bfView2, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        linearBackFaceDepthDescriptorOwned = true;
    }

    // Choose a single preview descriptor according to the current selection.
    previewDescriptor = VK_NULL_HANDLE;
    switch (selectedPreview) {
        case PreviewTarget::Sky: 
            previewDescriptor = skyDescriptor; 
            break;
        case PreviewTarget::Solid360Cube: 
            previewDescriptor = cube360FaceDescriptor[this->selectedCubeFaceIndex]; 
            break;
        case PreviewTarget::Solid360DepthCube:
            previewDescriptor = cube360FaceDepthDescriptor[this->selectedCubeFaceIndex];
            break;
        case PreviewTarget::Solid360Equirect: 
            previewDescriptor = cube360EquirectDescriptor; 
            break;            
        case PreviewTarget::SolidColor: 
            previewDescriptor = solidColorDescriptor; 
            break;
        // Prefer the GPU-linearized scene depth if available, otherwise fall back to raw depth view
        case PreviewTarget::SolidDepth:
            if (linearSceneDepthDescriptor != VK_NULL_HANDLE) previewDescriptor = linearSceneDepthDescriptor;
            else previewDescriptor = solidDepthDescriptor;
            break;
        case PreviewTarget::WaterColor: 
            previewDescriptor = waterColorDescriptor; 
            break;
        case PreviewTarget::WaterDepth: 
            previewDescriptor = waterDepthLinearDescriptor; 
            break;
        // Prefer the GPU-linearized back-face depth if available, otherwise fall back to raw back-face depth view
        case PreviewTarget::BackFaceColor:
            if (linearBackFaceDepthDescriptor != VK_NULL_HANDLE) previewDescriptor = linearBackFaceDepthDescriptor;
            else previewDescriptor = backFaceDepthDescriptor;
            break;
        case PreviewTarget::LinearSceneDepth: 
            previewDescriptor = linearSceneDepthDescriptor; 
            break;
        case PreviewTarget::BackFaceDepth: 
            previewDescriptor = linearBackFaceDepthDescriptor; 
            break;
        case PreviewTarget::ShadowCascade:
            if (shadowViewMode == RenderTargetsWidget::ShadowViewMode::Linearized) {
                if (linearShadowDepthDescriptor[selectedShadowCascade] != VK_NULL_HANDLE)
                    previewDescriptor = linearShadowDepthDescriptor[selectedShadowCascade];
                else if (shadowMapper) previewDescriptor = shadowMapper->getImGuiDescriptorSet(selectedShadowCascade);
            } else {
                if (shadowMapper) previewDescriptor = shadowMapper->getImGuiDescriptorSet(selectedShadowCascade);
            }
            break;
        default: 
            previewDescriptor = VK_NULL_HANDLE; 
            break;
    }

    // Periodic debug: print resource counts after update (throttled)
    static int dbgCounter = 0;
    if (app && (++dbgCounter % 120) == 0) {
        auto &rm = app->resources;
        size_t imgCount = rm.getImageMap().size();
        size_t ivCount = rm.getImageViewMap().size();
        size_t bufCount = rm.getBufferMap().size();
        size_t memCount = rm.getDeviceMemoryMap().size();
        size_t descSetCount = rm.getDescriptorSetMap().size();
        fprintf(stderr, "[RenderTargetsWidget] updateDescriptors END frame=%d resources post-update images=%zu imageViews=%zu buffers=%zu memories=%zu descSets=%zu\n",
                currentFrame, imgCount, ivCount, bufCount, memCount, descSetCount);
    }
}

void RenderTargetsWidget::render() {
    ImGui::Begin("Render Targets", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    if (!sceneRenderer || !sceneRenderer->waterRenderer || !solidRenderer) {
        ImGui::TextUnformatted("Renderers not available.");
        ImGui::End();
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
        fprintf(stderr, "RenderTargetsWidget: selectedPreview=%d selectedShadowCascade=%d showAllCascades=%d selectedCubeFace=%d\n",
            static_cast<int>(selectedPreview), selectedShadowCascade, showAllCascades, selectedCubeFaceIndex);

        // Render only the selected preview using a single preview descriptor
        VkDescriptorSet ds = previewDescriptor;
        ImVec2 imgSize = previewSize;
        bool available = (ds != VK_NULL_HANDLE);

        if (available) ImGui::Image((ImTextureID)ds, imgSize); else ImGui::Text("Preview unavailable");
    ImGui::Separator();



    // Optionally show all cascades (in selected shadow view mode)
    // Only show the full cascade grid when the shadow cascade preview is selected.
    if (selectedPreview == PreviewTarget::ShadowCascade && showAllCascades && shadowMapper) {
        float shadowSize = PREVIEW_WIDTH;
        for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
            ImGui::Text("Shadow Cascade %d", i);
            if (shadowViewMode == RenderTargetsWidget::ShadowViewMode::Linearized && linearShadowDepthDescriptor[i] != VK_NULL_HANDLE) {
                ImGui::Image((ImTextureID)linearShadowDepthDescriptor[i], ImVec2(shadowSize, shadowSize));
            } else {
                VkDescriptorSet ds = shadowMapper->getImGuiDescriptorSet(i);
                if (ds != VK_NULL_HANDLE) ImGui::Image((ImTextureID)ds, ImVec2(shadowSize, shadowSize));
            }
            ImGui::Separator();
        }
    }

    ImGui::End();
}
