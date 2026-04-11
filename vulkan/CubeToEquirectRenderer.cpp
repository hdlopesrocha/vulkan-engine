#include "CubeToEquirectRenderer.hpp"
#include "VulkanApp.hpp"
#include "../utils/FileReader.hpp"
#include <stdexcept>
#include <array>

CubeToEquirectRenderer::CubeToEquirectRenderer() {}
CubeToEquirectRenderer::~CubeToEquirectRenderer() {}

void CubeToEquirectRenderer::cleanup(VulkanApp* app) {
    if (!app) return;
    VkDevice device = app->getDevice();

    auto destroyImageAndMemory = [&](VkImageView &iv, VkImage &img, VkDeviceMemory &mem) {
        if (iv == VK_NULL_HANDLE && img == VK_NULL_HANDLE && mem == VK_NULL_HANDLE) return;
        VkImageView tmp_iv = iv;
        VkImage tmp_img = img;
        VkDeviceMemory tmp_mem = mem;
        if (app->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = app->getCurrentFrame();
            if (fi < app->inFlightFences.size()) f = app->inFlightFences[fi];
            app->deferDestroyUntilFence(f, [tmp_iv, tmp_img, tmp_mem, device, app]() {
                if (tmp_iv != VK_NULL_HANDLE) {
                    app->resources.removeImageView(tmp_iv);
                    vkDestroyImageView(device, tmp_iv, nullptr);
                }
                if (tmp_img != VK_NULL_HANDLE) {
                    app->resources.removeImage(tmp_img);
                    vkDestroyImage(device, tmp_img, nullptr);
                }
                if (tmp_mem != VK_NULL_HANDLE) {
                    app->resources.removeDeviceMemory(tmp_mem);
                    vkFreeMemory(device, tmp_mem, nullptr);
                }
            });
        } else {
            if (tmp_iv != VK_NULL_HANDLE) {
                app->resources.removeImageView(tmp_iv);
                vkDestroyImageView(device, tmp_iv, nullptr);
            }
            if (tmp_img != VK_NULL_HANDLE) {
                app->resources.removeImage(tmp_img);
                vkDestroyImage(device, tmp_img, nullptr);
            }
            if (tmp_mem != VK_NULL_HANDLE) {
                app->resources.removeDeviceMemory(tmp_mem);
                vkFreeMemory(device, tmp_mem, nullptr);
            }
        }
        iv = VK_NULL_HANDLE;
        img = VK_NULL_HANDLE;
        mem = VK_NULL_HANDLE;
    };

    auto destroyVkObject = [&](auto &handle, auto removeFn, auto destroyFn) {
        if (handle == VK_NULL_HANDLE) return;
        auto tmp = handle;
        if (app->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = app->getCurrentFrame();
            if (fi < app->inFlightFences.size()) f = app->inFlightFences[fi];
            app->deferDestroyUntilFence(f, [tmp, app, removeFn, destroyFn, device]() {
                (app->resources.*removeFn)(tmp);
                destroyFn(device, tmp, nullptr);
            });
        } else {
            (app->resources.*removeFn)(tmp);
            destroyFn(device, tmp, nullptr);
        }
        handle = VK_NULL_HANDLE;
    };

    destroyImageAndMemory(cube360EquirectView, cube360EquirectImage, cube360EquirectMemory);
    destroyVkObject(cube360EquirectFramebuffer, &VulkanResourceManager::removeFramebuffer, vkDestroyFramebuffer);
    destroyVkObject(cube360EquirectRenderPass, &VulkanResourceManager::removeRenderPass, vkDestroyRenderPass);
    destroyVkObject(cube360EquirectPipeline, &VulkanResourceManager::removePipeline, vkDestroyPipeline);
    destroyVkObject(cube360EquirectPipelineLayout, &VulkanResourceManager::removePipelineLayout, vkDestroyPipelineLayout);
    destroyVkObject(cube360EquirectVertModule, &VulkanResourceManager::removeShaderModule, vkDestroyShaderModule);
    destroyVkObject(cube360EquirectFragModule, &VulkanResourceManager::removeShaderModule, vkDestroyShaderModule);
    destroyVkObject(cube360EquirectDescriptorSetLayout, &VulkanResourceManager::removeDescriptorSetLayout, vkDestroyDescriptorSetLayout);

    if (cube360EquirectSampleDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorSet tmp = cube360EquirectSampleDescriptorSet;
        if (app->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = app->getCurrentFrame();
            if (fi < app->inFlightFences.size()) f = app->inFlightFences[fi];
            app->deferDestroyUntilFence(f, [tmp, app]() {
                app->resources.removeDescriptorSet(tmp);
            });
        } else {
            app->resources.removeDescriptorSet(tmp);
        }
        cube360EquirectSampleDescriptorSet = VK_NULL_HANDLE;
    }
}

void CubeToEquirectRenderer::createRenderPass(VulkanApp* app) {
    if (cube360EquirectRenderPass != VK_NULL_HANDLE) return;
    VkDevice device = app->getDevice();

    VkAttachmentDescription colorAtt{};
    colorAtt.format = app->getSwapchainImageFormat();
    colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

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
    rpci.pAttachments = &colorAtt;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies = &dep;

    if (vkCreateRenderPass(device, &rpci, nullptr, &cube360EquirectRenderPass) != VK_SUCCESS)
        throw std::runtime_error("Failed to create cube360 equirect render pass!");
    app->resources.addRenderPass(cube360EquirectRenderPass, "CubeToEquirectRenderer: renderPass");
}

void CubeToEquirectRenderer::createPipeline(VulkanApp* app) {
    if (cube360EquirectPipeline != VK_NULL_HANDLE) return;
    VkDevice device = app->getDevice();

    if (cube360EquirectVertModule == VK_NULL_HANDLE) {
        cube360EquirectVertModule = app->createShaderModule(FileReader::readFile("shaders/fullscreen.vert.spv"));
    }
    if (cube360EquirectFragModule == VK_NULL_HANDLE) {
        cube360EquirectFragModule = app->createShaderModule(FileReader::readFile("shaders/cubemap_to_equirect.frag.spv"));
    }
    if (cube360EquirectDescriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &binding;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &cube360EquirectDescriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create cube360 equirect descriptor set layout!");
        app->resources.addDescriptorSetLayout(cube360EquirectDescriptorSetLayout, "CubeToEquirectRenderer: descriptorSetLayout");
    }
    if (cube360EquirectPipelineLayout == VK_NULL_HANDLE) {
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(float) * 2;
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &cube360EquirectDescriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &cube360EquirectPipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create cube360 equirect pipeline layout!");
        app->resources.addPipelineLayout(cube360EquirectPipelineLayout, "CubeToEquirectRenderer: pipelineLayout");
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};
    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = cube360EquirectVertModule;
    shaderStages[0].pName = "main";
    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = cube360EquirectFragModule;
    shaderStages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth = 1.0f;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

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
    cb.attachmentCount = 1;
    cb.pAttachments = &ca;

    std::array<VkDynamicState, 2> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dyn.pDynamicStates = dynamicStates.data();

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.stageCount = static_cast<uint32_t>(shaderStages.size());
    pi.pStages = shaderStages.data();
    pi.pVertexInputState = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vp;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &dyn;
    pi.layout = cube360EquirectPipelineLayout;
    pi.renderPass = cube360EquirectRenderPass;
    pi.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &cube360EquirectPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create cube360 equirect pipeline!");
    app->resources.addPipeline(cube360EquirectPipeline, "CubeToEquirectRenderer: pipeline");
}

void CubeToEquirectRenderer::createOutputTarget(VulkanApp* app) {
    if (cube360EquirectImage != VK_NULL_HANDLE) return;
    VkDevice device = app->getDevice();

    app->createImage(EQ_WIDTH, EQ_HEIGHT, VK_FORMAT_R8G8B8A8_UNORM,
                     VK_IMAGE_TILING_OPTIMAL, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, cube360EquirectImage, cube360EquirectMemory);

    VkImageViewCreateInfo iv{};
    iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    iv.image = cube360EquirectImage;
    iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv.format = VK_FORMAT_R8G8B8A8_UNORM;
    iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    iv.subresourceRange.baseMipLevel = 0;
    iv.subresourceRange.levelCount = 1;
    iv.subresourceRange.baseArrayLayer = 0;
    iv.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &iv, nullptr, &cube360EquirectView) != VK_SUCCESS)
        throw std::runtime_error("Failed to create cube360 equirect image view!");
    app->resources.addImageView(cube360EquirectView, "CubeToEquirectRenderer: imageView");

    VkFramebufferCreateInfo fb{};
    fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb.renderPass = cube360EquirectRenderPass;
    fb.attachmentCount = 1;
    fb.pAttachments = &cube360EquirectView;
    fb.width = EQ_WIDTH;
    fb.height = EQ_HEIGHT;
    fb.layers = 1;
    if (vkCreateFramebuffer(device, &fb, nullptr, &cube360EquirectFramebuffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to create cube360 equirect framebuffer!");
    app->resources.addFramebuffer(cube360EquirectFramebuffer, "CubeToEquirectRenderer: framebuffer");
}

void CubeToEquirectRenderer::createDescriptorResources(VulkanApp* app, VkSampler sampler, VkImageView cubeMapView) {
    if (cube360EquirectDescriptorSetLayout == VK_NULL_HANDLE || cube360EquirectPipelineLayout == VK_NULL_HANDLE) {
        createPipeline(app);
    }
    if (cube360EquirectSampleDescriptorSet == VK_NULL_HANDLE) {
        cube360EquirectSampleDescriptorSet = app->createDescriptorSet(cube360EquirectDescriptorSetLayout);
    }

    VkDescriptorImageInfo descInfo{};
    descInfo.sampler = sampler;
    descInfo.imageView = cubeMapView;
    descInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = cube360EquirectSampleDescriptorSet;
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &descInfo;
    app->updateDescriptorSet({ write });
}

void CubeToEquirectRenderer::ensureResources(VulkanApp* app) {
    createRenderPass(app);
    createPipeline(app);
    createOutputTarget(app);
}

void CubeToEquirectRenderer::render(VulkanApp* app, VkSampler sampler, VkImageView cubeMapView) {
    if (!app || sampler == VK_NULL_HANDLE || cubeMapView == VK_NULL_HANDLE) return;
    ensureResources(app);
    createDescriptorResources(app, sampler, cubeMapView);

    app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        VkClearValue clear{};
        clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = cube360EquirectRenderPass;
        rp.framebuffer = cube360EquirectFramebuffer;
        rp.renderArea.offset = {0, 0};
        rp.renderArea.extent = { EQ_WIDTH, EQ_HEIGHT };
        rp.clearValueCount = 1;
        rp.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cube360EquirectPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cube360EquirectPipelineLayout,
                                0, 1, &cube360EquirectSampleDescriptorSet, 0, nullptr);
        float resolution[2] = { static_cast<float>(EQ_WIDTH), static_cast<float>(EQ_HEIGHT) };
        vkCmdPushConstants(cmd, cube360EquirectPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(resolution), resolution);
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(EQ_WIDTH), static_cast<float>(EQ_HEIGHT), 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{{0, 0}, { EQ_WIDTH, EQ_HEIGHT }};
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
    });
}
