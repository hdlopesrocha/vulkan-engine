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

static constexpr uint32_t CUBE360_EQUIRECT_WIDTH = 1024;
static constexpr uint32_t CUBE360_EQUIRECT_HEIGHT = 512;

RenderTargetsWidget::RenderTargetsWidget(VulkanApp* app, SceneRenderer* scene, SolidRenderer* solid, SkyRenderer* sky,
                                                                                 ShadowRenderer* shadow, ShadowParams* shadowParams, Settings* settings)
        : Widget("Render Targets"), app(app), sceneRenderer(scene), solidRenderer(solid), skyRenderer(sky),
            shadowMapper(shadow), shadowParams(shadowParams), settings(settings) {
}

bool RenderTargetsWidget::runLinearizePass(VulkanApp* app, VkImageView srcView, VkSampler srcSampler, VkSampler previewSampler,
                                          VkImageView dstView, VkFramebuffer dstFb,
                                          VkDescriptorSet &dstDescriptor, bool &dstDescriptorOwned,
                                          uint32_t width, uint32_t height,
                                          float zNear, float zFar, float mode) {
    if (!app || srcView == VK_NULL_HANDLE || dstView == VK_NULL_HANDLE || dstFb == VK_NULL_HANDLE) return false;
    if (linearizePipeline == VK_NULL_HANDLE || linearizeDescriptorSet == VK_NULL_HANDLE || linearizePipelineLayout == VK_NULL_HANDLE) return false;

    fprintf(stderr, "[RenderTargetsWidget] runLinearizePass: src=%p dst=%p fb=%p size=%ux%u mode=%f\n", (void*)srcView, (void*)dstView, (void*)dstFb, (unsigned)width, (unsigned)height, mode);

    VkDescriptorImageInfo di{};
    di.sampler = srcSampler;
    di.imageView = srcView;
    di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = linearizeDescriptorSet;
    w.dstBinding = 0;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.descriptorCount = 1;
    w.pImageInfo = &di;
    app->updateDescriptorSet({ w });

    app->runSingleTimeCommands([&](VkCommandBuffer cmd){
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
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, linearizePipelineLayout, 0, 1, &linearizeDescriptorSet, 0, nullptr);
        VkViewport vp{}; vp.x = 0.0f; vp.y = 0.0f; vp.width = static_cast<float>(width); vp.height = static_cast<float>(height); vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{}; sc.offset = {0,0}; sc.extent = { width, height };
        vkCmdSetScissor(cmd, 0, 1, &sc);
        float pc[3] = { zNear, zFar, mode };
        vkCmdPushConstants(cmd, linearizePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), pc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
    });

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
        if (app && app->hasPendingCommandBuffers()) {
            VkDescriptorSet tmp = ds;
            // Prefer deferring until the current in-flight fence signals so the
            // descriptor set isn't freed while still in use by GPU work.
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = app->getCurrentFrame();
            if (fi < app->inFlightFences.size()) f = app->inFlightFences[fi];
            app->deferDestroyUntilFence(f, [tmp](){ ImGui_ImplVulkan_RemoveTexture(tmp); });
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
    removeOwnedDesc(backFaceDepthDescriptor, backFaceDepthDescriptorOwned);
    removeOwnedDesc(waterDepthLinearDescriptor, waterDepthLinearDescriptorOwned);
}

void RenderTargetsWidget::setFrameInfo(uint32_t frameIndex, int width, int height) {
    currentFrame = static_cast<int>(frameIndex);
    // If the size changed, destroy linear preview targets so they are recreated
    // at the new size (avoids renderArea vs framebuffer extent mismatches).
    if (cachedWidth != width || cachedHeight != height) {
        destroyLinearTargets();
    }
    cachedWidth = width;
    cachedHeight = height;
}

void RenderTargetsWidget::destroyLinearTargets() {
    VulkanApp* a = app;
    if (!a) return;

    // Remove ImGui descriptors owned by this widget and destroy associated
    // images, views, framebuffers and pipeline. Defer if GPU work is pending.
    auto removeDescIfOwned = [&](VkDescriptorSet &ds, bool &owned) {
        if (ds == VK_NULL_HANDLE) return;
        if (!owned) { ds = VK_NULL_HANDLE; return; }
        if (a->hasPendingCommandBuffers()) {
            VkDescriptorSet tmp = ds;
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp]() { ImGui_ImplVulkan_RemoveTexture(tmp); });
        } else {
            ImGui_ImplVulkan_RemoveTexture(ds);
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
                    a->resources.removeImageView(tmp_iv);
                    vkDestroyImageView(device, tmp_iv, nullptr);
                }
                if (tmp_img != VK_NULL_HANDLE) {
                    a->resources.removeImage(tmp_img);
                    vkDestroyImage(device, tmp_img, nullptr);
                }
                if (tmp_mem != VK_NULL_HANDLE) {
                    a->resources.removeDeviceMemory(tmp_mem);
                    vkFreeMemory(device, tmp_mem, nullptr);
                }
            });
        } else {
            if (tmp_iv != VK_NULL_HANDLE) {
                a->resources.removeImageView(tmp_iv);
                vkDestroyImageView(device, tmp_iv, nullptr);
            }
            if (tmp_img != VK_NULL_HANDLE) {
                a->resources.removeImage(tmp_img);
                vkDestroyImage(device, tmp_img, nullptr);
            }
            if (tmp_mem != VK_NULL_HANDLE) {
                a->resources.removeDeviceMemory(tmp_mem);
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
                a->resources.removeFramebuffer(tmp);
                vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
            });
        } else {
            a->resources.removeFramebuffer(tmp);
            vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
        }
        fb = VK_NULL_HANDLE;
    };

    destroyFramebuffer(linearSceneFramebuffer);
    destroyFramebuffer(linearBackFaceFramebuffer);
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) destroyFramebuffer(linearShadowFramebuffer[i]);

    destroyImageAndMemory(linearSceneDepthView, linearSceneDepthImage, linearSceneDepthMemory);
    destroyImageAndMemory(linearBackFaceDepthView, linearBackFaceDepthImage, linearBackFaceDepthMemory);
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        destroyImageAndMemory(linearShadowDepthView[i], linearShadowDepthImage[i], linearShadowDepthMemory[i]);
    }

    // Destroy pipeline, layout, descriptor set/layout and render pass
    if (linearizePipeline != VK_NULL_HANDLE) {
        VkPipeline tmp = linearizePipeline;
        if (a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                a->resources.removePipeline(tmp);
                vkDestroyPipeline(a->getDevice(), tmp, nullptr);
            });
        } else {
            a->resources.removePipeline(tmp);
            vkDestroyPipeline(a->getDevice(), tmp, nullptr);
        }
        linearizePipeline = VK_NULL_HANDLE;
    }

    if (linearizePipelineLayout != VK_NULL_HANDLE) {
        VkPipelineLayout tmp = linearizePipelineLayout;
        if (a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                a->resources.removePipelineLayout(tmp);
                vkDestroyPipelineLayout(a->getDevice(), tmp, nullptr);
            });
        } else {
            a->resources.removePipelineLayout(tmp);
            vkDestroyPipelineLayout(a->getDevice(), tmp, nullptr);
        }
        linearizePipelineLayout = VK_NULL_HANDLE;
    }

    if (linearizeDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorSet tmp = linearizeDescriptorSet;
        if (a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                a->resources.removeDescriptorSet(tmp);
            });
        } else {
            a->resources.removeDescriptorSet(tmp);
        }
        linearizeDescriptorSet = VK_NULL_HANDLE;
    }

    if (linearizeDescriptorSetLayout != VK_NULL_HANDLE) {
        VkDescriptorSetLayout tmp = linearizeDescriptorSetLayout;
        if (a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                a->resources.removeDescriptorSetLayout(tmp);
                vkDestroyDescriptorSetLayout(a->getDevice(), tmp, nullptr);
            });
        } else {
            a->resources.removeDescriptorSetLayout(tmp);
            vkDestroyDescriptorSetLayout(a->getDevice(), tmp, nullptr);
        }
        linearizeDescriptorSetLayout = VK_NULL_HANDLE;
    }

    if (linearizeRenderPass != VK_NULL_HANDLE) {
        VkRenderPass tmp = linearizeRenderPass;
        if (a->hasPendingCommandBuffers()) {
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = a->getCurrentFrame();
            if (fi < a->inFlightFences.size()) f = a->inFlightFences[fi];
            a->deferDestroyUntilFence(f, [tmp, a]() {
                a->resources.removeRenderPass(tmp);
                vkDestroyRenderPass(a->getDevice(), tmp, nullptr);
            });
        } else {
            a->resources.removeRenderPass(tmp);
            vkDestroyRenderPass(a->getDevice(), tmp, nullptr);
        }
        linearizeRenderPass = VK_NULL_HANDLE;
    }

    // Reset tracked sizes
    linearSceneWidth = 0;
    linearSceneHeight = 0;
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) linearShadowSize[i] = 0;
}

void RenderTargetsWidget::cleanup() {
    auto removeOwnedDesc = [&](VkDescriptorSet &ds, bool &owned) {
        if (ds == VK_NULL_HANDLE) return;
        if (!owned) { ds = VK_NULL_HANDLE; return; }
        if (app && app->hasPendingCommandBuffers()) {
            VkDescriptorSet tmp = ds;
            VkFence f = VK_NULL_HANDLE;
            uint32_t fi = app->getCurrentFrame();
            if (fi < app->inFlightFences.size()) f = app->inFlightFences[fi];
            app->deferDestroyUntilFence(f, [tmp](){ ImGui_ImplVulkan_RemoveTexture(tmp); });
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
                    a->resources.removeImageView(tmp_iv);
                    vkDestroyImageView(device, tmp_iv, nullptr);
                }
                if (tmp_img != VK_NULL_HANDLE) {
                    a->resources.removeImage(tmp_img);
                    vkDestroyImage(device, tmp_img, nullptr);
                }
                if (tmp_mem != VK_NULL_HANDLE) {
                    a->resources.removeDeviceMemory(tmp_mem);
                    vkFreeMemory(device, tmp_mem, nullptr);
                }
            });
        } else {
            if (tmp_iv != VK_NULL_HANDLE) {
                a->resources.removeImageView(tmp_iv);
                vkDestroyImageView(device, tmp_iv, nullptr);
            }
            if (tmp_img != VK_NULL_HANDLE) {
                a->resources.removeImage(tmp_img);
                vkDestroyImage(device, tmp_img, nullptr);
            }
            if (tmp_mem != VK_NULL_HANDLE) {
                a->resources.removeDeviceMemory(tmp_mem);
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
                a->resources.removeBuffer(tmpBuf);
                vkDestroyBuffer(device, tmpBuf, nullptr);
                a->resources.removeDeviceMemory(tmpMem);
                vkFreeMemory(device, tmpMem, nullptr);
            });
        } else {
            a->resources.removeBuffer(tmpBuf);
            vkDestroyBuffer(device, tmpBuf, nullptr);
            a->resources.removeDeviceMemory(tmpMem);
            vkFreeMemory(device, tmpMem, nullptr);
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
                (a->resources.*removeFn)(tmp);
                destroyFn(device, tmp);
            });
        } else {
            (a->resources.*removeFn)(tmp);
            destroyFn(a->getDevice(), tmp);
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
                a->resources.removeFramebuffer(tmp);
                vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
            });
        } else {
            a->resources.removeFramebuffer(tmp);
            vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
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
                a->resources.removeFramebuffer(tmp);
                vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
            });
        } else {
            a->resources.removeFramebuffer(tmp);
            vkDestroyFramebuffer(a->getDevice(), tmp, nullptr);
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
                a->resources.removePipeline(tmp);
                vkDestroyPipeline(a->getDevice(), tmp, nullptr);
            });
        } else {
            a->resources.removePipeline(tmp);
            vkDestroyPipeline(a->getDevice(), tmp, nullptr);
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
                a->resources.removePipelineLayout(tmp);
                vkDestroyPipelineLayout(a->getDevice(), tmp, nullptr);
            });
        } else {
            a->resources.removePipelineLayout(tmp);
            vkDestroyPipelineLayout(a->getDevice(), tmp, nullptr);
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
                a->resources.removeDescriptorSetLayout(tmp);
                vkDestroyDescriptorSetLayout(a->getDevice(), tmp, nullptr);
            });
        } else {
            a->resources.removeDescriptorSetLayout(tmp);
            vkDestroyDescriptorSetLayout(a->getDevice(), tmp, nullptr);
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
                a->resources.removeRenderPass(tmp);
                vkDestroyRenderPass(a->getDevice(), tmp, nullptr);
            });
        } else {
            a->resources.removeRenderPass(tmp);
            vkDestroyRenderPass(a->getDevice(), tmp, nullptr);
        }
        linearizeRenderPass = VK_NULL_HANDLE;
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

    VkSampler sampler = (sceneRenderer && sceneRenderer->waterRenderer) ? sceneRenderer->waterRenderer->getLinearSampler() : VK_NULL_HANDLE;
    if (sampler == VK_NULL_HANDLE) return;

    // Sky equirectangular texture
    if (skyRenderer) {
        VkImageView skyView = skyRenderer->getSkyView(frameIndex);
        if (skyView != VK_NULL_HANDLE && skyDescriptor == VK_NULL_HANDLE) {
            skyDescriptor = ImGui_ImplVulkan_AddTexture(
                sampler, skyView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            skyDescriptorOwned = true;
        }
    }

    // Solid color
    if (solidRenderer) {
        VkImageView solidView = solidRenderer->getColorView(frameIndex);
        if (solidView != VK_NULL_HANDLE && solidColorDescriptor == VK_NULL_HANDLE) {
            solidColorDescriptor = ImGui_ImplVulkan_AddTexture(
                sampler, solidView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            solidColorDescriptorOwned = true;
        }
    }

    // Solid depth (D32_SFLOAT – use shadow sampler which has compareEnable=VK_FALSE)
    if (solidRenderer && shadowMapper) {
        VkImageView depthView = solidRenderer->getDepthView(frameIndex);
        if (depthView != VK_NULL_HANDLE && solidDepthDescriptor == VK_NULL_HANDLE) {
            solidDepthDescriptor = ImGui_ImplVulkan_AddTexture(
                shadowMapper->getShadowMapSampler(), depthView,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            solidDepthDescriptorOwned = true;
        }
    }

    VkSampler solid360Sampler = sampler;
    if (sceneRenderer && sceneRenderer->solid360Renderer) {
        VkSampler s = sceneRenderer->solid360Renderer->getSolid360Sampler();
        if (s != VK_NULL_HANDLE) solid360Sampler = s;
    }

    VkImageView cube360EquirectView = (sceneRenderer && sceneRenderer->solid360Renderer) ? sceneRenderer->solid360Renderer->getSolid360View() : VK_NULL_HANDLE;
    if (cube360EquirectView != VK_NULL_HANDLE && selectedPreview == PreviewTarget::Solid360Equirect) {
        cube360EquirectRenderer.render(app, solid360Sampler, cube360EquirectView);
        if (cube360EquirectDescriptor == VK_NULL_HANDLE) {
            VkImageView equirectView = cube360EquirectRenderer.getEquirectView();
            if (equirectView != VK_NULL_HANDLE) {
                cube360EquirectDescriptor = ImGui_ImplVulkan_AddTexture(
                    solid360Sampler, equirectView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                cube360EquirectDescriptorOwned = true;
            }
        }
    }

    // Per-face cube descriptors for detailed orientation inspection
    for (uint32_t f = 0; f < 6; ++f) {
        VkImageView faceView = (sceneRenderer && sceneRenderer->solid360Renderer) ? sceneRenderer->solid360Renderer->getCube360FaceView(f) : VK_NULL_HANDLE;
        if (faceView != VK_NULL_HANDLE && cube360FaceDescriptor[f] == VK_NULL_HANDLE) {
            cube360FaceDescriptor[f] = ImGui_ImplVulkan_AddTexture(solid360Sampler, faceView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            cube360FaceDescriptorOwned[f] = true;
        }
    }

    // Water color (first attachment of water geometry pass — R32G32B32A32_SFLOAT worldPos)
    VkImageView waterView = (sceneRenderer && sceneRenderer->waterRenderer) ? sceneRenderer->waterRenderer->getWaterDepthView(frameIndex) : VK_NULL_HANDLE;
    if (waterView != VK_NULL_HANDLE && waterColorDescriptor == VK_NULL_HANDLE) {
        waterColorDescriptor = ImGui_ImplVulkan_AddTexture(
            sampler, waterView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        waterColorDescriptorOwned = true;
    }

    // (Water normals preview removed)

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

        // Create target images + views (RGBA8) if missing
        if (linearSceneDepthImage == VK_NULL_HANDLE) {
            app->createImage(static_cast<uint32_t>(cachedWidth), static_cast<uint32_t>(cachedHeight), VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_TILING_OPTIMAL, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, linearSceneDepthImage, linearSceneDepthMemory);
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
            app->createImage(static_cast<uint32_t>(cachedWidth), static_cast<uint32_t>(cachedHeight), VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_TILING_OPTIMAL, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, linearBackFaceDepthImage, linearBackFaceDepthMemory);
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

        // Create framebuffers for the two linearized targets
        if (linearSceneFramebuffer == VK_NULL_HANDLE && linearSceneDepthView != VK_NULL_HANDLE && linearizeRenderPass != VK_NULL_HANDLE) {
            VkImageView atts[] = { linearSceneDepthView };
            VkFramebufferCreateInfo fb{};
            fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb.renderPass = linearizeRenderPass;
            fb.attachmentCount = 1;
            fb.pAttachments = atts;
            fb.width = static_cast<uint32_t>(cachedWidth);
            fb.height = static_cast<uint32_t>(cachedHeight);
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
            fb.width = static_cast<uint32_t>(cachedWidth);
            fb.height = static_cast<uint32_t>(cachedHeight);
            fb.layers = 1;
            if (vkCreateFramebuffer(device, &fb, nullptr, &linearBackFaceFramebuffer) == VK_SUCCESS) {
                app->resources.addFramebuffer(linearBackFaceFramebuffer, "RenderTargetsWidget: linearBackFaceFramebuffer");
            } else linearBackFaceFramebuffer = VK_NULL_HANDLE;
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

        // Prepare a sampler for sampling depth textures (prefer shadow sampler)
        VkSampler depthSampler = VK_NULL_HANDLE;
        if (shadowMapper) depthSampler = shadowMapper->getShadowMapSampler();
        if (depthSampler == VK_NULL_HANDLE) depthSampler = sampler;

        // Run the pass for scene depth (use perspective linearization)
        if (solidRenderer && linearizePipeline != VK_NULL_HANDLE && linearSceneFramebuffer != VK_NULL_HANDLE) {
            VkImageView src = solidRenderer->getDepthView(frameIndex);
            fprintf(stderr, "[RenderTargetsWidget] Scene linearize check: pipeline=%p descSet=%p fb=%p view=%p src=%p\n",
                    (void*)linearizePipeline, (void*)linearizeDescriptorSet, (void*)linearSceneFramebuffer, (void*)linearSceneDepthView, (void*)src);
            if (src != VK_NULL_HANDLE) {
                float nearP = 0.1f, farP = 1000.0f;
                if (settings) { nearP = settings->nearPlane; farP = settings->farPlane; }
                runLinearizePass(app, src, depthSampler, sampler, linearSceneDepthView, linearSceneFramebuffer,
                                 linearSceneDepthDescriptor, linearSceneDepthDescriptorOwned,
                                 static_cast<uint32_t>(cachedWidth), static_cast<uint32_t>(cachedHeight), nearP, farP, 0.0f);
                fprintf(stderr, "[RenderTargetsWidget] Scene linearize: pass completed\n");
            }
        }

        // Back-face depth pass
        // Back-face depth pass (use perspective linearization)
        if (sceneRenderer && sceneRenderer->waterRenderer && linearizePipeline != VK_NULL_HANDLE && linearBackFaceFramebuffer != VK_NULL_HANDLE) {
            VkImageView src = (sceneRenderer && sceneRenderer->backFaceRenderer) ? sceneRenderer->backFaceRenderer->getBackFaceDepthView(frameIndex) : VK_NULL_HANDLE;
            fprintf(stderr, "[RenderTargetsWidget] Backface linearize check: pipeline=%p descSet=%p fb=%p view=%p src=%p\n",
                    (void*)linearizePipeline, (void*)linearizeDescriptorSet, (void*)linearBackFaceFramebuffer, (void*)linearBackFaceDepthView, (void*)src);
            if (src != VK_NULL_HANDLE) {
                float nearP = 0.1f, farP = 1000.0f;
                if (settings) { nearP = settings->nearPlane; farP = settings->farPlane; }
                runLinearizePass(app, src, depthSampler, sampler, linearBackFaceDepthView, linearBackFaceFramebuffer,
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
                runLinearizePass(app, src, depthSampler, sampler, linearSceneDepthView, linearSceneFramebuffer,
                                 linearSceneDepthDescriptor, linearSceneDepthDescriptorOwned,
                                 static_cast<uint32_t>(cachedWidth), static_cast<uint32_t>(cachedHeight), nearP, farP, 0.0f);
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
                if (linearShadowDepthImage[c] == VK_NULL_HANDLE) {
                    app->createImage(shadowSize, shadowSize, VK_FORMAT_R8G8B8A8_UNORM,
                                     VK_IMAGE_TILING_OPTIMAL, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, linearShadowDepthImage[c], linearShadowDepthMemory[c]);
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

                VkImageView src = shadowMapper->getShadowMapView(c);
                if (src != VK_NULL_HANDLE && linearShadowFramebuffer[c] != VK_NULL_HANDLE) {
                    float nearP = 0.0f, farP = 1.0f;
                    runLinearizePass(app, src, depthSampler, sampler, linearShadowDepthView[c], linearShadowFramebuffer[c],
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
                shadowMapper->getShadowMapSampler(), bfView,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            backFaceDepthDescriptorOwned = true;
        }
    }

    // Alias linear-depth previews to renderer-provided depth image views so
    // the widget displays depth entirely via GPU sampling (no CPU readback).
    // Scene linear depth: if we didn't create a GPU-linearized image above,
    // alias to the solid renderer depth view so we still show something.
    if (linearSceneDepthDescriptor == VK_NULL_HANDLE && solidRenderer) {
        VkImageView sceneDepthView = solidRenderer->getDepthView(frameIndex);
        if (sceneDepthView != VK_NULL_HANDLE) {
            VkSampler depthSampler = shadowMapper ? shadowMapper->getShadowMapSampler() : sampler;
            linearSceneDepthDescriptor = ImGui_ImplVulkan_AddTexture(depthSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            linearSceneDepthDescriptorOwned = true;
        }
    }
    // Back-face depth (water): alias to water back-face depth view
    VkImageView bfView2 = (sceneRenderer && sceneRenderer->backFaceRenderer) ? sceneRenderer->backFaceRenderer->getBackFaceDepthView(frameIndex) : VK_NULL_HANDLE;
    if (linearBackFaceDepthDescriptor == VK_NULL_HANDLE && bfView2 != VK_NULL_HANDLE) {
        VkSampler depthSampler = shadowMapper ? shadowMapper->getShadowMapSampler() : sampler;
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

    updateDescriptors(currentFrame);

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
    const char* previewItems[] = {
        "Sky",
        "Solid360Cube",
        "Solid360Equirect",
        "SolidColor",
        "SolidDepth",
        "WaterColor",
        "WaterDepth",
        "BackFaceColor",
        "BackFaceDepth",
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
    ImGui::SameLine();
    ImGui::Checkbox("Show All Cascades", &showAllCascades);
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

    // Debug: print current selection to stderr for quick diagnostics
    fprintf(stderr, "RenderTargetsWidget: selectedPreview=%d selectedShadowCascade=%d showAllCascades=%d\n", static_cast<int>(selectedPreview), selectedShadowCascade, showAllCascades);

    float aspect = 1.0f;
    if (cachedWidth > 0 && cachedHeight > 0) aspect = static_cast<float>(cachedHeight) / static_cast<float>(cachedWidth);
    ImVec2 previewSize(PREVIEW_WIDTH, PREVIEW_WIDTH * aspect);

    // Render only the selected preview using a single preview descriptor
    VkDescriptorSet ds = previewDescriptor;
    ImVec2 imgSize = previewSize;
    bool available = (ds != VK_NULL_HANDLE);

    if (selectedPreview == PreviewTarget::Solid360Cube) {
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
