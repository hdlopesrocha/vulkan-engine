#include "RenderTargetsWidget.hpp"
#include "Settings.hpp"
#include "../vulkan/VulkanApp.hpp"
#include "../vulkan/WaterRenderer.hpp"
#include "../vulkan/SolidRenderer.hpp"
#include "../vulkan/SkyRenderer.hpp"
#include "../vulkan/ShadowRenderer.hpp"
#include "../vulkan/ShadowParams.hpp"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <string>
#include <cstdio>

RenderTargetsWidget::RenderTargetsWidget(VulkanApp* app, WaterRenderer* water, SolidRenderer* solid, SkyRenderer* sky,
                                                                                 ShadowRenderer* shadow, ShadowParams* shadowParams, Settings* settings)
        : Widget("Render Targets"), app(app), waterRenderer(water), solidRenderer(solid), skyRenderer(sky),
            shadowMapper(shadow), shadowParams(shadowParams), settings(settings) {
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
    removeOwnedDesc(waterColorDescriptor, waterColorDescriptorOwned);
    removeOwnedDesc(solidDepthDescriptor, solidDepthDescriptorOwned);
    removeOwnedDesc(solid360Descriptor, solid360DescriptorOwned);
    removeOwnedDesc(backFaceDepthDescriptor, backFaceDepthDescriptorOwned);
    removeOwnedDesc(waterDepthLinearDescriptor, waterDepthLinearDescriptorOwned);
}

void RenderTargetsWidget::setFrameInfo(uint32_t frameIndex, int width, int height) {
    currentFrame = static_cast<int>(frameIndex);
    cachedWidth = width;
    cachedHeight = height;
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
    removeOwnedDesc(backFaceDepthDescriptor, backFaceDepthDescriptorOwned);
    removeOwnedDesc(waterDepthLinearDescriptor, waterDepthLinearDescriptorOwned);
    removeOwnedDesc(linearSceneDepthDescriptor, linearSceneDepthDescriptorOwned);
    removeOwnedDesc(linearBackFaceDepthDescriptor, linearBackFaceDepthDescriptorOwned);
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
}

void RenderTargetsWidget::updateDescriptors(uint32_t frameIndex) {
    if (!waterRenderer) return;

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

    // Cleanup previous ephemeral descriptors before recreating (only descriptors,
    // not device-local images or persistent staging buffers). Calling the full
    // `cleanup()` here destroyed images each frame which caused repeated
    // allocations. Remove only ImGui descriptor sets so underlying Vulkan
    // images/buffers persist across frames and are recreated only on resize.
    {
        auto removeDesc = [&](VkDescriptorSet &ds, bool &owned) {
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
        removeDesc(skyDescriptor, skyDescriptorOwned);
        removeDesc(solidColorDescriptor, solidColorDescriptorOwned);
        removeDesc(waterColorDescriptor, waterColorDescriptorOwned);
        removeDesc(solidDepthDescriptor, solidDepthDescriptorOwned);
        removeDesc(solid360Descriptor, solid360DescriptorOwned);
        removeDesc(backFaceDepthDescriptor, backFaceDepthDescriptorOwned);
        removeDesc(waterDepthLinearDescriptor, waterDepthLinearDescriptorOwned);
        removeDesc(linearSceneDepthDescriptor, linearSceneDepthDescriptorOwned);
        removeDesc(linearBackFaceDepthDescriptor, linearBackFaceDepthDescriptorOwned);
        for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
            removeDesc(linearShadowDepthDescriptor[i], linearShadowDepthDescriptorOwned[i]);
        }
    }

    VkSampler sampler = waterRenderer->getLinearSampler();
    if (sampler == VK_NULL_HANDLE) return;

    // Sky equirectangular texture
    if (skyRenderer) {
        VkImageView skyView = skyRenderer->getSkyView(frameIndex);
        if (skyView != VK_NULL_HANDLE) {
            skyDescriptor = ImGui_ImplVulkan_AddTexture(
                sampler, skyView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            skyDescriptorOwned = true;
        }
    }

    // Solid color
    if (solidRenderer) {
        VkImageView solidView = solidRenderer->getColorView(frameIndex);
        if (solidView != VK_NULL_HANDLE) {
            solidColorDescriptor = ImGui_ImplVulkan_AddTexture(
                sampler, solidView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            solidColorDescriptorOwned = true;
        }
    }

    // Solid depth (D32_SFLOAT – use shadow sampler which has compareEnable=VK_FALSE)
    if (solidRenderer && shadowMapper) {
        VkImageView depthView = solidRenderer->getDepthView(frameIndex);
        if (depthView != VK_NULL_HANDLE) {
            solidDepthDescriptor = ImGui_ImplVulkan_AddTexture(
                shadowMapper->getShadowMapSampler(), depthView,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            solidDepthDescriptorOwned = true;
        }
    }

    // Solid 360° equirectangular reflection
    VkImageView solid360View = waterRenderer->getSolid360View();
    if (solid360View != VK_NULL_HANDLE) {
        solid360Descriptor = ImGui_ImplVulkan_AddTexture(
            sampler, solid360View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        solid360DescriptorOwned = true;
    }

    // Water color (first attachment of water geometry pass — R32G32B32A32_SFLOAT worldPos)
    VkImageView waterView = waterRenderer->getWaterDepthView();
    if (waterView != VK_NULL_HANDLE) {
        waterColorDescriptor = ImGui_ImplVulkan_AddTexture(
            sampler, waterView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        waterColorDescriptorOwned = true;
    }

    // (Water normals preview removed)

    // Water linear depth (alpha channel of the world-pos attachment swizzled into RGB)
    VkImageView waterDepthLinearView = waterRenderer->getWaterDepthAlphaView();
    if (waterDepthLinearView != VK_NULL_HANDLE) {
        waterDepthLinearDescriptor = ImGui_ImplVulkan_AddTexture(
            sampler, waterDepthLinearView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        waterDepthLinearDescriptorOwned = true;
    }

    // CPU readback removed: previews use renderer-provided GPU image views.
    // Water back-face depth (volume thickness pre-pass — D32_SFLOAT)
    if (shadowMapper) {
        VkImageView bfView = waterRenderer->getBackFaceDepthView();
        if (bfView != VK_NULL_HANDLE) {
            backFaceDepthDescriptor = ImGui_ImplVulkan_AddTexture(
                shadowMapper->getShadowMapSampler(), bfView,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            backFaceDepthDescriptorOwned = true;
        }
    }

    // Alias linear-depth previews to renderer-provided depth image views so
    // the widget displays depth entirely via GPU sampling (no CPU readback).
    // Scene linear depth: alias to solid renderer depth view
    if (solidRenderer) {
        VkImageView sceneDepthView = solidRenderer->getDepthView(frameIndex);
        if (sceneDepthView != VK_NULL_HANDLE) {
            VkSampler depthSampler = shadowMapper ? shadowMapper->getShadowMapSampler() : sampler;
            linearSceneDepthDescriptor = ImGui_ImplVulkan_AddTexture(depthSampler, sceneDepthView, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
            linearSceneDepthDescriptorOwned = true;
        }
    }
    // Back-face depth (water): alias to water back-face depth view
    VkImageView bfView2 = waterRenderer->getBackFaceDepthView();
    if (bfView2 != VK_NULL_HANDLE) {
        VkSampler depthSampler = shadowMapper ? shadowMapper->getShadowMapSampler() : sampler;
        linearBackFaceDepthDescriptor = ImGui_ImplVulkan_AddTexture(depthSampler, bfView2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        linearBackFaceDepthDescriptorOwned = true;
    }

    // Choose a single preview descriptor according to the current selection.
    previewDescriptor = VK_NULL_HANDLE;
    switch (selectedPreview) {
        case PreviewTarget::Sky: previewDescriptor = skyDescriptor; break;
        case PreviewTarget::Solid360: previewDescriptor = solid360Descriptor; break;
        case PreviewTarget::SolidColor: previewDescriptor = solidColorDescriptor; break;
        case PreviewTarget::SolidDepth: previewDescriptor = solidDepthDescriptor; break;
        case PreviewTarget::WaterWorldPos: previewDescriptor = waterColorDescriptor; break;
        case PreviewTarget::WaterLinearDepth: previewDescriptor = waterDepthLinearDescriptor; break;
        case PreviewTarget::WaterBackFace: previewDescriptor = backFaceDepthDescriptor; break;
        case PreviewTarget::LinearSceneDepth: previewDescriptor = linearSceneDepthDescriptor; break;
        case PreviewTarget::LinearBackFaceDepth: previewDescriptor = linearBackFaceDepthDescriptor; break;
        case PreviewTarget::ShadowCascade:
            if (shadowViewMode == RenderTargetsWidget::ShadowViewMode::Linearized) {
                if (linearShadowDepthDescriptor[selectedShadowCascade] != VK_NULL_HANDLE)
                    previewDescriptor = linearShadowDepthDescriptor[selectedShadowCascade];
                else if (shadowMapper) previewDescriptor = shadowMapper->getImGuiDescriptorSet(selectedShadowCascade);
            } else {
                if (shadowMapper) previewDescriptor = shadowMapper->getImGuiDescriptorSet(selectedShadowCascade);
            }
            break;
        default: previewDescriptor = VK_NULL_HANDLE; break;
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

    if (!waterRenderer || !solidRenderer) {
        ImGui::TextUnformatted("Renderers not available.");
        ImGui::End();
        return;
    }

    updateDescriptors(currentFrame);

    ImGui::SliderFloat("Preview Scale", &previewScale, 0.1f, 1.0f);
    ImGui::Separator();

    // Preview selector: show only one image at a time
    ImGui::Text("Preview Selector");
    ImGui::SameLine();
    if (ImGui::BeginPopupContextItem("preview_selector")) {
        ImGui::TextUnformatted("Choose preview");
        ImGui::EndPopup();
    }
    // Radio buttons laid out in two columns
    ImGui::Columns(2, "preview_cols", false);
    auto rb = [&](const char* label, RenderTargetsWidget::PreviewTarget v){
        bool active = (selectedPreview == v);
        if (ImGui::RadioButton(label, active)) selectedPreview = v;
    };
    rb("Sky", PreviewTarget::Sky); ImGui::NextColumn();
    rb("Solid 360", PreviewTarget::Solid360); ImGui::NextColumn();
    rb("Solid Color", PreviewTarget::SolidColor); ImGui::NextColumn();
    rb("Solid Depth", PreviewTarget::SolidDepth); ImGui::NextColumn();
    rb("Water WorldPos", PreviewTarget::WaterWorldPos); ImGui::NextColumn();
    rb("Water Linear", PreviewTarget::WaterLinearDepth); ImGui::NextColumn();
    rb("Water BackFace", PreviewTarget::WaterBackFace); ImGui::NextColumn();
    rb("Linear Scene Depth", PreviewTarget::LinearSceneDepth); ImGui::NextColumn();
    rb("Linear BackFace", PreviewTarget::LinearBackFaceDepth); ImGui::NextColumn();
    rb("Shadow Cascade", PreviewTarget::ShadowCascade); ImGui::NextColumn();
    ImGui::Columns(1);
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

    ImVec2 previewSize(static_cast<float>(cachedWidth) * previewScale,
                       static_cast<float>(cachedHeight) * previewScale);

    // Render only the selected preview using a single preview descriptor
    VkDescriptorSet ds = previewDescriptor;
    ImVec2 imgSize = previewSize;
    const char* label = "Preview";
    bool available = (ds != VK_NULL_HANDLE);
    switch (selectedPreview) {
        case PreviewTarget::Sky: label = "Sky (Equirectangular)"; imgSize = ImVec2(512.0f * previewScale * 2.0f, 512.0f * previewScale); break;
        case PreviewTarget::Solid360: label = "Solid 360 (Reflection)"; imgSize = ImVec2(512.0f * previewScale * 2.0f, 512.0f * previewScale); break;
        case PreviewTarget::SolidColor: label = "Solid (Scene Color)"; break;
        case PreviewTarget::SolidDepth: label = "Solid (Depth Buffer)"; break;
        case PreviewTarget::WaterWorldPos: label = "Water (World Pos)"; break;
        case PreviewTarget::WaterLinearDepth: label = "Water (Linear Depth)"; break;
        case PreviewTarget::WaterBackFace: label = "Water (Back-Face Depth)"; break;
        case PreviewTarget::LinearSceneDepth: label = "Scene (Linearized Depth)"; break;
        case PreviewTarget::LinearBackFaceDepth: label = "BackFace (Linearized Depth)"; break;
        case PreviewTarget::ShadowCascade:
            label = "Shadow Cascade";
            if (shadowMapper) imgSize = ImVec2(static_cast<float>(shadowMapper->getShadowMapSize()) * previewScale, static_cast<float>(shadowMapper->getShadowMapSize()) * previewScale);
            break;
        default: label = "Unknown preview"; break;
    }
    ImGui::TextUnformatted(label);
    if (available) ImGui::Image((ImTextureID)ds, imgSize); else ImGui::Text("Preview unavailable");
    ImGui::Separator();

    // Optionally show all cascades (in selected shadow view mode)
    // Only show the full cascade grid when the shadow cascade preview is selected.
    if (selectedPreview == PreviewTarget::ShadowCascade && showAllCascades && shadowMapper) {
        float shadowSize = static_cast<float>(shadowMapper->getShadowMapSize()) * previewScale;
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
