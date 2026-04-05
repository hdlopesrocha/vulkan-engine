#include "RenderTargetsWidget.hpp"
#include "../vulkan/VulkanApp.hpp"
#include "../vulkan/WaterRenderer.hpp"
#include "../vulkan/SolidRenderer.hpp"
#include "../vulkan/SkyRenderer.hpp"
#include "../vulkan/ShadowRenderer.hpp"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>

RenderTargetsWidget::RenderTargetsWidget(WaterRenderer* water, SolidRenderer* solid, SkyRenderer* sky,
                                         ShadowRenderer* shadow)
    : Widget("Render Targets"), waterRenderer(water), solidRenderer(solid), skyRenderer(sky),
      shadowMapper(shadow) {
}

RenderTargetsWidget::~RenderTargetsWidget() {
    cleanup();
}

void RenderTargetsWidget::setFrameInfo(uint32_t frameIndex, int width, int height) {
    currentFrame = static_cast<int>(frameIndex);
    cachedWidth = width;
    cachedHeight = height;
}

void RenderTargetsWidget::cleanup() {
    if (skyDescriptor != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(skyDescriptor);
        skyDescriptor = VK_NULL_HANDLE;
    }
    if (solidColorDescriptor != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(solidColorDescriptor);
        solidColorDescriptor = VK_NULL_HANDLE;
    }
    if (waterColorDescriptor != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(waterColorDescriptor);
        waterColorDescriptor = VK_NULL_HANDLE;
    }
    if (solidDepthDescriptor != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(solidDepthDescriptor);
        solidDepthDescriptor = VK_NULL_HANDLE;
    }
    if (solid360Descriptor != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(solid360Descriptor);
        solid360Descriptor = VK_NULL_HANDLE;
    }
}

void RenderTargetsWidget::updateDescriptors(uint32_t frameIndex) {
    if (!waterRenderer) return;

    cleanup();

    VkSampler sampler = waterRenderer->getLinearSampler();
    if (sampler == VK_NULL_HANDLE) return;

    // Sky equirectangular texture
    if (skyRenderer) {
        VkImageView skyView = skyRenderer->getSkyView(frameIndex);
        if (skyView != VK_NULL_HANDLE) {
            skyDescriptor = ImGui_ImplVulkan_AddTexture(
                sampler, skyView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }

    // Solid color
    if (solidRenderer) {
        VkImageView solidView = solidRenderer->getColorView(frameIndex);
        if (solidView != VK_NULL_HANDLE) {
            solidColorDescriptor = ImGui_ImplVulkan_AddTexture(
                sampler, solidView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }

    // Solid depth (D32_SFLOAT – use shadow sampler which has compareEnable=VK_FALSE)
    if (solidRenderer && shadowMapper) {
        VkImageView depthView = solidRenderer->getDepthView(frameIndex);
        if (depthView != VK_NULL_HANDLE) {
            solidDepthDescriptor = ImGui_ImplVulkan_AddTexture(
                shadowMapper->getShadowMapSampler(), depthView,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }

    // Solid 360° equirectangular reflection
    VkImageView solid360View = waterRenderer->getSolid360View();
    if (solid360View != VK_NULL_HANDLE) {
        solid360Descriptor = ImGui_ImplVulkan_AddTexture(
            sampler, solid360View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    // Water color (first attachment of water geometry pass)
    VkImageView waterView = waterRenderer->getWaterDepthView();
    if (waterView != VK_NULL_HANDLE) {
        waterColorDescriptor = ImGui_ImplVulkan_AddTexture(
            sampler, waterView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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

    // Sky equirect: 2:1 aspect ratio
    if (skyDescriptor != VK_NULL_HANDLE) {
        float skyW = 512.0f * previewScale * 2.0f;
        float skyH = 512.0f * previewScale;
        ImGui::Text("Sky (Equirectangular)");
        ImGui::Image((ImTextureID)skyDescriptor, ImVec2(skyW, skyH));
        ImGui::Separator();
    }

    // Solid 360° reflection (equirect: 2:1 aspect)
    if (solid360Descriptor != VK_NULL_HANDLE) {
        float s360W = 512.0f * previewScale * 2.0f;
        float s360H = 512.0f * previewScale;
        ImGui::Text("Solid 360 (Reflection)");
        ImGui::Image((ImTextureID)solid360Descriptor, ImVec2(s360W, s360H));
        ImGui::Separator();
    }

    ImVec2 previewSize(static_cast<float>(cachedWidth) * previewScale,
                       static_cast<float>(cachedHeight) * previewScale);

    // Solid color
    if (solidColorDescriptor != VK_NULL_HANDLE) {
        ImGui::Text("Solid (Scene Color)");
        ImGui::Image((ImTextureID)solidColorDescriptor, previewSize);
        ImGui::Separator();
    }

    // Solid depth
    if (solidDepthDescriptor != VK_NULL_HANDLE) {
        ImGui::Text("Solid (Depth Buffer)");
        ImGui::Image((ImTextureID)solidDepthDescriptor, previewSize);
        ImGui::Separator();
    }

    // Water depth
    if (waterColorDescriptor != VK_NULL_HANDLE) {
        ImGui::Text("Water (Depth Buffer)");
        ImGui::Image((ImTextureID)waterColorDescriptor, previewSize);
        ImGui::Separator();
    }

    // Shadow cascades (reuse ShadowRenderer's own ImGui descriptor sets)
    if (shadowMapper) {
        float shadowSize = static_cast<float>(shadowMapper->getShadowMapSize()) * previewScale;
        for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
            VkDescriptorSet ds = shadowMapper->getImGuiDescriptorSet(i);
            if (ds != VK_NULL_HANDLE) {
                ImGui::Text("Shadow Cascade %d", i);
                ImGui::Image((ImTextureID)ds, ImVec2(shadowSize, shadowSize));
                ImGui::Separator();
            }
        }
    }

    ImGui::End();
}
