#include "ImpostorWidget.hpp"
#include "../vulkan/VulkanApp.hpp"
#include "../vulkan/renderer/VegetationRenderer.hpp"
#include <imgui.h>
#include <glm/trigonometric.hpp>
#include <cmath>
#include <cstdio>

ImpostorWidget::ImpostorWidget()
    : Widget("Impostor Viewer", u8"\uf06e") {}

void ImpostorWidget::setVulkanApp(VulkanApp* app) {
    vulkanApp = app;
    capture.init(app);
}

void ImpostorWidget::setSource(VkImageView albedo, VkImageView normal,
                                VkImageView opacity, VkSampler sampler,
                                int numBillboards) {
    srcAlbedo      = albedo;
    srcNormal      = normal;
    srcOpacity     = opacity;
    srcSampler     = sampler;
    billboardCount = numBillboards;
    if (selectedBillboard >= billboardCount)
        selectedBillboard = 0;

    // Auto-capture all billboard types so impostors work without user interaction.
    if (vulkanApp && srcAlbedo != VK_NULL_HANDLE && srcSampler != VK_NULL_HANDLE) {
        capture.captureAll(vulkanApp,
                           srcAlbedo, srcNormal, srcOpacity, srcSampler,
                           captureScale);
        if (vegRenderer && capture.getCaptureArrayView() != VK_NULL_HANDLE) {
            vegRenderer->setImpostorData(vulkanApp,
                                         capture.getCaptureArrayView(),
                                         capture.getCaptureNormalArrayView(),
                                         capture.getCaptureArraySampler());
        }
    }
}

void ImpostorWidget::rewire() {
    if (vulkanApp && vegRenderer && capture.isReady() &&
        capture.getCaptureArrayView() != VK_NULL_HANDLE) {
        vegRenderer->setImpostorData(vulkanApp,
                                     capture.getCaptureArrayView(),
                                     capture.getCaptureNormalArrayView(),
                                     capture.getCaptureArraySampler());
    }
}

void ImpostorWidget::cleanup() {
    capture.cleanup(vulkanApp);
}

void ImpostorWidget::render() {
    if (!isVisible()) return;

    ImGui::Begin(displayTitle().c_str(), &isOpen);

    // ── Source selection ─────────────────────────────────────────────
    const bool srcReady = (srcAlbedo  != VK_NULL_HANDLE &&
                           srcNormal  != VK_NULL_HANDLE &&
                           srcOpacity != VK_NULL_HANDLE &&
                           srcSampler != VK_NULL_HANDLE &&
                           billboardCount > 0);

    if (!srcReady) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                           "Bake billboards first, then open this widget.");
        ImGui::End();
        return;
    }

    // Billboard index selector
    {
        ImGui::Text("Billboard:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(140.0f);
        const char* names[] = { "Foliage (0)", "Grass (1)", "Wild (2)" };
        const int   nameCount = std::min(billboardCount, 3);
        if (selectedBillboard >= nameCount) selectedBillboard = 0;
        ImGui::Combo("##BillboardSel", &selectedBillboard, names, nameCount);
    }

    ImGui::Spacing();

    // Scale
    ImGui::Text("Capture scale:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::DragFloat("##Scale", &captureScale, 0.5f, 1.0f, 200.0f, "%.1f");

    ImGui::Spacing();

    if (ImGui::Button("Capture Impostor Views (All Types)", ImVec2(-1, 0))) {
        capture.captureAll(vulkanApp,
                           srcAlbedo, srcNormal, srcOpacity, srcSampler,
                           captureScale);
        // Wire captured arrays to VegetationRenderer if available.
        if (vegRenderer && capture.getCaptureArrayView() != VK_NULL_HANDLE) {
            vegRenderer->setImpostorData(vulkanApp,
                                         capture.getCaptureArrayView(),
                                         capture.getCaptureNormalArrayView(),
                                         capture.getCaptureArraySampler());
        }
    }

    if (!capture.isReady()) {
        ImGui::Spacing();
        ImGui::TextDisabled("No impostor data captured yet.");
        ImGui::End();
        return;
    }

    // ── Thumbnail grid ───────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Captured views (%u, Fibonacci sphere):", ImpostorCapture::NUM_VIEWS);
    ImGui::SameLine();
    ImGui::Checkbox("Show normals", &previewNormals);
    ImGui::Spacing();

    const float thumbSize = 48.0f;
    const int   columns   = 5;
    for (uint32_t i = 0; i < ImpostorCapture::NUM_VIEWS; ++i) {
        if (i % columns != 0) ImGui::SameLine();
        VkDescriptorSet ds = previewNormals
            ? capture.getImGuiNormalDescSet(static_cast<uint32_t>(selectedBillboard), i)
            : capture.getImGuiDescSet(static_cast<uint32_t>(selectedBillboard), i);
        if (ds != VK_NULL_HANDLE) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::Image((ImTextureID)ds, ImVec2(thumbSize, thumbSize));
            if (ImGui::IsItemHovered()) {
                const glm::vec3& vd = capture.getViewDir(i);
                ImGui::SetTooltip("View %u\ndir=(%.2f, %.2f, %.2f)", i, vd.x, vd.y, vd.z);
            }
            ImGui::PopID();
        }
    }

    // ── Rotatable preview ────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Preview  (drag to rotate)  [%s]:", previewNormals ? "Normals" : "Albedo");
    ImGui::Spacing();

    const float cosP = std::cos(previewPitch);
    const glm::vec3 previewDir(cosP * std::sin(previewYaw),
                                std::sin(previewPitch),
                                cosP * std::cos(previewYaw));
    const uint32_t closestIdx = capture.closestView(glm::normalize(previewDir));

    VkDescriptorSet previewDs = previewNormals
        ? capture.getImGuiNormalDescSet(static_cast<uint32_t>(selectedBillboard), closestIdx)
        : capture.getImGuiDescSet(static_cast<uint32_t>(selectedBillboard), closestIdx);
    if (previewDs != VK_NULL_HANDLE) {
        const float previewSize = 256.0f;
        ImGui::Image((ImTextureID)previewDs, ImVec2(previewSize, previewSize));

        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            const ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
            const float sensitivity = 0.005f;
            previewYaw   += delta.x * sensitivity;
            previewPitch -= delta.y * sensitivity;
            const float kMax = 1.5533f; // ~89°
            if (previewPitch >  kMax) previewPitch =  kMax;
            if (previewPitch < -kMax) previewPitch = -kMax;
        }

        ImGui::SameLine();
        ImGui::BeginGroup();
        ImGui::Text("Closest view : %u",       closestIdx);
        ImGui::Text("Yaw          : %.1f deg", glm::degrees(previewYaw));
        ImGui::Text("Pitch        : %.1f deg", glm::degrees(previewPitch));
        {
            const glm::vec3& d = capture.getViewDir(closestIdx);
            ImGui::Text("Dir          : %.2f  %.2f  %.2f", d.x, d.y, d.z);
        }
        ImGui::EndGroup();
    }

    ImGui::End();
}
