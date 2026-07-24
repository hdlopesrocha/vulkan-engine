#include "ImpostorWidget.hpp"
#include <imgui.h>
#include <glm/trigonometric.hpp>
#include <cmath>

ImpostorWidget::ImpostorWidget(std::shared_ptr<ImpostorService> svc)
    : Widget("Impostor Viewer", u8"\uf06e"), impostorService(std::move(svc)) {}

void ImpostorWidget::render() {
    if (!isVisible()) return;

    ImGui::Begin(displayTitle().c_str(), &isOpen);

    const bool srcReady = impostorService && impostorService->isReady();

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
        const int   nameCount = 3;
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
        impostorService->captureAll(captureScale);
    }

    if (!impostorService->isReady()) {
        ImGui::Spacing();
        ImGui::TextDisabled("No impostor data captured yet.");
        ImGui::End();
        return;
    }

    // ── Preview mode ────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Captured views (%u, Fibonacci sphere):", ImpostorCapture::NUM_VIEWS);
    ImGui::SameLine();
    const char* modeNames[] = { "Albedo", "Normal", "Depth" };
    ImGui::SetNextItemWidth(100.0f);
    ImGui::Combo("##PreviewMode", &previewMode, modeNames, 3);
    ImGui::Spacing();

    const float thumbSize = 48.0f;
    const int   columns   = 5;
    for (uint32_t i = 0; i < ImpostorCapture::NUM_VIEWS; ++i) {
        if (i % columns != 0) ImGui::SameLine();
        ImTextureID tid = 0;
        if (previewMode == 0)
            tid = impostorService->getImTextureID(static_cast<uint32_t>(selectedBillboard), i);
        else if (previewMode == 1)
            tid = impostorService->getImGuiNormalTextureID(static_cast<uint32_t>(selectedBillboard), i);
        else
            tid = impostorService->getImGuiDepthTextureID(static_cast<uint32_t>(selectedBillboard), i);
        if (tid) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::Image(tid, ImVec2(thumbSize, thumbSize));
            if (ImGui::IsItemHovered()) {
                const glm::vec3& vd = impostorService->getViewDir(i);
                ImGui::SetTooltip("View %u\ndir=(%.2f, %.2f, %.2f)", i, vd.x, vd.y, vd.z);
            }
            ImGui::PopID();
        }
    }

    // ── Rotatable preview ────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Separator();
    const char* modeLabel = (previewMode == 0) ? "Albedo" : (previewMode == 1) ? "Normals" : "Depth";
    ImGui::Text("Preview  (drag to rotate)  [%s]:", modeLabel);
    ImGui::Spacing();

    const float cosP = std::cos(previewPitch);
    const glm::vec3 previewDir(cosP * std::sin(previewYaw),
                                std::sin(previewPitch),
                                cosP * std::cos(previewYaw));
    const uint32_t closestIdx = impostorService->closestView(glm::normalize(previewDir));

    ImTextureID previewTid = 0;
    if (previewMode == 0)
        previewTid = impostorService->getImTextureID(static_cast<uint32_t>(selectedBillboard), closestIdx);
    else if (previewMode == 1)
        previewTid = impostorService->getImGuiNormalTextureID(static_cast<uint32_t>(selectedBillboard), closestIdx);
    else
        previewTid = impostorService->getImGuiDepthTextureID(static_cast<uint32_t>(selectedBillboard), closestIdx);
    if (previewTid) {
        const float previewSize = 256.0f;
        ImGui::Image(previewTid, ImVec2(previewSize, previewSize));

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
            const glm::vec3& d = impostorService->getViewDir(closestIdx);
            ImGui::Text("Dir          : %.2f  %.2f  %.2f", d.x, d.y, d.z);
        }
        ImGui::EndGroup();
    }

    ImGui::End();
}
