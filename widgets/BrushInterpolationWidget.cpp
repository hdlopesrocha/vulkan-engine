#include "BrushInterpolationWidget.hpp"
#include "Brush3dWidget.hpp" // shared sdfTypeNames
#include "components/ImGuiHelpers.hpp"
#include "../math/geometry/Transformation.hpp"

#include <imgui.h>
#include <cstdio>

BrushInterpolationWidget::BrushInterpolationWidget(Brush3dManager& mgr)
    : Widget("Brush Target SDF", u8"\uf1e0"), manager(mgr) {}

void BrushInterpolationWidget::render() {
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    BrushEntry* source = manager.getSelectedEntry();
    if (!source) {
        ImGui::TextDisabled("No brush entry selected.");
        return;
    }
    if (!source->useInterpolation) {
        ImGui::TextDisabled(
            "Enable 'Interpolate with target SDF' on the current brush entry "
            "to edit its target SDF here.");
        return;
    }

    BrushSdfTarget& target = source->interpolationTarget;

    ImGui::Text("Source: %s", Brush3dWidget::sdfTypeNames[
        source->sdfType >= 0 && source->sdfType < Brush3dWidget::kSdfTypeCount
            ? source->sdfType : 0]);
    ImGui::SameLine();
    if (ImGui::Button("Copy Source SDF -> Target")) {
        target = brushTargetFromEntry(*source);
    }
    ImGui::SliderFloat("Interpolation (source -> target)",
                       &source->interpolationFactor, 0.0f, 1.0f, "%.3f");
    ImGui::Separator();

    int sdfType = target.sdfType;
    if (ImGui::Combo("Target SDF Type", &sdfType,
                     Brush3dWidget::sdfTypeNames,
                     Brush3dWidget::kSdfTypeCount)) {
        target.sdfType = sdfType;
    }

    ImGui::SeparatorText("Target Transform");
    ImGui::DragFloat3("Translate##target", &target.translate.x, 1.0f,
                      -100000.0f, 100000.0f);
    ImGui::DragFloat3("Scale##target", &target.scale.x, 1.0f, 0.1f, 100000.0f);
    bool rotationChanged = false;
    rotationChanged |= ImGui::DragFloat("Yaw##target", &target.yaw, 1.0f, -360.0f, 360.0f);
    rotationChanged |= ImGui::DragFloat("Pitch##target", &target.pitch, 1.0f, -360.0f, 360.0f);
    rotationChanged |= ImGui::DragFloat("Roll##target", &target.roll, 1.0f, -360.0f, 360.0f);
    if (rotationChanged) {
        target.rot = Transformation::getRotation(target.yaw, target.pitch, target.roll);
    }
    ImGui::DragFloat("Target Min Size", &target.minSize, 1.0f, 0.1f, 1000.0f);

    ImGui::SeparatorText("Target Primitive");
    if (target.sdfType == 2) { // Capsule
        ImGui::DragFloat3("Point A##target", &target.capsuleA.x, 1.0f);
        ImGui::DragFloat3("Point B##target", &target.capsuleB.x, 1.0f);
        ImGui::DragFloat("Radius##target", &target.capsuleRadius, 0.1f, 0.01f, 10000.0f);
    } else if (target.sdfType == 5) { // Torus
        ImGui::DragFloat("Major Radius##target", &target.torusRadii.x, 0.01f, 0.01f, 1.0f);
        ImGui::DragFloat("Minor Radius##target", &target.torusRadii.y, 0.01f, 0.01f, 1.0f);
    } else if (target.sdfType == 8) { // Tapered cylinder
        ImGui::DragFloat("Bottom Radius##target", &target.taperedCylinderRadii.x,
                         0.01f, 0.01f, 10.0f);
        ImGui::DragFloat("Top Radius##target", &target.taperedCylinderRadii.y,
                         0.01f, 0.01f, 10.0f);
    } else if (target.sdfType == 9) { // Tapered capsule
        ImGui::DragFloat3("Point A##target", &target.capsuleA.x, 1.0f);
        ImGui::DragFloat3("Point B##target", &target.capsuleB.x, 1.0f);
        ImGui::DragFloat("Radius A##target", &target.taperedCapsuleRadii.x,
                         0.01f, 0.01f, 10.0f);
        ImGui::DragFloat("Radius B##target", &target.taperedCapsuleRadii.y,
                         0.01f, 0.01f, 10.0f);
    } else if (target.sdfType == 10) { // SVG extrusion
        char pathBuf[512];
        std::snprintf(pathBuf, sizeof(pathBuf), "%s", target.svgPath.c_str());
        if (ImGui::InputText("SVG File##target", pathBuf, sizeof(pathBuf))) {
            target.svgPath = pathBuf;
        }
        ImGui::Checkbox("Tint by SVG fill colours##target", &target.svgTintFromFill);
        ImGui::TextDisabled("Profile in local XZ; extruded along Y, clipped to Scale.");
    } else {
        ImGui::TextDisabled("No extra primitive parameters for this SDF type.");
    }

    ImGui::SeparatorText("Target Sweep");
    if (ImGui::Checkbox("Sweep Mode##target", &target.sweepMode)) {
        if (target.sweepMode) target.previousTranslate = target.translate;
    }
    if (target.sweepMode) {
        ImGui::Text("Prev: %.1f, %.1f, %.1f -> Cur: %.1f, %.1f, %.1f",
                    target.previousTranslate.x, target.previousTranslate.y,
                    target.previousTranslate.z, target.translate.x,
                    target.translate.y, target.translate.z);
    }

    ImGui::Separator();
    ImGui::TextDisabled(
        "Interpolation mixes the two raw SDFs. The source brush's effect is "
        "applied after the mix; target effects are not used.");
}
