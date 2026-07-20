#include "ControllerParametersWidget.hpp"
#include <glm/gtc/type_ptr.hpp>

#include "components/ImGuiHelpers.hpp"

#include "../utils/Brush3dManager.hpp"
#include "../utils/Brush3dEntry.hpp"
#include "../events/ControllerContext.hpp"

ControllerParametersWidget::ControllerParametersWidget(ControllerManager* cm_, Brush3dManager* brushManager_)
    : Widget("Controller Parameters", u8"\uf085"), cm(cm_), brushManager(brushManager_) {}

// Draw the active page indicator for a given controller context.
static void drawContextPage(const ControllerContext& ctx, const ImVec4& col, const char* label) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(36, 0));
    ImVec2 center = ImVec2(pos.x + 10.0f, pos.y + 10.0f);
    dl->AddCircleFilled(center, 8.0f, ImGui::ColorConvertFloat4ToU32(col));
    dl->AddText(ImVec2(pos.x + 22.0f, pos.y), ImGui::ColorConvertFloat4ToU32(ImVec4(1,1,1,1)), label);
}

// One collapsible/inline section per controller showing its active page and
// navigation buttons. The contexts are independent, so each can be on a
// different page/subpage at the same time.
static void drawControllerSection(ControllerContext& ctx, const char* name) {
    // Namespace this section so the buttons ("Prev Page", etc.) get unique IDs
    // across the four controllers instead of colliding on their labels.
    ImGui::PushID(name);

    ImGui::Text("%s", name);

    ImVec4 col = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
    const char* label = "?";
    switch (ctx.activeCategory()) {
        case PageCategory::CAMERA: label = "CAM"; col = ImVec4(0.2f,0.5f,0.9f,1.0f); break;
        case PageCategory::BRUSH:  label = "BRU"; col = ImVec4(0.2f,0.9f,0.3f,1.0f); break;
    }
    drawContextPage(ctx, col, label);
    ImGui::Text("  %s > %s", ctx.activePageName().c_str(), ctx.activeSubpageName().c_str());

    if (ImGui::Button("Prev Page"))  ctx.prevPage();
    ImGui::SameLine();
    if (ImGui::Button("Next Page"))  ctx.nextPage();
    if (ImGui::Button("Prev Sub"))   ctx.prevSubpage();
    ImGui::SameLine();
    if (ImGui::Button("Next Sub"))   ctx.nextSubpage();
    ImGui::Separator();

    ImGui::PopID();
}

void ControllerParametersWidget::render() {
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;
    if (!cm) return;

    // All per-controller page trees (keyboard / mouse / gamepad / wiimote).
    drawControllerSection(cm->keyboardContext, "Keyboard");
    drawControllerSection(cm->mouseContext,    "Mouse");
    drawControllerSection(cm->gamepadContext,  "Gamepad");
    drawControllerSection(cm->wiimoteContext,  "Wiimote");

    ImGui::Separator();

    ControllerContext& kb = cm->keyboardContext;
    ControllerParameters& params = *cm->getParameters();

    if (kb.activeCategory() == PageCategory::CAMERA) {
        ImGui::DragFloat("Move Speed", &params.cameraMoveSpeed, 0.1f, 0.0f, 1024.0f);
        ImGui::DragFloat("Angular Speed (deg/s)", &params.cameraAngularSpeedDeg, 1.0f, 0.0f, 360.0f);
    } else {
        // Brush attribute editing for the selected entry.
        if (!brushManager) { ImGui::Text("No Brush Manager available"); return; }
        BrushEntry* be = brushManager->getSelectedEntry();
        if (!be) { ImGui::Text("No brush selected"); return; }
        const PageControl ctrl = kb.activeControl();
        switch (ctrl) {
            case PageControl::TRANSLATE: // Transform subpage: translate+rotate+scale
                ImGui::DragFloat3("Position", glm::value_ptr(be->translate), 0.1f);
                ImGui::DragFloat("Yaw", &be->yaw, 1.0f, -360.0f, 360.0f);
                ImGui::DragFloat("Pitch", &be->pitch, 1.0f, -360.0f, 360.0f);
                ImGui::DragFloat("Roll", &be->roll, 1.0f, -360.0f, 360.0f);
                ImGui::DragFloat3("Scale", glm::value_ptr(be->scale), 0.01f, 0.0f, 1024.0f);
                break;
            case PageControl::TEXTURE:
                ImGui::DragInt("Material Index", &be->materialIndex, 1.0f, 0, 63);
                break;
            case PageControl::ATTRIBUTE: {
                const char* types[] = {"Sphere","Box","Capsule","Octahedron","Pyramid","Torus","Cone","Cylinder"};
                int t = be->sdfType;
                if (ImGui::Combo("SDF Type", &t, types, IM_ARRAYSIZE(types))) be->sdfType = t;
                ImGui::Checkbox("Use Effect", &be->useEffect);
                break;
            }
            case PageControl::UI:
            default:
                ImGui::Text("UI page (non-propagating)");
                break;
        }
    }
}
