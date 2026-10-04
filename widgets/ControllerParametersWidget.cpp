#include "ControllerParametersWidget.hpp"
#include <glm/gtc/type_ptr.hpp>

#include "components/ImGuiHelpers.hpp"

#include "../utils/Brush3dManager.hpp"
#include "types/BrushEntry.hpp"
#include "../events/ControllerContext.hpp"

ControllerParametersWidget::ControllerParametersWidget(ControllerManager* cm_, Brush3dManager* brushManager_)
    : Widget("Controller Parameters", u8"\uf085"), cm(cm_), brushManager(brushManager_) {}

// One collapsible/inline section per controller showing its active page,
// navigation buttons, and brush apply mode toggle (Click vs Drag).
// The contexts are independent, so each can be on a different page/subpage
// at the same time.
static void drawControllerSection(ControllerContext& ctx, const char* name,
                                   BrushApplyMode& brushMode) {
    // Namespace this section so the buttons ("Prev Page", etc.) get unique IDs
    // across the four controllers instead of colliding on their labels.
    ImGui::PushID(name);

    ImGui::Text("%s", name);

    ImGuiHelpers::CategoryBadge(ctx.activeCategory());
    ImGui::Text("  %s > %s", ctx.activePageName().c_str(), ctx.activeSubpageName().c_str());

    if (ImGui::Button("Prev Page"))  ctx.prevPage();
    ImGui::SameLine();
    if (ImGui::Button("Next Page"))  ctx.nextPage();
    if (ImGui::Button("Prev Sub"))   ctx.prevSubpage();
    ImGui::SameLine();
    if (ImGui::Button("Next Sub"))   ctx.nextSubpage();

    // Brush apply mode (Click / Drag) — shown inline per controller
    const char* items[] = { "Click", "Drag" };
    int current = (brushMode == BrushApplyMode::Drag) ? 1 : 0;
    ImGui::SameLine();
    if (ImGui::Combo("Apply", &current, items, IM_ARRAYSIZE(items))) {
        brushMode = (current == 1) ? BrushApplyMode::Drag : BrushApplyMode::Click;
    }

    ImGui::Separator();

    ImGui::PopID();
}

void ControllerParametersWidget::render() {
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;
    if (!cm) return;

    // All per-controller page trees (keyboard / mouse / gamepad / wiimote),
    // each with its own brush apply mode toggle shown inline.
    auto* params = cm->getParameters();
    drawControllerSection(cm->keyboardContext, "Keyboard", params->keyboardBrushMode);
    drawControllerSection(cm->mouseContext,    "Mouse",    params->mouseBrushMode);
    drawControllerSection(cm->gamepadContext,  "Gamepad",  params->gamepadBrushMode);
    drawControllerSection(cm->wiimoteContext,  "Wiimote",  params->wiimoteBrushMode);

    ImGui::Separator();

    ControllerContext& kb = cm->keyboardContext;

    if (kb.activeCategory() == PageCategory::CAMERA) {
        ImGui::DragFloat("Move Speed", &cm->getParameters()->cameraMoveSpeed, 0.1f, 0.0f, 1024.0f);
        ImGui::DragFloat("Angular Speed (deg/s)", &cm->getParameters()->cameraAngularSpeedDeg, 1.0f, 0.0f, 360.0f);
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
