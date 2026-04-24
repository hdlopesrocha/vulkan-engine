#include "ControllerParametersWidget.hpp"
#include <glm/gtc/type_ptr.hpp>

#include "components/ImGuiHelpers.hpp"

#include "../utils/Brush3dManager.hpp"
#include "../utils/Brush3dEntry.hpp"

ControllerParametersWidget::ControllerParametersWidget(ControllerParameters* params, Brush3dManager* brushManager)
    : Widget("Controller Parameters", u8"\uf085"), params(params), brushManager(brushManager) {}

void ControllerParametersWidget::render() {
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;
    // Draw small icon for current page at the top-left of the widget
    {
        ControllerParameters::pageType p = params->currentPage;
        const char* label = "?";
        ImVec4 col = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
        switch (p) {
            case ControllerParameters::CAMERA: label = "CAM"; col = ImVec4(0.2f,0.5f,0.9f,1.0f); break;
            case ControllerParameters::BRUSH_POSITION: label = "POS"; col = ImVec4(0.2f,0.9f,0.3f,1.0f); break;
            case ControllerParameters::BRUSH_SCALE: label = "SCL"; col = ImVec4(0.95f,0.6f,0.1f,1.0f); break;
            case ControllerParameters::BRUSH_ROTATION: label = "ROT"; col = ImVec4(0.7f,0.3f,0.9f,1.0f); break;
            case ControllerParameters::BRUSH_PROPERTIES: label = "PRP"; col = ImVec4(0.6f,0.6f,0.6f,1.0f); break;
        }
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(36, 0));
        ImVec2 center = ImVec2(pos.x + 10.0f, pos.y + 10.0f);
        dl->AddCircleFilled(center, 8.0f, ImGui::ColorConvertFloat4ToU32(col));
        dl->AddText(ImVec2(pos.x + 22.0f, pos.y), ImGui::ColorConvertFloat4ToU32(ImVec4(1,1,1,1)), label);
    }

    int current = static_cast<int>(params->currentPage);
    const char* items[] = {
        ControllerParameters::pageTypeName(ControllerParameters::CAMERA),
        ControllerParameters::pageTypeName(ControllerParameters::BRUSH_POSITION),
        ControllerParameters::pageTypeName(ControllerParameters::BRUSH_SCALE),
        ControllerParameters::pageTypeName(ControllerParameters::BRUSH_ROTATION),
        ControllerParameters::pageTypeName(ControllerParameters::BRUSH_PROPERTIES)
    };

    if (ImGui::Combo("Page", &current, items, IM_ARRAYSIZE(items))) {
        params->currentPage = static_cast<ControllerParameters::pageType>(current);
    }

    switch (params->currentPage) {
        case ControllerParameters::CAMERA:
            ImGui::DragFloat("Move Speed", &params->cameraMoveSpeed, 0.1f, 0.0f, 1024.0f);
            ImGui::DragFloat("Angular Speed (deg/s)", &params->cameraAngularSpeedDeg, 1.0f, 0.0f, 360.0f);
            break;

        case ControllerParameters::BRUSH_POSITION: {
            if (!brushManager) {
                ImGui::Text("No Brush Manager available");
                break;
            }
            BrushEntry* be = brushManager->getSelectedEntry();
            if (!be) {
                ImGui::Text("No brush selected");
                break;
            }
            ImGui::DragFloat3("Position", glm::value_ptr(be->translate), 0.1f);
            break;
        }

        case ControllerParameters::BRUSH_SCALE: {
            if (!brushManager) { ImGui::Text("No Brush Manager available"); break; }
            BrushEntry* be = brushManager->getSelectedEntry();
            if (!be) { ImGui::Text("No brush selected"); break; }
            ImGui::DragFloat3("Scale", glm::value_ptr(be->scale), 0.01f, 0.0f, 1024.0f);
            break;
        }

        case ControllerParameters::BRUSH_ROTATION: {
            if (!brushManager) { ImGui::Text("No Brush Manager available"); break; }
            BrushEntry* be = brushManager->getSelectedEntry();
            if (!be) { ImGui::Text("No brush selected"); break; }
            float rot[3] = { be->yaw, be->pitch, be->roll };
            if (ImGui::DragFloat3("Rotation (deg)", rot, 1.0f, -360.0f, 360.0f)) {
                be->yaw = rot[0]; be->pitch = rot[1]; be->roll = rot[2];
            }
            break;
        }

        case ControllerParameters::BRUSH_PROPERTIES: {
            if (!brushManager) { ImGui::Text("No Brush Manager available"); break; }
            BrushEntry* be = brushManager->getSelectedEntry();
            if (!be) { ImGui::Text("No brush selected"); break; }
            ImGui::Checkbox("Use Effect", &be->useEffect);
            break;
        }
    }

    
}
