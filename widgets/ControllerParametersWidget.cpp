#include "ControllerParametersWidget.hpp"
#include <glm/gtc/type_ptr.hpp>

#include "../utils/Brush3dManager.hpp"
#include "../utils/Brush3dEntry.hpp"

ControllerParametersWidget::ControllerParametersWidget(ControllerParameters* params, Brush3dManager* brushManager)
    : Widget("Controller Parameters"), params(params), brushManager(brushManager) {}

void ControllerParametersWidget::render() {
    if (!ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::End();
        return;
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

    ImGui::End();
}
