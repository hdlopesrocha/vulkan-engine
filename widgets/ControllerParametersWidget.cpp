#include "ControllerParametersWidget.hpp"
#include <glm/gtc/type_ptr.hpp>

ControllerParametersWidget::ControllerParametersWidget(ControllerParameters* params)
    : Widget("Controller Parameters"), params(params) {}

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

        case ControllerParameters::BRUSH_POSITION:
            ImGui::DragFloat3("Position", glm::value_ptr(params->brushPosition), 0.1f);
            break;

        case ControllerParameters::BRUSH_SCALE:
            ImGui::DragFloat3("Scale", glm::value_ptr(params->brushScale), 0.01f, 0.0f, 1024.0f);
            break;

        case ControllerParameters::BRUSH_ROTATION:
            ImGui::DragFloat3("Rotation (deg)", glm::value_ptr(params->brushRotation), 1.0f, -360.0f, 360.0f);
            break;

        case ControllerParameters::BRUSH_PROPERTIES:
            ImGui::Checkbox("Example Property", &params->brushPropertyExample);
            break;
    }

    ImGui::End();
}
