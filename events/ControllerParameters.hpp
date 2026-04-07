#pragma once

#include <glm/glm.hpp>

class ControllerParameters {
public:
    enum pageType { CAMERA = 0, BRUSH_POSITION = 1, BRUSH_SCALE = 2, BRUSH_ROTATION = 3, BRUSH_PROPERTIES = 4 };

    ControllerParameters() = default;

    pageType currentPage = CAMERA;

    // Camera-related parameters
    float cameraMoveSpeed = 8.0f;
    float cameraAngularSpeedDeg = 90.0f;

    static inline const char* pageTypeName(pageType p) {
        switch (p) {
            case CAMERA: return "Camera";
            case BRUSH_POSITION: return "Brush Position";
            case BRUSH_SCALE: return "Brush Scale";
            case BRUSH_ROTATION: return "Brush Rotation";
            case BRUSH_PROPERTIES: return "Brush Properties";
            default: return "Unknown";
        }
    }
};
