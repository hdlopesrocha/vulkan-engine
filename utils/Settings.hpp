#pragma once

class Settings {
public:
    void resetToDefaults() {
        *this = Settings{};
    }

    // Global toggles
    bool enableShadows = true;
    // Toggle rendering of the main solid scene (terrain/meshes)
    bool renderSolid = true;
    bool waterEnabled = true;
    bool vegetationEnabled = true;
    bool wireframeMode = false;
    bool waterWireframeMode = false;
    bool normalMappingEnabled = true;
    bool roughnessEnabled = true;
    bool aoEnabled = true;

    // Debug visuals
    bool showDebugCubes = false;
    bool showBoundingBoxes = false;
    bool showSDFDebug = false;

    // Input settings
    bool flipKeyboardRotation = false;
    bool flipGamepadRotation = false;
    float moveSpeed = 2.5f;
    float angularSpeedDeg = 45.0f;

    // Debug
    int debugMode = 0;

    // Triplanar
    float triplanarThreshold = 0.12f;
    float triplanarExponent = 1.0f;

    // Tessellation
    bool tessellationEnabled = false;
    bool shadowTessellationEnabled = false;
    bool adaptiveTessellation = true;
    float tessellationFactor = 1.0f;
    float tessMaxDistance = 512.0f;
    float tessMinDistance = 1.0f;

    // Present mode
    bool vsyncEnabled = true;

    // Camera clip planes
    float nearPlane = 0.1f;
    float farPlane = 8092.0f;

    // Impostor rendering: vegetation beyond this distance is drawn as a pre-captured
    // camera-facing quad.  Set to 0 to disable (default: disabled).
    float impostorDistance = 100.0f;
};
