#pragma once

class Settings {
public:
    Settings() { resetToDefaults(); }

    void resetToDefaults() {
        enableShadows = true;
        flipKeyboardRotation = false;
        flipGamepadRotation = false;
        moveSpeed = 2.5f;
        angularSpeedDeg = 45.0f;
        wireframeMode = false;
        debugMode = 0;
        normalMappingEnabled = true;
        waterEnabled = false;
        vegetationEnabled = false;
        triplanarThreshold = 0.12f;
        triplanarExponent = 1.0f;
        tessellationEnabled = true;
        shadowTessellationEnabled = true;
        adaptiveTessellation = true;
        tessMinLevel = 1.0f;
        tessMaxLevel = 32.0f;
        tessMaxDistance = 30.0f;
        tessMinDistance = 10.0f;
        vsyncEnabled = true;
    }

    // Global toggles
    bool enableShadows = true;
    bool waterEnabled = true;
    bool vegetationEnabled = true;
    bool wireframeMode = false;
    bool normalMappingEnabled = true;

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
    bool tessellationEnabled = true;
    bool shadowTessellationEnabled = true;
    bool adaptiveTessellation = true;
    float tessMinLevel = 1.0f;
    float tessMaxLevel = 32.0f;
    float tessMaxDistance = 30.0f;
    float tessMinDistance = 10.0f;

    // Present mode
    bool vsyncEnabled = true;
};
