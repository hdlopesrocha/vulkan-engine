#pragma once

#include "Widget.hpp"
#include <imgui.h>
#include <functional>

class SettingsWidget : public Widget {
public:
    SettingsWidget();
    
    void render() override;
    
    // Getters for the settings
    bool getShadowsEnabled() const { return enableShadows; }
    bool getFlipKeyboardRotation() const { return flipKeyboardRotation; }
    bool getFlipGamepadRotation() const { return flipGamepadRotation; }
    float getMoveSpeed() const { return moveSpeed; }
    float getAngularSpeedDeg() const { return angularSpeedDeg; }
    bool getWireframeEnabled() const { return wireframeMode; }
        int getDebugMode() const { return debugMode; }
    bool getNormalMappingEnabled() const { return normalMappingEnabled; }

    // Callback setter for debug actions
    void setDumpShadowDepthCallback(std::function<void()> cb) { onDumpShadowDepth = cb; }
    
private:
    bool enableShadows = true;                 // Global toggle for shadow mapping
    
    bool flipKeyboardRotation = false;         // Flip keyboard rotation axes
    bool flipGamepadRotation = false;          // Flip gamepad rotation axes
    float moveSpeed = 2.5f;                    // movement units/sec
    float angularSpeedDeg = 45.0f;             // degrees/sec for rotation
    bool wireframeMode = false;                // render wireframe when true
        int debugMode = 0;                         // 0=Default,1=Fragment Normal,2=World Normal,3=UV,4=Tangent,5=Bitangent,6=Normal (TBN),7=Albedo,8=Normal Tex,9=Bump,10=Pre-Projection,11=Normal from Derivatives,12=Light Vector,13=N·L,14=Shadow Diagnostics,15=Triplanar Weights
    bool normalMappingEnabled = true;          // Global toggle for normal mapping
    std::function<void()> onDumpShadowDepth;
    // Tessellation settings
    bool adaptiveTessellation = true;
    float tessMinLevel = 1.0f;
    float tessMaxLevel = 32.0f;
    float tessMaxDistance = 30.0f;
    
    void resetToDefaults();
public:
    // Tessellation getters
    bool getAdaptiveTessellation() const { return adaptiveTessellation; }
    float getTessMinLevel() const { return tessMinLevel; }
    float getTessMaxLevel() const { return tessMaxLevel; }
    float getTessMaxDistance() const { return tessMaxDistance; }
};
