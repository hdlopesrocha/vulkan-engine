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
    float getTriplanarThreshold() const { return triplanarThreshold; }
    float getTriplanarExponent() const { return triplanarExponent; }

    // Callback setter for debug actions
    void setDumpShadowDepthCallback(std::function<void()> cb) { onDumpShadowDepth = cb; }
    
private:
    bool enableShadows = true;                 // Global toggle for shadow mapping
    
    bool flipKeyboardRotation = false;         // Flip keyboard rotation axes
    bool flipGamepadRotation = false;          // Flip gamepad rotation axes
    float moveSpeed = 2.5f;                    // movement units/sec
    float angularSpeedDeg = 45.0f;             // degrees/sec for rotation
    bool wireframeMode = false;                // render wireframe when true
        int debugMode = 0;                         // 0=Default,1=Fragment Normal,2=World Normal,3=UV,4=Normal(TBN),5=Albedo,6=Normal Tex,7=Bump,8=Pre-Projection,9=Normal from Derivatives,10=Light Vector,11=N·L,12=Shadow Diagnostics,13=Triplanar Weights
    bool normalMappingEnabled = true;          // Global toggle for normal mapping
    std::function<void()> onDumpShadowDepth;
    // Triplanar settings
    float triplanarThreshold = 0.5f; // small dead zone before blending starts
    float triplanarExponent = 1.0f;   // >1 makes transitions steeper
    // Tessellation settings
    bool tessellationEnabled = true; // global toggle to disable all tessellation + displacement
    bool shadowTessellationEnabled = true; // global toggle to disable all tessellation + displacement
    bool adaptiveTessellation = true;
    float tessMinLevel = 1.0f;
    float tessMaxLevel = 32.0f;
    float tessMaxDistance = 30.0f;
    float tessMinDistance = 10.0f;
    
    void resetToDefaults();
public:
    // Tessellation getters
    bool getTessellationEnabled() const { return tessellationEnabled; }
    bool getShadowTessellationEnabled() const { return shadowTessellationEnabled; }
    bool getAdaptiveTessellation() const { return adaptiveTessellation; }
    float getTessMinLevel() const { return tessMinLevel; }
    float getTessMaxLevel() const { return tessMaxLevel; }
    float getTessMaxDistance() const { return tessMaxDistance; }
    float getTessMinDistance() const { return tessMinDistance; }
};
