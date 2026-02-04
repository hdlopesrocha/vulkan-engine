#pragma once

#include "Widget.hpp"
#include "Settings.hpp"
#include <imgui.h>
#include <functional>

class SettingsWidget : public Widget {
public:
    explicit SettingsWidget(Settings& settings);
    
    void render() override;
    
    // Getters for the settings
    bool getShadowsEnabled() const { return settings.enableShadows; }
    bool getFlipKeyboardRotation() const { return settings.flipKeyboardRotation; }
    bool getFlipGamepadRotation() const { return settings.flipGamepadRotation; }
    float getMoveSpeed() const { return settings.moveSpeed; }
    float getAngularSpeedDeg() const { return settings.angularSpeedDeg; }
    bool getWireframeEnabled() const { return settings.wireframeMode; }
        int getDebugMode() const { return settings.debugMode; }
    bool getNormalMappingEnabled() const { return settings.normalMappingEnabled; }
    bool getWaterEnabled() const { return settings.waterEnabled; }
    bool getVegetationEnabled() const { return settings.vegetationEnabled; }
    float getTriplanarThreshold() const { return settings.triplanarThreshold; }
    float getTriplanarExponent() const { return settings.triplanarExponent; }
    bool getShowDebugCubes() const { return settings.showDebugCubes; }

    // Callback setter for debug actions
    void setDumpShadowDepthCallback(std::function<void()> cb) { onDumpShadowDepth = cb; }
    
private:
    Settings& settings;
    std::function<void()> onDumpShadowDepth;

    void resetToDefaults();
public:
    // Tessellation getters
    bool getTessellationEnabled() const { return settings.tessellationEnabled; }
    bool getShadowTessellationEnabled() const { return settings.shadowTessellationEnabled; }
    bool getAdaptiveTessellation() const { return settings.adaptiveTessellation; }
    float getTessMinLevel() const { return settings.tessMinLevel; }
    float getTessMaxLevel() const { return settings.tessMaxLevel; }
    float getTessMaxDistance() const { return settings.tessMaxDistance; }
    float getTessMinDistance() const { return settings.tessMinDistance; }
    
    // V-Sync getter
    bool getVSyncEnabled() const { return settings.vsyncEnabled; }
};
