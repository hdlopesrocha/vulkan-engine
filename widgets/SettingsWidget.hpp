#pragma once

#include "Widget.hpp"
#include "../utils/Settings.hpp"
#include "../utils/ShadowParams.hpp"
#include <imgui.h>

class SettingsWidget : public Widget {
public:
    explicit SettingsWidget(Settings& settings, ShadowParams* shadowParams_ = nullptr);
    
    void render() override;
    
private:
    Settings& settings;
    ShadowParams* shadowParams;

    void resetToDefaults();
};
