#pragma once

#include "Widget.hpp"
#include "types/Settings.hpp"
#include <imgui.h>

class SettingsWidget : public Widget {
public:
    explicit SettingsWidget(Settings& settings);
    
    void render() override;
    
private:
    Settings& settings;

    void resetToDefaults();
};
