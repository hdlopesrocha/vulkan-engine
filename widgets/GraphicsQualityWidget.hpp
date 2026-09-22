#pragma once

#include "Widget.hpp"

class EventManager;

// Right-edge main-UI overlay with one-click graphics-quality presets. The
// widget only queues SetGraphicsQualityEvent; MyApp executes the matching
// GraphicsSettingsCommand (global Settings edits only, per-layer water params
// stay authored), so the widget stays free of renderer state.
class GraphicsQualityWidget : public Widget {
public:
    explicit GraphicsQualityWidget(EventManager* eventManager);

    void render() override;

private:
    EventManager* eventManager = nullptr;
};
